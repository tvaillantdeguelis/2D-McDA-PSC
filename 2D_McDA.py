#!/usr/bin/env python
# coding: utf8

"""Main program of 2D-McDA. Takes granule to process as input."""

__author__     = "Thibault Vaillant de Guélis"
__version__    = "1.01"
__email__      = "thibault.vaillantdeguelis@outlook.com"
__status__     = "Prototype"

import sys
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt 

from my_modules.standard_outputs import print_time, print_elapsed_time
from my_modules.readers.calipso_reader import CALIOPReader, automatic_path_detection, get_first_profileID_of_chunk, range_from_altitude
from my_modules.paths import split_granule_date
from my_modules.calipso_constants import *
from my_modules.writers.hdf_writer import SDSData, write_hdf
from my_modules.calipso_calculator import compute_par_ab532, compute_ab_mol_and_b_mol, \
    nsf_from_V_domain_to_betap_domain, rms_from_P_domain_to_betap_domain, compute_shotnoise, \
    compute_backgroundnoise

from config import NB_PROF_OVERLAP
from feature_detection import detect_features, separate_homogeneous_features, classify_homogeneous_features_with_psc_v2
from merged_3channels_feature_mask import merged_feature_masks


def get_start_end_indexes(prof_min, prof_max, nb_prof_slice, nb_prof_overlap):
    """Compute start and end indexes of each slice of profiles to process
        Outputs: index_start_slice_array = array of slice start profile indexes
                 index_end_slice_array = array of slice end profile indexes"""
    
    nb_prof = prof_max - prof_min + 1
    
    # If total number of profiles to process less than slice size
    if nb_prof <= nb_prof_slice:
        # Put prof_min as start index
        index_start_slice_array = np.array((prof_min,))
        # Put prof_max as end index
        index_end_slice_array = np.array((prof_max,))

    else:
        # Create array of start indexes
        # Note: if last slice size < nb_prof_slice/2, then it is merge 
        # with previous
        index_start_slice_array = np.arange(prof_min, 
                                            prof_max - int(nb_prof_slice/2.) + 2, 
                                            nb_prof_slice-nb_prof_overlap)
        # Create array of end indexes
        index_end_slice_array = index_start_slice_array + nb_prof_slice - 1
        index_end_slice_array[-1] = prof_max

    return index_start_slice_array, index_end_slice_array


def rm_prof(array, nb_prof_to_remove, side):
    """Remove first profiles from previous file added to allow window 
       image technique at the edges"""
    
    if side == 'start':
        if array.ndim == 1:
            array = array[nb_prof_to_remove:]
        elif array.ndim == 2:
            array = array[nb_prof_to_remove:, :]
        elif array.ndim == 3:
            array = array[:, nb_prof_to_remove:, :]
        else:
            sys.exit('Error: ndim array unknown')
    elif side == 'end':
        if array.ndim == 1:
            array = array[:-nb_prof_to_remove]
        elif array.ndim == 2:
            array = array[:-nb_prof_to_remove, :]
        elif array.ndim == 3:
            array = array[:, :-nb_prof_to_remove, :]
        else:
            sys.exit('Error: ndim array unknown')
    else:
        sys.exit('Error: side unknown')

    return array


def compute_uncertainty(nb_bins_shift, mol_ab, rms, nsf):

    # Definition
    FCORR = np.array((1.573, 1.345, 1.188, 1.131, 1.188, 1.345, 1.573, 1.345, 1.188, 1.131))
    NB_PIXELS = 15*12 # 5-km horizontal × 180-m vertical resolution

    nb_bins_shift_abs = np.squeeze(np.abs(nb_bins_shift))
    background_noise = np.ma.copy(rms)
    shot_noise = nsf * np.sqrt(mol_ab)

    ab_std = FCORR[nb_bins_shift_abs][:, np.newaxis] * 1/np.sqrt(NB_PIXELS) * np.sqrt(background_noise**2 + shot_noise**2)

    return ab_std, background_noise, shot_noise


if __name__ == '__main__':
    tic_main_program = print_time()
    
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    # PARAMETERS
    if len(sys.argv) > 1:
        GRANULE_DATE = sys.argv[1]
        VERSION_CAL_LID_L1 = sys.argv[2]
        TYPE_CAL_LID_L1 = sys.argv[3]
        PREVIOUS_GRANULE = None if sys.argv[4] == 'None' else sys.argv[4]
        NEXT_GRANULE = None if sys.argv[5] == 'None' else sys.argv[5]
        SLICE_START_END_TYPE = sys.argv[6] # 'profindex' or 'longitude'
        SLICE_START = None if sys.argv[7] == 'None' else float(sys.argv[7])
        SLICE_END = None if sys.argv[8] == 'None' else float(sys.argv[8])
        SAVE_DEVELOPMENT_DATA = sys.argv[9] == 'True'
        VERSION_2D_McDA = sys.argv[10]
        TYPE_2D_McDA = sys.argv[11]
        OUT_FOLDER = sys.argv[12]
    else:
        GRANULE_DATE = "2010-01-18T00-19-57ZN"
        VERSION_CAL_LID_L1 = "V4.51"
        TYPE_CAL_LID_L1 = "Standard"
        PREVIOUS_GRANULE = None
        NEXT_GRANULE = None
        SLICE_START_END_TYPE = "longitude" # "profindex" or "longitude" (Use "profindex" if SLICE_START/END = None to process the whole granule)
        SLICE_START = 170.68 # profindex or longitude
        SLICE_END = 27.93 # profindex or longitude
        SAVE_DEVELOPMENT_DATA = False # if True save step by step data
        VERSION_2D_McDA = "V1.0"
        TYPE_2D_McDA = "Prototype"
        OUT_FOLDER = "/home/vaillant/codes/projects/2D_McDA_for_PSCs/out/data/"    
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


    # ********************************
    # *** Configuration parameters ***
    print("\n*****Configuration parameters...*****")
    
    print("\tGRANULE_DATE =", GRANULE_DATE)
    print("\tVERSION_CAL_LID_L1 =", VERSION_CAL_LID_L1)
    print("\tTYPE_CAL_LID_L1 =", TYPE_CAL_LID_L1)
    print("\tPREVIOUS_GRANULE =", PREVIOUS_GRANULE)
    print("\tNEXT_GRANULE =", NEXT_GRANULE)
    print("\tSLICE_START_END_TYPE =", SLICE_START_END_TYPE)
    print("\tSLICE_START =", SLICE_START)
    print("\tSLICE_END =", SLICE_END)
    print("\tSAVE_DEVELOPMENT_DATA =", SAVE_DEVELOPMENT_DATA)
    print("\tVERSION_2D_McDA =", VERSION_2D_McDA)
    print("\tTYPE_2D_McDA =", TYPE_2D_McDA)
    print("\tOUT_FOLDER =", OUT_FOLDER)


    # ***************************
    # *** Load CALIOP L1 data ***
    print("\n*****Load CALIOP L1 data...*****")
    
    tic = datetime.now()

    cal_l1 = CALIOPReader(product='L1',
                          version=VERSION_CAL_LID_L1,
                          data_type=TYPE_CAL_LID_L1,
                          granule_date=GRANULE_DATE,
                          slice_start=SLICE_START,
                          slice_end=SLICE_END,
                          slice_start_end_type=SLICE_START_END_TYPE)

    # Print filepaths of loading files
    print(f"\tGranule path: {cal_l1.filepath}")

    # Print lat/lon of min and max prof indices
    print(f"\tFrom min profile index {cal_l1.prof_min:d} "
          f"(lat = {cal_l1.lat_min:.2f} / lon = {cal_l1.lon_min:.2f}) "
          f"to max profile index {cal_l1.prof_max:d} "
          f"(lat = {cal_l1.lat_max:.2f} / lon = {cal_l1.lon_max:.2f})")

    # Load L1 parameters
    cal_l1_keys = [
        "Profile_ID",
        "Profile_Time",
        "Profile_UTC_Time",
        "Latitude",
        "Longitude",
        "Lidar_Data_Altitudes",
        "Total_Attenuated_Backscatter_532",
        "Perpendicular_Attenuated_Backscatter_532",
        "Attenuated_Backscatter_1064",
        "Molecular_Number_Density",
        "Ozone_Number_Density",
        "Met_Data_Altitudes",
        "Number_Bins_Shift",
        "Spacecraft_Altitude",
        "Off_Nadir_Angle",
        "Parallel_RMS_Baseline_532",
        "Perpendicular_RMS_Baseline_532",
        "RMS_Baseline_1064",
        "Calibration_Constant_532",
        "Calibration_Constant_1064",
        "Depolarization_Gain_Ratio_532",
        "Laser_Energy_532",
        "Laser_Energy_1064",
        "Parallel_Amplifier_Gain_532",
        "Perpendicular_Amplifier_Gain_532",
        "Amplifier_Gain_1064",
        "Noise_Scale_Factor_532_Parallel",
        "Noise_Scale_Factor_532_Perpendicular",
        "Noise_Scale_Factor_1064"
    ]
    data_dict_cal_lid_l1 = {}
    for key in cal_l1_keys:
        data_dict_cal_lid_l1[key] = cal_l1.get_data(key)

    # "Parallel_RMS_Baseline_532_AB_domain"
    # "Noise_Scale_Factor_532_Parallel_AB_domain"
    # "Perpendicular_RMS_Baseline_532_AB_domain"
    # "RMS_Baseline_1064_AB_domain"
    # "Noise_Scale_Factor_1064_AB_domain"
    

    # ### LOAD PREVIOUS GRANULE
    # previous_file_data_used = False
    # # If beginning of file and previous file given
    # if cal_l1.prof_min == 0:
    #     if PREVIOUS_GRANULE:
    #         # Load previous granule
    #         cal_l1_prev = CALIOPReader(product='L1',
    #                                            version=VERSION_CAL_LID_L1,
    #                                             data_type=TYPE_CAL_LID_L1,
    #                                             granule_date=PREVIOUS_GRANULE,
    #                                             grid='333mx30m',
    #                                             slice_start=-NB_PROF_OVERLAP,
    #                                             slice_end=None,
    #                                             slice_start_end_type='profindex')
        
    #         # Print filepaths of loading files
    #         print(f"\tPrevious granule path: {cal_l1_prev.filepath}")
        
    #         data_dict_cal_lid_l1_prev = {}
    #         for key in cal_l1_keys:
    #             data_dict_cal_lid_l1_prev[key] = cal_l1_prev.get_data(key)

    #         # If less than 1 second between last profile of previous file and
    #         # first profile of current file (no missing granule)
    #         time_between_profiles = np.abs(data_dict_cal_lid_l1["Profile_Time"][0] -
    #                                     data_dict_cal_lid_l1_prev["Profile_Time"][-1])
    #         print(f"\tTime between last profile of previous file and first profile of current file"
    #             f" = {time_between_profiles:.2f} s")
    #         if time_between_profiles < 1:
    #             print("\tAppend previous granule")
    #             previous_file_data_used = True
    #             # Append last profiles of previous file to data
    #             for key in cal_l1_keys:
    #                 if key != "Lidar_Data_Altitudes":
    #                     data_dict_cal_lid_l1[key] = np.append(data_dict_cal_lid_l1_prev[key],
    #                                                           data_dict_cal_lid_l1[key], axis=0)
    #         else:
    #             print("\tPrevious granule does not seem consecutive.")
    #     else :
    #         print("\tNo previous file to load.")


    # ### LOAD NEXT GRANULE
    # next_file_data_used = False
    # # If end of file and next file given
    # nb_profiles_in_granule = cal_l1._lon_granule_l1.size
    # if cal_l1.prof_max == nb_profiles_in_granule - 1:
    #     if NEXT_GRANULE:
    #         # Load next granule
    #         cal_l1_next = CALIOPReader(product='L1',
    #                                            version=VERSION_CAL_LID_L1,
    #                                             data_type=TYPE_CAL_LID_L1,
    #                                             granule_date=NEXT_GRANULE,
    #                                             grid='333mx30m',
    #                                             slice_start=None,
    #                                             slice_end=NB_PROF_OVERLAP,
    #                                             slice_start_end_type='profindex')
        
    #         # Print filepaths of loading files
    #         print(f"\tNext granule path: {cal_l1_next.filepath}")
        
    #         data_dict_cal_lid_l1_next = {}
    #         for key in cal_l1_keys:
    #             data_dict_cal_lid_l1_next[key] = cal_l1_next.get_data(key)

    #         # If less than 1 second between first profile of next file and
    #         # last profile of current file (no missing granule)
    #         time_between_profiles = np.abs(data_dict_cal_lid_l1_next["Profile_Time"][0] -
    #                                        data_dict_cal_lid_l1["Profile_Time"][-1])
    #         print(f"\tTime between last profile of previous file and first profile of current file"
    #             f" = {time_between_profiles:.2f} s")
    #         if time_between_profiles < 1:
    #             print("\tAppend next granule")
    #             next_file_data_used = True
    #             # Append first profiles of next file to data
    #             for key in cal_l1_keys:
    #                 if key != "Lidar_Data_Altitudes":
    #                     data_dict_cal_lid_l1[key] = np.append(data_dict_cal_lid_l1[key],
    #                                                           data_dict_cal_lid_l1_next[key], axis=0)
    #         else:
    #             print("\tNext granule does not seem consecutive.")
    #     else:
    #         print("\tNo next file to load.")

    print_elapsed_time(tic)


    # **********************************************************
    # *** Compute the 532 nm parallel attenuated backscatter ***
    print("\n*****Compute the 532 nm parallel attenuated backscatter...*****")

    tic = datetime.now()

    data_dict_cal_lid_l1["Parallel_Attenuated_Backscatter_532"] = compute_par_ab532(data_dict_cal_lid_l1["Total_Attenuated_Backscatter_532"], 
                                                                                    data_dict_cal_lid_l1["Perpendicular_Attenuated_Backscatter_532"])

    print_elapsed_time(tic)


    # ********************************************************
    # *** Estimate molecular attenuated backscatter signals ***
    print("\n*****Estimate molecular attenuated backscatter signals...*****")

    tic = datetime.now()

    data_dict_cal_lid_l1["Molecular_Total_Attenuated_Backscatter_532"], _ = compute_ab_mol_and_b_mol(data_dict_cal_lid_l1["Molecular_Number_Density"], 
                                                                                                        data_dict_cal_lid_l1["Ozone_Number_Density"], 
                                                                                                        data_dict_cal_lid_l1["Lidar_Data_Altitudes"],
                                                                                                        data_dict_cal_lid_l1["Met_Data_Altitudes"], 
                                                                                                        wl=532)
    data_dict_cal_lid_l1["Molecular_Parallel_Attenuated_Backscatter_532"], _ = compute_ab_mol_and_b_mol(data_dict_cal_lid_l1["Molecular_Number_Density"], 
                                                                                                        data_dict_cal_lid_l1["Ozone_Number_Density"], 
                                                                                                        data_dict_cal_lid_l1["Lidar_Data_Altitudes"],
                                                                                                        data_dict_cal_lid_l1["Met_Data_Altitudes"], 
                                                                                                        wl=532, 
                                                                                                        polar='par')
    data_dict_cal_lid_l1["Molecular_Perpendicular_Attenuated_Backscatter_532"], _ = compute_ab_mol_and_b_mol(data_dict_cal_lid_l1["Molecular_Number_Density"], 
                                                                                                             data_dict_cal_lid_l1["Ozone_Number_Density"], 
                                                                                                             data_dict_cal_lid_l1["Lidar_Data_Altitudes"],
                                                                                                             data_dict_cal_lid_l1["Met_Data_Altitudes"], 
                                                                                                             wl=532, 
                                                                                                             polar='per')
    data_dict_cal_lid_l1["Molecular_Attenuated_Backscatter_1064"], _ = compute_ab_mol_and_b_mol(data_dict_cal_lid_l1["Molecular_Number_Density"], 
                                                                                                data_dict_cal_lid_l1["Ozone_Number_Density"], 
                                                                                                data_dict_cal_lid_l1["Lidar_Data_Altitudes"],
                                                                                                data_dict_cal_lid_l1["Met_Data_Altitudes"], 
                                                                                                wl=1064)
    
    print_elapsed_time(tic)


    # **********************************************************************
    # *** Compute signal uncertainty (from shot noise and background noise)  ***
    print("\n*****Compute signal uncertainty (from shot noise and background noise) ...*****")

    tic = datetime.now()

    # Convert RMS and NSF to AB domain
    range_alt = range_from_altitude(data_dict_cal_lid_l1["Spacecraft_Altitude"][:, np.newaxis], 
                                    data_dict_cal_lid_l1["Lidar_Data_Altitudes"][np.newaxis, :], 
                                    data_dict_cal_lid_l1["Off_Nadir_Angle"][:, np.newaxis])
    data_dict_cal_lid_l1["Parallel_RMS_Baseline_532_AB_domain"] = rms_from_P_domain_to_betap_domain(data_dict_cal_lid_l1["Parallel_RMS_Baseline_532"][:, np.newaxis], 
                                                                                                    range_alt, 
                                                                                                    data_dict_cal_lid_l1["Laser_Energy_532"][:, np.newaxis], 
                                                                                                    data_dict_cal_lid_l1["Parallel_Amplifier_Gain_532"][:, np.newaxis], 
                                                                                                    data_dict_cal_lid_l1["Calibration_Constant_532"][:, np.newaxis], 
                                                                                                    np.array((1,)))
    data_dict_cal_lid_l1["Noise_Scale_Factor_532_Parallel_AB_domain"] = nsf_from_V_domain_to_betap_domain(data_dict_cal_lid_l1["Noise_Scale_Factor_532_Parallel"][:, np.newaxis], 
                                                                                                          range_alt, 
                                                                                                          data_dict_cal_lid_l1["Laser_Energy_532"][:, np.newaxis], 
                                                                                                          data_dict_cal_lid_l1["Calibration_Constant_532"][:, np.newaxis], 
                                                                                                          np.array((1,)))
    data_dict_cal_lid_l1["Perpendicular_RMS_Baseline_532_AB_domain"] = rms_from_P_domain_to_betap_domain(data_dict_cal_lid_l1["Perpendicular_RMS_Baseline_532"][:, np.newaxis], 
                                                                                                         range_alt, 
                                                                                                         data_dict_cal_lid_l1["Laser_Energy_532"][:, np.newaxis], 
                                                                                                         data_dict_cal_lid_l1["Perpendicular_Amplifier_Gain_532"][:, np.newaxis], 
                                                                                                         data_dict_cal_lid_l1["Calibration_Constant_532"][:, np.newaxis], 
                                                                                                         data_dict_cal_lid_l1["Depolarization_Gain_Ratio_532"][:, np.newaxis])
    data_dict_cal_lid_l1["Noise_Scale_Factor_532_Perpendicular_AB_domain"] = nsf_from_V_domain_to_betap_domain(data_dict_cal_lid_l1["Noise_Scale_Factor_532_Perpendicular"][:, np.newaxis], 
                                                                                                               range_alt, 
                                                                                                               data_dict_cal_lid_l1["Laser_Energy_532"][:, np.newaxis], 
                                                                                                               data_dict_cal_lid_l1["Calibration_Constant_532"][:, np.newaxis], 
                                                                                                               data_dict_cal_lid_l1["Depolarization_Gain_Ratio_532"][:, np.newaxis]) 
    data_dict_cal_lid_l1["RMS_Baseline_1064_AB_domain"] = rms_from_P_domain_to_betap_domain(data_dict_cal_lid_l1["RMS_Baseline_1064"][:, np.newaxis], 
                                                                                            range_alt, 
                                                                                            data_dict_cal_lid_l1["Laser_Energy_1064"][:, np.newaxis], 
                                                                                            data_dict_cal_lid_l1["Amplifier_Gain_1064"][:, np.newaxis], 
                                                                                            data_dict_cal_lid_l1["Calibration_Constant_1064"][:, np.newaxis], 
                                                                                            np.array((1,)))
    data_dict_cal_lid_l1["Noise_Scale_Factor_1064_AB_domain"] = nsf_from_V_domain_to_betap_domain(data_dict_cal_lid_l1["Noise_Scale_Factor_1064"][:, np.newaxis], 
                                                                                                  range_alt, 
                                                                                                  data_dict_cal_lid_l1["Laser_Energy_1064"][:, np.newaxis], 
                                                                                                  data_dict_cal_lid_l1["Calibration_Constant_1064"][:, np.newaxis], 
                                                                                                  np.array((1,)))

    # Compute uncertainties
    data_dict_cal_lid_l1["Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Parallel"],\
        data_dict_cal_lid_l1["Background_Noise_532_Parallel"],\
        data_dict_cal_lid_l1["Shot_Noise_532_Parallel"] = compute_uncertainty(data_dict_cal_lid_l1["Number_Bins_Shift"],
                                                                              data_dict_cal_lid_l1["Molecular_Parallel_Attenuated_Backscatter_532"],
                                                                              data_dict_cal_lid_l1["Parallel_RMS_Baseline_532_AB_domain"],
                                                                              data_dict_cal_lid_l1["Noise_Scale_Factor_532_Parallel_AB_domain"])
    data_dict_cal_lid_l1["Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Perpendicular"],\
        data_dict_cal_lid_l1["Background_Noise_532_Perpendicular"],\
        data_dict_cal_lid_l1["Shot_Noise_532_Perpendicular"] = compute_uncertainty(data_dict_cal_lid_l1["Number_Bins_Shift"],
                                                                                   data_dict_cal_lid_l1["Molecular_Perpendicular_Attenuated_Backscatter_532"],
                                                                                   data_dict_cal_lid_l1["Perpendicular_RMS_Baseline_532_AB_domain"],
                                                                                   data_dict_cal_lid_l1["Noise_Scale_Factor_532_Perpendicular_AB_domain"])
    data_dict_cal_lid_l1["Attenuated_Backscatter_Uncertainty_Standard_Deviation_1064"],\
        data_dict_cal_lid_l1["Background_Noise_1064"],\
        data_dict_cal_lid_l1["Shot_Noise_1064"] = compute_uncertainty(data_dict_cal_lid_l1["Number_Bins_Shift"],
                                                                      data_dict_cal_lid_l1["Molecular_Attenuated_Backscatter_1064"],
                                                                      data_dict_cal_lid_l1["RMS_Baseline_1064_AB_domain"],
                                                                      data_dict_cal_lid_l1["Noise_Scale_Factor_1064_AB_domain"])
    
    print_elapsed_time(tic)


    # ***************************************
    # *** Compute 532 nm scattering ratio ***
    print("\n*****Compute 532 nm scattering ratio ...*****")

    tic = datetime.now()

    data_dict_cal_lid_l1["Attenuated_Scattering_Ratio_532"] = data_dict_cal_lid_l1["Total_Attenuated_Backscatter_532"] / data_dict_cal_lid_l1["Molecular_Total_Attenuated_Backscatter_532"]

    print_elapsed_time(tic)


    # *********************************************************************
    # *** Get data between 8.2 km and 30.1 km at 5-km×180-m resolution  ***
    print("\n*****Get data between 8.2 km and 30.1 km at 5-km×180-m resolution ...*****")

    tic = datetime.now()

    # Initialization
    data_dict_5kmx180m = {}
    data_dict_5km_met = {}
    START_INDEX_R4 = LAYER_ALTITUDE_R4_INDEX_RANGE[0] # at 30.1 km
    END_INDEX_R4 = LAYER_ALTITUDE_R4_INDEX_RANGE[1] # at 20.2km; 55 bins alreadu at 180-m resolution in R4
    START_INDEX_R3 = LAYER_ALTITUDE_R3_INDEX_RANGE[0] # at 20.2km
    END_INDEX_R3 = LAYER_ALTITUDE_R3_INDEX_RANGE[1] - 2 # at 8.2 km; 200 bins of 60-m in R3 => 200-2 = 198 => 198/3 = 66.0 bins of 180 m
    NB_VERT_BINS_TO_AVERAGE_IN_R3 = 3 # resolution 180m on a 60m grid
    NB_HORIZ_BINS_TO_AVERAGE = 15 # resolution 5km on a 333m grid
    nb_vert_bins_180m_R4 = END_INDEX_R4 - START_INDEX_R4 + 1
    nb_vert_bins_180m_R3 = int((END_INDEX_R3 - START_INDEX_R3 + 1)/3)
    
    # Get first profile ID of chunk
    if PREVIOUS_GRANULE:
        print("cal_l1_prev.prof_min:", cal_l1_prev.prof_min)
        prof_index_first_in_chunk = get_first_profileID_of_chunk(cal_l1_prev.prof_min)
        print("prof_index_first_in_chunk:", prof_index_first_in_chunk)
    else:
        prof_index_first_in_chunk = get_first_profileID_of_chunk(cal_l1.prof_min)
    
    # Get number of 5-km chunks
    nb_chunk_5km = int(data_dict_cal_lid_l1["Latitude"][prof_index_first_in_chunk:].size/NB_HORIZ_BINS_TO_AVERAGE)
    cal_lid_l1_prof_index_range_mult_of_15 = np.arange(prof_index_first_in_chunk, prof_index_first_in_chunk+nb_chunk_5km*15)

    # 1-D vertical data at 180-m resolution
    for key in ["Lidar_Data_Altitudes",]:
        # Initialization
        data_dict_5kmx180m[key] = np.ma.ones(nb_vert_bins_180m_R4+nb_vert_bins_180m_R3)*FILL_VALUE_FLOAT
        # Copy R4 region (20.2 - 30.1 km)
        data_dict_5kmx180m[key][:nb_vert_bins_180m_R4] = data_dict_cal_lid_l1[key][START_INDEX_R4:END_INDEX_R4+1]
        # Take middle altitude of 3 60-m vertical bins in R3 region (8.2 - 20.2 km)
        data_dict_5kmx180m[key][nb_vert_bins_180m_R4:] = data_dict_cal_lid_l1[key][START_INDEX_R3+1:END_INDEX_R3:3]

    # 1-D horizontal data at 5-km resolution
    for key in ["Latitude", "Longitude", "Number_Bins_Shift", "Profile_ID", "Profile_Time", "Profile_UTC_Time"]: 
        # Take middle (8th) profile of 5-km horizontal bins
        data_dict_5kmx180m[key] = data_dict_cal_lid_l1[key][cal_lid_l1_prof_index_range_mult_of_15][int(NB_HORIZ_BINS_TO_AVERAGE/2)::NB_HORIZ_BINS_TO_AVERAGE]
    
    # 2-D data at 5-km×180-m resolution
    key_list = ["Total_Attenuated_Backscatter_532", "Parallel_Attenuated_Backscatter_532", "Perpendicular_Attenuated_Backscatter_532", "Attenuated_Backscatter_1064",
                "Molecular_Total_Attenuated_Backscatter_532", "Molecular_Parallel_Attenuated_Backscatter_532", "Molecular_Perpendicular_Attenuated_Backscatter_532", "Molecular_Attenuated_Backscatter_1064",
                "Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Parallel", "Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Perpendicular", "Attenuated_Backscatter_Uncertainty_Standard_Deviation_1064",
                "Attenuated_Scattering_Ratio_532"]
    if SAVE_DEVELOPMENT_DATA:
        key_list = key_list + ["Background_Noise_532_Parallel", "Background_Noise_532_Perpendicular", "Background_Noise_1064", 
                               "Shot_Noise_532_Parallel", "Shot_Noise_532_Perpendicular", "Shot_Noise_1064",
                               "Noise_Scale_Factor_532_Parallel_AB_domain", "Noise_Scale_Factor_532_Perpendicular_AB_domain", "Noise_Scale_Factor_1064_AB_domain"]
    for key in key_list:
        # Initialization
        data_dict_5kmx180m[key] = np.ma.ones((nb_chunk_5km, nb_vert_bins_180m_R4+nb_vert_bins_180m_R3))*FILL_VALUE_FLOAT
        # Average data
        for prof_i in np.arange(nb_chunk_5km):
            for alt_i in np.arange(nb_vert_bins_180m_R4):
                data_dict_5kmx180m[key][prof_i, alt_i] =\
                    np.ma.mean(data_dict_cal_lid_l1[key][prof_index_first_in_chunk+prof_i*NB_HORIZ_BINS_TO_AVERAGE:prof_index_first_in_chunk+(prof_i+1)*NB_HORIZ_BINS_TO_AVERAGE,
                                                         START_INDEX_R4+alt_i])
            for alt_i in np.arange(nb_vert_bins_180m_R3):
                data_dict_5kmx180m[key][prof_i, nb_vert_bins_180m_R4+alt_i] =\
                    np.ma.mean(data_dict_cal_lid_l1[key][prof_index_first_in_chunk+prof_i*NB_HORIZ_BINS_TO_AVERAGE:prof_index_first_in_chunk+(prof_i+1)*NB_HORIZ_BINS_TO_AVERAGE,
                                                         START_INDEX_R3+alt_i*NB_VERT_BINS_TO_AVERAGE_IN_R3:START_INDEX_R3+(alt_i+1)*NB_VERT_BINS_TO_AVERAGE_IN_R3])

    # Print number of profiles in the granule
    print(f"\tNumber of 5-km profiles to process: {nb_chunk_5km}")

    print_elapsed_time(tic)
    

    # *************************
    # *** Feature detection ***
    print("\n\n*****Feature detection...*****")

    tic_algo = print_time()

    # Initialization
    data_dict_2d_mcda = {}
    data_dict_2d_mcda_dev = {}

    # Feature detection at 532 nm parallel
    data_dict_2d_mcda["Parallel_Detection_Flags_532"], \
    data_dict_2d_mcda_dev["Parallel_Detection_Flags_532_steps"], \
    data_dict_2d_mcda_dev["Parallel_Attenuated_Backscatter_532_steps"], \
    data_dict_2d_mcda_dev["Parallel_Spikes_532"] =\
        detect_features(data_dict_5kmx180m["Parallel_Attenuated_Backscatter_532"],
                        data_dict_5kmx180m["Molecular_Parallel_Attenuated_Backscatter_532"],
                        data_dict_5kmx180m["Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Parallel"],
                        '532_par')

    # Feature detection at 532 nm perpendicular
    # Note: use surface detection at 532 nm parallel
    data_dict_2d_mcda["Perpendicular_Detection_Flags_532"], \
    data_dict_2d_mcda_dev["Perpendicular_Detection_Flags_532_steps"], \
    data_dict_2d_mcda_dev["Perpendicular_Attenuated_Backscatter_532_steps"], \
    data_dict_2d_mcda_dev["Perpendicular_Spikes_532"] =\
        detect_features(data_dict_5kmx180m["Perpendicular_Attenuated_Backscatter_532"],
                        data_dict_5kmx180m["Molecular_Perpendicular_Attenuated_Backscatter_532"],
                        data_dict_5kmx180m["Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Perpendicular"],
                        '532_per')

    # Feature detection at 1064 nm
    data_dict_2d_mcda["Detection_Flags_1064"], \
    data_dict_2d_mcda_dev["Detection_Flags_1064_steps"], \
    data_dict_2d_mcda_dev["Attenuated_Backscatter_1064_steps"], \
    data_dict_2d_mcda_dev["Spikes_1064"] =\
        detect_features(data_dict_5kmx180m["Attenuated_Backscatter_1064"],
                        data_dict_5kmx180m["Molecular_Attenuated_Backscatter_1064"],
                        data_dict_5kmx180m["Attenuated_Backscatter_Uncertainty_Standard_Deviation_1064"],
                        '1064')


    # *******************************************
    # *** Merged 3 channels feature detection ***
    print("\n\n*****Merged 3 channels feature detection...*****")

    data_dict_2d_mcda["Composite_Detection_Flags"] = \
        merged_feature_masks(data_dict_2d_mcda["Parallel_Detection_Flags_532"],
                                data_dict_2d_mcda["Perpendicular_Detection_Flags_532"],
                                data_dict_2d_mcda["Detection_Flags_1064"])

    print_elapsed_time(tic_algo)
    
    
    # # If beginning of file and previous file given
    # if previous_file_data_used:
    #     # ******************************************
    #     # *** Remove profiles from previous file ***
    #     print("\n\n*****Remove profiles from previous file...*****")
    #     for key in data_dict:
    #         if key != "Lidar_Data_Altitudes":
    #             data_dict_5kmx180m[key] = rm_prof(data_dict_5kmx180m[key], int(NB_PROF_OVERLAP/15), 'start') # /15 because we remove 5-km chunk here
    #     for key in data_dict_2d_mcda:
    #         data_dict_2d_mcda[key] = rm_prof(data_dict_2d_mcda[key], int(NB_PROF_OVERLAP/15), 'start') # /15 because we remove 5-km chunk here
    #     for key in data_dict_2d_mcda_dev:
    #         data_dict_2d_mcda_dev[key] = rm_prof(data_dict_2d_mcda_dev[key], int(NB_PROF_OVERLAP/15), 'start') # /15 because we remove 5-km chunk here


    # # If end of file and next file given
    # elif next_file_data_used:
    #     # **************************************
    #     # *** Remove profiles from next file ***
    #     print("\n\n*****Remove profiles from next file...*****")
    #     for key in data_dict:
    #         if key != "Lidar_Data_Altitudes":
    #             data_dict_5kmx180m[key] = rm_prof(data_dict_5kmx180m[key], int(NB_PROF_OVERLAP/15), 'end')
    #     for key in data_dict_2d_mcda:
    #         data_dict_2d_mcda[key] = rm_prof(data_dict_2d_mcda[key], int(NB_PROF_OVERLAP/15), 'end')
    #     for key in data_dict_2d_mcda_dev:
    #         data_dict_2d_mcda_dev[key] = rm_prof(data_dict_2d_mcda_dev[key], int(NB_PROF_OVERLAP/15), 'end')

    SEPARATION_TYPE = "all_levels_and_channels" # "pixel", "channel", "best_detection_level", or "all_levels_and_channels"

    if True:
        if SEPARATION_TYPE == "pixel":
            data_dict_2d_mcda["homogeneous_feature_mask"] = np.ma.ones(data_dict_2d_mcda["Parallel_Detection_Flags_532"].shape) # not used
        else:
            # **************************************
            # *** Separated homogeneous features ***
            print("\n\n############################################################\n"\
                "*****Separated homogeneous features...*****")

            tic_algo = print_time()
        
            data_dict_2d_mcda["homogeneous_feature_mask"] = \
                    separate_homogeneous_features(data_dict_2d_mcda["Composite_Detection_Flags"],
                                                data_dict_2d_mcda["Parallel_Detection_Flags_532"],
                                                data_dict_2d_mcda["Perpendicular_Detection_Flags_532"],
                                                data_dict_2d_mcda["Detection_Flags_1064"],
                                                separation_type=SEPARATION_TYPE)
            
            print_elapsed_time(tic_algo)

    if True:
        # ***********************************************************
        # *** Apply PSC v2 classification to homogeneous features ***
        print("\n\n############################################################\n"\
            "*****Apply PSC v2 classification to homogeneous features...*****")

        tic_algo = print_time()

        data_dict_2d_mcda["homogeneous_feature_classification"], \
        data_dict_2d_mcda["homogeneous_feature_mean_ab_532_per"], \
        data_dict_2d_mcda["homogeneous_feature_mean_asr_532"] = \
            classify_homogeneous_features_with_psc_v2(data_dict_2d_mcda["homogeneous_feature_mask"], 
                                                      data_dict_5kmx180m["Perpendicular_Attenuated_Backscatter_532"],
                                                      data_dict_5kmx180m["Attenuated_Scattering_Ratio_532"],
                                                      data_dict_2d_mcda_dev["Parallel_Spikes_532"],
                                                      data_dict_2d_mcda_dev["Perpendicular_Spikes_532"],
                                                      separation_type=SEPARATION_TYPE)
        
        print_elapsed_time(tic_algo)


    # *****************************
    # *** Save data in HDF file ***
    print("\n\n############################################################\n"\
          "*****Save data in HDF file...*****")
    
        # Create a dictionary of parameters to save in HDF file
    params = {}
    
    # Add prof_ID to params
    params['prof_ID'] = SDSData('Profile_ID', data_dict_5kmx180m["Profile_ID"])
    params['prof_ID'].description = "Profile number from start of file"
    # params['prof_ID'].valid_range = (1, 228630)
    params['prof_ID'].dim_labels = ['Profile_ID']

    # Add prof_time to params
    params['prof_time'] = SDSData('Profile_Time', data_dict_5kmx180m["Profile_Time"])
    params['prof_time'].units = "seconds...TAI"
    # params['prof_time'].valid_range = (4.204e8, 1.072e9)
    params['prof_time'].dim_labels = ['Profile_ID']

    # Add prof_UTC_time to params
    params['prof_UTC_time'] = SDSData('Profile_UTC_Time', data_dict_5kmx180m["Profile_UTC_Time"])
    params['prof_UTC_time'].units = "UTC - yymmdd.ffffffff"
    # params['prof_UTC_time'].valid_range = (60426.0, 261231.0)
    params['prof_UTC_time'].dim_labels = ['Profile_ID']

    # Add lat to params
    params['lat'] = SDSData('Latitude', data_dict_5kmx180m["Latitude"], FILL_VALUE_FLOAT)
    params['lat'].units = "degrees"
    # params['lat'].valid_range = (-90.0, 90.0)
    params['lat'].dim_labels = ['Profile_ID']

    # Add lon to params
    params['lon'] = SDSData('Longitude', data_dict_5kmx180m["Longitude"], FILL_VALUE_FLOAT)
    params['lon'].units = "degrees"
    # params['lon'].valid_range = (-90.0, 90.0)
    params['lon'].dim_labels = ['Profile_ID']
    
    # Add alt to params
    params['alt'] = SDSData('Altitude', data_dict_5kmx180m["Lidar_Data_Altitudes"])
    params['alt'].units = "kilometer"
    # params['alt'].valid_range = (-0.5, 30.1)
    params['alt'].dim_labels = ['Altitude']
    
    # Add feature_mask_532_par to params
    params['feature_mask_532_par'] = SDSData('Parallel_Detection_Flags_532',
                                             data_dict_2d_mcda["Parallel_Detection_Flags_532"])
    params['feature_mask_532_par'].valid_range = (0, 255)
    params['feature_mask_532_par'].dim_labels = ['Profile_ID', 'Altitude']

    # Add feature_mask_532_per to params
    params['feature_mask_532_per'] = SDSData('Perpendicular_Detection_Flags_532',
                                             data_dict_2d_mcda["Perpendicular_Detection_Flags_532"])
    params['feature_mask_532_per'].valid_range = (0, 255)
    params['feature_mask_532_per'].dim_labels = ['Profile_ID', 'Altitude']

    # Add feature_mask_1064 to params
    params['feature_mask_1064'] = SDSData('Detection_Flags_1064',
                                          data_dict_2d_mcda["Detection_Flags_1064"])
    params['feature_mask_1064'].valid_range = (0, 255)
    params['feature_mask_1064'].dim_labels = ['Profile_ID', 'Altitude']

    # Add feature_mask_merged to params
    params['feature_mask_merged'] = SDSData('Composite_Detection_Flags',
                                            data_dict_2d_mcda["Composite_Detection_Flags"])
    params['feature_mask_merged'].valid_range = (0, 255)
    params['feature_mask_merged'].dim_labels = ['Profile_ID', 'Altitude']
    
    # Add spikes array
    params['parallel_spikes_532'] = SDSData('Parallel_Spikes_532',
                                            data_dict_2d_mcda_dev["Parallel_Spikes_532"])
    params['parallel_spikes_532'].valid_range = (0, 1)
    params['parallel_spikes_532'].dim_labels = ['Profile_ID', 'Altitude']
    
    params['perpendicular_spikes_532'] = SDSData('Perpendicular_Spikes_532',
                                                 data_dict_2d_mcda_dev["Perpendicular_Spikes_532"])
    params['perpendicular_spikes_532'].valid_range = (0, 1)
    params['perpendicular_spikes_532'].dim_labels = ['Profile_ID', 'Altitude']

    params['spikes_1064'] = SDSData('Spikes_1064',
                                    data_dict_2d_mcda_dev["Spikes_1064"])
    params['spikes_1064'].valid_range = (0, 1)
    params['spikes_1064'].dim_labels = ['Profile_ID', 'Altitude']

    if True:
        # Add homogeneous_feature_mask to params
        params['homogeneous_feature_mask'] = SDSData('Homogeneous_Feature_Mask',
                                                data_dict_2d_mcda["homogeneous_feature_mask"])
        params['homogeneous_feature_mask'].valid_range = (0, 255)
        params['homogeneous_feature_mask'].dim_labels = ['Profile_ID', 'Altitude']

    if True:
        # Add homogeneous_feature_mask to params
        params['homogeneous_feature_classification'] = SDSData('Homogeneous_Feature_Classification',
                                                data_dict_2d_mcda["homogeneous_feature_classification"])
        params['homogeneous_feature_classification'].valid_range = (0, 255)
        params['homogeneous_feature_classification'].dim_labels = ['Profile_ID', 'Altitude']

        # Add homogeneous_feature_mask to params
        params['homogeneous_feature_mean_ab_532_per'] = SDSData('Homogeneous_Feature_Mean_Perpendicular_Attenuated_Backscatter_532',
                                                data_dict_2d_mcda["homogeneous_feature_mean_ab_532_per"])
        params['homogeneous_feature_mean_ab_532_per'].valid_range = (0, 255)
        params['homogeneous_feature_mean_ab_532_per'].dim_labels = ['Profile_ID', 'Altitude']

        # Add homogeneous_feature_mask to params
        params['homogeneous_feature_mean_asr_532'] = SDSData('Homogeneous_Feature_Mean_Attenuated_Scattering_Ratio_532',
                                                data_dict_2d_mcda["homogeneous_feature_mean_asr_532"])
        params['homogeneous_feature_mean_asr_532'].valid_range = (0, 255)
        params['homogeneous_feature_mean_asr_532'].dim_labels = ['Profile_ID', 'Altitude']
    

    # Parameters saved for development
    if SAVE_DEVELOPMENT_DATA:
        
        # Add Parallel_Attenuated_Backscatter_532 to params
        params['Parallel_Attenuated_Backscatter_532'] =\
            SDSData('Parallel_Attenuated_Backscatter_532',
                    data_dict_5kmx180m["Parallel_Attenuated_Backscatter_532"], FILL_VALUE_FLOAT)
        params['Parallel_Attenuated_Backscatter_532'].units = "km-1 sr-1"
        params['Parallel_Attenuated_Backscatter_532'].dim_labels = ['Profile_ID', 'Altitude']
    
        # Add Perpendicular_Attenuated_Backscatter_532 to params
        params['Perpendicular_Attenuated_Backscatter_532'] =\
            SDSData('Perpendicular_Attenuated_Backscatter_532',
                    data_dict_5kmx180m["Perpendicular_Attenuated_Backscatter_532"], FILL_VALUE_FLOAT)
        params['Perpendicular_Attenuated_Backscatter_532'].units = "km-1 sr-1"
        params['Perpendicular_Attenuated_Backscatter_532'].dim_labels = ['Profile_ID', 'Altitude']
    
        # Add Attenuated_Backscatter_1064 to params
        params['Attenuated_Backscatter_1064'] =\
            SDSData('Attenuated_Backscatter_1064',
                    data_dict_5kmx180m["Attenuated_Backscatter_1064"], FILL_VALUE_FLOAT)
        params['Attenuated_Backscatter_1064'].units = "km-1 sr-1"
        params['Attenuated_Backscatter_1064'].dim_labels = ['Profile_ID', 'Altitude']

        # Add uncertainties 532 par
        params['Molecular_Parallel_Attenuated_Backscatter_532'] =\
            SDSData('Molecular_Parallel_Attenuated_Backscatter_532',
                    data_dict_5kmx180m["Molecular_Parallel_Attenuated_Backscatter_532"], FILL_VALUE_FLOAT)
        params['Molecular_Parallel_Attenuated_Backscatter_532'].units = "km-1 sr-1"
        params['Molecular_Parallel_Attenuated_Backscatter_532'].dim_labels = ['Profile_ID', 'Altitude']

        params['Noise_Scale_Factor_532_Parallel_AB_domain'] =\
            SDSData('Noise_Scale_Factor_532_Parallel_AB_domain',
                    data_dict_5kmx180m["Noise_Scale_Factor_532_Parallel_AB_domain"], FILL_VALUE_FLOAT)
        params['Noise_Scale_Factor_532_Parallel_AB_domain'].units = "km-0.5 sr-0.5"
        params['Noise_Scale_Factor_532_Parallel_AB_domain'].dim_labels = ['Profile_ID', 'Altitude']

        params['Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Parallel'] =\
            SDSData('Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Parallel',
                    data_dict_5kmx180m["Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Parallel"], FILL_VALUE_FLOAT)
        params['Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Parallel'].units = "km-1 sr-1"
        params['Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Parallel'].dim_labels = ['Profile_ID', 'Altitude']

        params['Background_Noise_532_Parallel'] =\
            SDSData('Background_Noise_532_Parallel',
                    data_dict_5kmx180m["Background_Noise_532_Parallel"], FILL_VALUE_FLOAT)
        params['Background_Noise_532_Parallel'].units = "km-1 sr-1"
        params['Background_Noise_532_Parallel'].dim_labels = ['Profile_ID', 'Altitude']

        params['Shot_Noise_532_Parallel'] =\
            SDSData('Shot_Noise_532_Parallel',
                    data_dict_5kmx180m["Shot_Noise_532_Parallel"], FILL_VALUE_FLOAT)
        params['Shot_Noise_532_Parallel'].units = "km-1 sr-1"
        params['Shot_Noise_532_Parallel'].dim_labels = ['Profile_ID', 'Altitude']

        # Add uncertainties 532 per
        params['Molecular_Perpendicular_Attenuated_Backscatter_532'] =\
            SDSData('Molecular_Perpendicular_Attenuated_Backscatter_532',
                    data_dict_5kmx180m["Molecular_Perpendicular_Attenuated_Backscatter_532"], FILL_VALUE_FLOAT)
        params['Molecular_Perpendicular_Attenuated_Backscatter_532'].units = "km-1 sr-1"
        params['Molecular_Perpendicular_Attenuated_Backscatter_532'].dim_labels = ['Profile_ID', 'Altitude']

        params['Noise_Scale_Factor_532_Perpendicular_AB_domain'] =\
            SDSData('Noise_Scale_Factor_532_Perpendicular_AB_domain',
                    data_dict_5kmx180m["Noise_Scale_Factor_532_Perpendicular_AB_domain"], FILL_VALUE_FLOAT)
        params['Noise_Scale_Factor_532_Perpendicular_AB_domain'].units = "km-0.5 sr-0.5"
        params['Noise_Scale_Factor_532_Perpendicular_AB_domain'].dim_labels = ['Profile_ID', 'Altitude']

        params['Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Perpendicular'] =\
            SDSData('Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Perpendicular',
                    data_dict_5kmx180m["Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Perpendicular"], FILL_VALUE_FLOAT)
        params['Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Perpendicular'].units = "km-1 sr-1"
        params['Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Perpendicular'].dim_labels = ['Profile_ID', 'Altitude']

        params['Background_Noise_532_Perpendicular'] =\
            SDSData('Background_Noise_532_Perpendicular',
                    data_dict_5kmx180m["Background_Noise_532_Perpendicular"], FILL_VALUE_FLOAT)
        params['Background_Noise_532_Perpendicular'].units = "km-1 sr-1"
        params['Background_Noise_532_Perpendicular'].dim_labels = ['Profile_ID', 'Altitude']

        params['Shot_Noise_532_Perpendicular'] =\
            SDSData('Shot_Noise_532_Perpendicular',
                    data_dict_5kmx180m["Shot_Noise_532_Perpendicular"], FILL_VALUE_FLOAT)
        params['Shot_Noise_532_Perpendicular'].units = "km-1 sr-1"
        params['Shot_Noise_532_Perpendicular'].dim_labels = ['Profile_ID', 'Altitude']

        # Add uncertainties 1064
        params['Molecular_Attenuated_Backscatter_1064'] =\
            SDSData('Molecular_Attenuated_Backscatter_1064',
                    data_dict_5kmx180m["Molecular_Attenuated_Backscatter_1064"], FILL_VALUE_FLOAT)
        params['Molecular_Attenuated_Backscatter_1064'].units = "km-1 sr-1"
        params['Molecular_Attenuated_Backscatter_1064'].dim_labels = ['Profile_ID', 'Altitude']

        params['Noise_Scale_Factor_1064_AB_domain'] =\
            SDSData('Noise_Scale_Factor_1064_AB_domain',
                    data_dict_5kmx180m["Noise_Scale_Factor_1064_AB_domain"], FILL_VALUE_FLOAT)
        params['Noise_Scale_Factor_1064_AB_domain'].units = "km-0.5 sr-0.5"
        params['Noise_Scale_Factor_1064_AB_domain'].dim_labels = ['Profile_ID', 'Altitude']

        params['Attenuated_Backscatter_Uncertainty_Standard_Deviation_1064'] =\
            SDSData('Attenuated_Backscatter_Uncertainty_Standard_Deviation_1064',
                    data_dict_5kmx180m["Attenuated_Backscatter_Uncertainty_Standard_Deviation_1064"], FILL_VALUE_FLOAT)
        params['Attenuated_Backscatter_Uncertainty_Standard_Deviation_1064'].units = "km-1 sr-1"
        params['Attenuated_Backscatter_Uncertainty_Standard_Deviation_1064'].dim_labels = ['Profile_ID', 'Altitude']

        params['Background_Noise_1064'] =\
            SDSData('Background_Noise_1064',
                    data_dict_5kmx180m["Background_Noise_1064"], FILL_VALUE_FLOAT)
        params['Background_Noise_1064'].units = "km-1 sr-1"
        params['Background_Noise_1064'].dim_labels = ['Profile_ID', 'Altitude']

        params['Shot_Noise_1064'] =\
            SDSData('Shot_Noise_1064',
                    data_dict_5kmx180m["Shot_Noise_1064"], FILL_VALUE_FLOAT)
        params['Shot_Noise_1064'].units = "km-1 sr-1"
        params['Shot_Noise_1064'].dim_labels = ['Profile_ID', 'Altitude']
    
        # Add feature_mask_532_par_steps to params
        params['feature_mask_532_par_steps'] =\
            SDSData('Parallel_Detection_Flags_532_steps',
                    data_dict_2d_mcda_dev["Parallel_Detection_Flags_532_steps"])
        params['feature_mask_532_par_steps'].valid_range = (0, 255)
        params['feature_mask_532_par_steps'].dim_labels = ['Step_532_par', 'Profile_ID', 'Altitude']
    
        # Add feature_mask_532_per_steps to params
        params['feature_mask_532_per_steps'] =\
            SDSData('Perpendicular_Detection_Flags_532_steps',
                    data_dict_2d_mcda_dev["Perpendicular_Detection_Flags_532_steps"])
        params['feature_mask_532_per_steps'].valid_range = (0, 255)
        params['feature_mask_532_per_steps'].dim_labels = ['Step_532_per', 'Profile_ID', 'Altitude']
    
        # Add feature_mask_1064_steps to params
        params['feature_mask_1064_steps'] =\
            SDSData('Detection_Flags_1064_steps', data_dict_2d_mcda_dev["Detection_Flags_1064_steps"])
        params['feature_mask_1064_steps'].valid_range = (0, 255)
        params['feature_mask_1064_steps'].dim_labels = ['Step_1064', 'Profile_ID', 'Altitude']
    
        # Add Parallel_Attenuated_Backscatter_532_steps to params
        params['Parallel_Attenuated_Backscatter_532_steps'] =\
            SDSData('Parallel_Attenuated_Backscatter_532_steps',
                    data_dict_2d_mcda_dev["Parallel_Attenuated_Backscatter_532_steps"], FILL_VALUE_FLOAT)
        params['Parallel_Attenuated_Backscatter_532_steps'].units = "km-1 sr-1"
        params['Parallel_Attenuated_Backscatter_532_steps'].dim_labels = ['Step_532_par', 'Profile_ID', 'Altitude']
    
        # Add Perpendicular_Attenuated_Backscatter_532_steps to params
        params['Perpendicular_Attenuated_Backscatter_532_steps'] =\
            SDSData('Perpendicular_Attenuated_Backscatter_532_steps',
                    data_dict_2d_mcda_dev["Perpendicular_Attenuated_Backscatter_532_steps"], FILL_VALUE_FLOAT)
        params['Perpendicular_Attenuated_Backscatter_532_steps'].units = "km-1 sr-1"
        params['Perpendicular_Attenuated_Backscatter_532_steps'].dim_labels = ['Step_532_per', 'Profile_ID', 'Altitude']
    
        # Add Attenuated_Backscatter_1064_steps to params
        params['Attenuated_Backscatter_1064_steps'] =\
            SDSData('Attenuated_Backscatter_1064_steps',
                    data_dict_2d_mcda_dev["Attenuated_Backscatter_1064_steps"], FILL_VALUE_FLOAT)
        params['Attenuated_Backscatter_1064_steps'].units = "km-1 sr-1"
        params['Attenuated_Backscatter_1064_steps'].dim_labels = ['Step_1064', 'Profile_ID', 'Altitude']
    
    # Create folder to store output data
    granule_date_dict = split_granule_date(GRANULE_DATE)
    outdata_folder = os.path.join(OUT_FOLDER, f"2D_McDA_PSCs.{VERSION_2D_McDA.replace('V', 'v')}",
                                  str(granule_date_dict['year']), f"{granule_date_dict['year']}_"
                                                                  f"{granule_date_dict['month']:02d}_"
                                                                  f"{granule_date_dict['day']:02d}")
    os.makedirs(outdata_folder, exist_ok=True)
    
    # Write in HDF file
    if (SLICE_START == 0 or SLICE_START == None) and (SLICE_END == None) and (SLICE_START_END_TYPE == 'profindex'):
        filename_end = '' # nothing, it is the whole file
    else:
        filename_end = f"_lon_{cal_l1.lon_min:.2f}_{cal_l1.lon_max:.2f}"
        
    filename = f"CAL_LID_L2_2D_McDA_PSCs-{TYPE_2D_McDA}-{VERSION_2D_McDA.replace('.', '-')}." \
               f"{GRANULE_DATE}{filename_end}.hdf"
    write_hdf(outdata_folder+"/"+filename, params)
    
    
    print_time(tic_main_program)
    