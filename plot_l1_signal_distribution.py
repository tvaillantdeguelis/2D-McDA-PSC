#!/usr/bin/env python
# coding: utf8
from datetime import datetime

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, gridspec

from my_modules.standard_outputs import print_time, print_elapsed_time
from my_modules.readers.calipso_reader import CALIOPReader
from my_modules.calipso_calculator import compute_par_ab532
from my_modules.figuretools import setstyle

if __name__ == '__main__':
    tic_main_program = print_time()
    
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    # PARAMETERS
    GRANULE_DATE = "2008-07-17T19-15-43ZN"
    VERSION_CAL_LID_L1 = "V4.51"
    TYPE_CAL_LID_L1 = "Standard"
    SLICE_START_END_TYPE = "longitude" # "profindex" or "longitude" (Use "profindex" if SLICE_START/END = None to process the whole granule)
    SLICE_START = 69.65 # profindex or longitude
    SLICE_END = -73.31 # profindex or longitude
    FIGURES_PATH = "/home/vaillant/codes/projects/2D_McDA_for_PSCs/out/figures/"
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


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

    print_elapsed_time(tic)


    # **********************************************************
    # *** Compute the 532 nm parallel attenuated backscatter ***
    print("\n*****Compute the 532 nm parallel attenuated backscatter...*****")

    tic = datetime.now()

    data_dict_cal_lid_l1["Parallel_Attenuated_Backscatter_532"] = compute_par_ab532(data_dict_cal_lid_l1["Total_Attenuated_Backscatter_532"], 
                                                                                    data_dict_cal_lid_l1["Perpendicular_Attenuated_Backscatter_532"])

    print_elapsed_time(tic)


    # # *********************************************************************
    # # *** Get data between 8.2 km and 30.1 km at 5-km×180-m resolution  ***
    # print("\n*****Get data between 8.2 km and 30.1 km at 5-km×180-m resolution ...*****")

    # tic = datetime.now()

    # # Initialization
    # data_dict_5kmx180m = {}
    # data_dict_5km_met = {}
    # START_INDEX_R4 = LAYER_ALTITUDE_R4_INDEX_RANGE[0] # at 30.1 km
    # END_INDEX_R4 = LAYER_ALTITUDE_R4_INDEX_RANGE[1] # at 20.2km; 55 bins alreadu at 180-m resolution in R4
    # START_INDEX_R3 = LAYER_ALTITUDE_R3_INDEX_RANGE[0] # at 20.2km
    # END_INDEX_R3 = LAYER_ALTITUDE_R3_INDEX_RANGE[1] - 2 # at 8.2 km; 200 bins of 60-m in R3 => 200-2 = 198 => 198/3 = 66.0 bins of 180 m
    # NB_VERT_BINS_TO_AVERAGE_IN_R3 = 3 # resolution 180m on a 60m grid
    # NB_HORIZ_BINS_TO_AVERAGE = 15 # resolution 5km on a 333m grid
    # nb_vert_bins_180m_R4 = END_INDEX_R4 - START_INDEX_R4 + 1
    # nb_vert_bins_180m_R3 = int((END_INDEX_R3 - START_INDEX_R3 + 1)/3)
    
    # # Get first profile ID of chunk
    # if PREVIOUS_GRANULE:
    #     print("cal_l1_prev.prof_min:", cal_l1_prev.prof_min)
    #     prof_index_first_in_chunk = get_first_profileID_of_chunk(cal_l1_prev.prof_min)
    #     print("prof_index_first_in_chunk:", prof_index_first_in_chunk)
    # else:
    #     prof_index_first_in_chunk = get_first_profileID_of_chunk(cal_l1.prof_min)
    
    # # Get number of 5-km chunks
    # nb_chunk_5km = int(data_dict_cal_lid_l1["Latitude"][prof_index_first_in_chunk:].size/NB_HORIZ_BINS_TO_AVERAGE)
    # cal_lid_l1_prof_index_range_mult_of_15 = np.arange(prof_index_first_in_chunk, prof_index_first_in_chunk+nb_chunk_5km*15)

    # # 1-D vertical data at 180-m resolution
    # for key in ["Lidar_Data_Altitudes",]:
    #     # Initialization
    #     data_dict_5kmx180m[key] = np.ma.ones(nb_vert_bins_180m_R4+nb_vert_bins_180m_R3)*FILL_VALUE_FLOAT
    #     # Copy R4 region (20.2 - 30.1 km)
    #     data_dict_5kmx180m[key][:nb_vert_bins_180m_R4] = data_dict_cal_lid_l1[key][START_INDEX_R4:END_INDEX_R4+1]
    #     # Take middle altitude of 3 60-m vertical bins in R3 region (8.2 - 20.2 km)
    #     data_dict_5kmx180m[key][nb_vert_bins_180m_R4:] = data_dict_cal_lid_l1[key][START_INDEX_R3+1:END_INDEX_R3:3]

    # # 1-D horizontal data at 5-km resolution
    # for key in ["Latitude", "Longitude", "Number_Bins_Shift", "Profile_ID", "Profile_Time", "Profile_UTC_Time"]: 
    #     # Take middle (8th) profile of 5-km horizontal bins
    #     data_dict_5kmx180m[key] = data_dict_cal_lid_l1[key][cal_lid_l1_prof_index_range_mult_of_15][int(NB_HORIZ_BINS_TO_AVERAGE/2)::NB_HORIZ_BINS_TO_AVERAGE]
    
    # # 2-D data at 5-km×180-m resolution
    # key_list = ["Total_Attenuated_Backscatter_532", "Parallel_Attenuated_Backscatter_532", "Perpendicular_Attenuated_Backscatter_532", "Attenuated_Backscatter_1064",
    #             "Molecular_Total_Attenuated_Backscatter_532", "Molecular_Parallel_Attenuated_Backscatter_532", "Molecular_Perpendicular_Attenuated_Backscatter_532", "Molecular_Attenuated_Backscatter_1064",
    #             "Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Parallel", "Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Perpendicular", "Attenuated_Backscatter_Uncertainty_Standard_Deviation_1064",
    #             "Attenuated_Scattering_Ratio_532"]
    # if SAVE_DEVELOPMENT_DATA:
    #     key_list = key_list + ["Background_Noise_532_Parallel", "Background_Noise_532_Perpendicular", "Background_Noise_1064", 
    #                            "Shot_Noise_532_Parallel", "Shot_Noise_532_Perpendicular", "Shot_Noise_1064",
    #                            "Noise_Scale_Factor_532_Parallel_AB_domain", "Noise_Scale_Factor_532_Perpendicular_AB_domain", "Noise_Scale_Factor_1064_AB_domain"]
    # for key in key_list:
    #     # Initialization
    #     data_dict_5kmx180m[key] = np.ma.ones((nb_chunk_5km, nb_vert_bins_180m_R4+nb_vert_bins_180m_R3))*FILL_VALUE_FLOAT
    #     # Average data
    #     for prof_i in np.arange(nb_chunk_5km):
    #         for alt_i in np.arange(nb_vert_bins_180m_R4):
    #             data_dict_5kmx180m[key][prof_i, alt_i] =\
    #                 np.ma.mean(data_dict_cal_lid_l1[key][prof_index_first_in_chunk+prof_i*NB_HORIZ_BINS_TO_AVERAGE:prof_index_first_in_chunk+(prof_i+1)*NB_HORIZ_BINS_TO_AVERAGE,
    #                                                      START_INDEX_R4+alt_i])
    #         for alt_i in np.arange(nb_vert_bins_180m_R3):
    #             data_dict_5kmx180m[key][prof_i, nb_vert_bins_180m_R4+alt_i] =\
    #                 np.ma.mean(data_dict_cal_lid_l1[key][prof_index_first_in_chunk+prof_i*NB_HORIZ_BINS_TO_AVERAGE:prof_index_first_in_chunk+(prof_i+1)*NB_HORIZ_BINS_TO_AVERAGE,
    #                                                      START_INDEX_R3+alt_i*NB_VERT_BINS_TO_AVERAGE_IN_R3:START_INDEX_R3+(alt_i+1)*NB_VERT_BINS_TO_AVERAGE_IN_R3])

    # # Print number of profiles in the granule
    # print(f"\tNumber of 5-km profiles to process: {nb_chunk_5km}")

    # print_elapsed_time(tic)
    
    alt_idx_above_13km = np.where(data_dict_cal_lid_l1["Lidar_Data_Altitudes"] >= 13)
    par_signal = data_dict_cal_lid_l1["Parallel_Attenuated_Backscatter_532"][:, alt_idx_above_13km]
    per_signal = data_dict_cal_lid_l1["Perpendicular_Attenuated_Backscatter_532"][:, alt_idx_above_13km]

    # ************
    # *** Plot ***
    print("\n*****Plot...*****")

    setstyle('ticks_nogrid')

    bins_par = np.linspace(-1e-3, 3e-3, 1000)
    bins_per = np.linspace(-1e-3, 3e-3, 1000)
    gs0 = gridspec.GridSpec(2, 1, hspace=0.3)
    plt.figure(figsize=(8, 8))
    plt.subplot(gs0[0])
    plt.hist(par_signal.flatten(), bins=bins_par)
    plt.xlabel(r"$\beta'_{\parallel}$ (km$^{-1}$ sr$^{-1}$)")
    plt.ylabel("Occurrence")
    plt.ylim(0, 30000)
    plt.title('Parallel', weight='bold')
    plt.subplot(gs0[1])
    plt.hist(per_signal.flatten(), bins=bins_per)
    plt.xlabel(r"$\beta'_{\perp}$ (km$^{-1}$ sr$^{-1}$)")
    plt.ylabel("Occurrence")
    plt.ylim(0, 30000)
    plt.title('Perpendicular', weight='bold')
    plt.suptitle('CALIPSO/CALIOP signal distribution', weight='bold')
    filename = FIGURES_PATH+"CALIOP_signal_distribution.png"
    plt.savefig(filename, dpi=500)
    print("\t%s saved" % filename)

    print_time(tic_main_program)