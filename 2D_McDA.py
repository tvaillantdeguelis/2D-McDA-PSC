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
from my_modules.writers.netcdf_writer import NetCDFVariable, write_netcdf
from my_modules.calipso_calculator import compute_par_ab532, compute_ab_mol_and_b_mol, \
    nsf_from_V_domain_to_betap_domain, rms_from_P_domain_to_betap_domain, compute_shotnoise, \
    compute_backgroundnoise

from config import NB_PROF_OVERLAP
from feature_detection import detect_features, neighbors
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


class DataVar():
    def __init__(self, key, data):
        self.key = key
        self.data = data
        self.fillvalue = None
        self.units = ''
        self.description = ''
        self.dimensions = []
        self.valid_range = ()


def save_data(data_dict_5kmx180m, data_dict_2d_mcda, data_dict_2d_mcda_dev, filetype='HDF', save_development_data=False):

    # Create a dictionary of parameters to save
    params = {}

    key = 'Profile_ID'
    params[key] = DataVar(key, data_dict_5kmx180m["Profile_ID"])
    params[key].description = "Profile number from start of file"
    # params[key].valid_range = (1, 228630)
    params[key].dimensions = ['Profile_ID']

    key = 'Profile_Time'
    params[key] = DataVar(key, data_dict_5kmx180m["Profile_Time"])
    params[key].units = "seconds...TAI"
    # params[key].valid_range = (4.204e8, 1.072e9)
    params[key].dimensions = ['Profile_ID']

    key = 'Profile_UTC_Time'
    params[key] = DataVar(key, data_dict_5kmx180m["Profile_UTC_Time"])
    params[key].units = "UTC - yymmdd.ffffffff"
    # params[key].valid_range = (60426.0, 261231.0)
    params[key].dimensions = ['Profile_ID']

    key = 'Latitude'
    params[key] = DataVar(key, data_dict_5kmx180m["Latitude"])
    params[key].fillvalue = FILL_VALUE_FLOAT
    params[key].units = "degrees"
    # params[key].valid_range = (-90.0, 90.0)
    params[key].dimensions = ['Profile_ID']

    key = 'Longitude'
    params[key] = DataVar(key, data_dict_5kmx180m["Longitude"])
    params[key].fillvalue = FILL_VALUE_FLOAT
    params[key].units = "degrees"
    # params[key].valid_range = (-90.0, 90.0)
    params[key].dimensions = ['Profile_ID']
    
    key = 'Altitude'
    params[key] = DataVar(key, data_dict_5kmx180m["Lidar_Data_Altitudes"])
    params[key].units = "kilometer"
    # params[key].valid_range = (-0.5, 30.1)
    params[key].dimensions = ['Altitude']
    
    key = 'Parallel_Detection_Flags_532'
    params[key] = DataVar(key, data_dict_2d_mcda["Parallel_Detection_Flags_532"])
    params[key].valid_range = (0, 255)
    params[key].dimensions = ['Profile_ID', 'Altitude']

    key = 'Perpendicular_Detection_Flags_532'
    params[key] = DataVar(key, data_dict_2d_mcda["Perpendicular_Detection_Flags_532"])
    params[key].valid_range = (0, 255)
    params[key].dimensions = ['Profile_ID', 'Altitude']

    key = 'Detection_Flags_1064'
    params[key] = DataVar(key, data_dict_2d_mcda["Detection_Flags_1064"])
    params[key].valid_range = (0, 255)
    params[key].dimensions = ['Profile_ID', 'Altitude']

    key = 'Composite_Detection_Flags'
    params[key] = DataVar(key, data_dict_2d_mcda["Composite_Detection_Flags"])
    params[key].valid_range = (0, 255)
    params[key].dimensions = ['Profile_ID', 'Altitude']
    
    key = 'Parallel_Spikes_532'
    params[key] = DataVar(key, data_dict_2d_mcda_dev["Parallel_Spikes_532"])
    params[key].valid_range = (0, 1)
    params[key].dimensions = ['Profile_ID', 'Altitude']
    
    key = 'Perpendicular_Spikes_532'
    params[key] = DataVar(key, data_dict_2d_mcda_dev["Perpendicular_Spikes_532"])
    params[key].valid_range = (0, 1)
    params[key].dimensions = ['Profile_ID', 'Altitude']

    key = 'Spikes_1064'
    params[key] = DataVar(key, data_dict_2d_mcda_dev["Spikes_1064"])
    params[key].valid_range = (0, 1)
    params[key].dimensions = ['Profile_ID', 'Altitude']

    key = 'Homogeneous_Chunks_Mask'
    params[key] = DataVar(key, data_dict_2d_mcda["homogeneous_chunks_mask"])
    params[key].valid_range = (0, 255)
    params[key].dimensions = ['Profile_ID', 'Altitude']

    key = 'Homogeneous_Chunks_Classification'
    params[key] = DataVar(key, data_dict_2d_mcda["homogeneous_chunks_classification"])
    params[key].valid_range = (0, 255)
    params[key].dimensions = ['Profile_ID', 'Altitude']

    key = 'Homogeneous_Chunks_Mean_Parallel_Attenuated_Backscatter_532'
    params[key] = DataVar(key, data_dict_2d_mcda["homogeneous_chunks_mean_ab_532_par"])
    params[key].valid_range = (0, 255)
    params[key].dimensions = ['Profile_ID', 'Altitude']

    key = 'Homogeneous_Chunks_Mean_Perpendicular_Attenuated_Backscatter_532'
    params[key] = DataVar(key, data_dict_2d_mcda["homogeneous_chunks_mean_ab_532_per"])
    params[key].valid_range = (0, 255)
    params[key].dimensions = ['Profile_ID', 'Altitude']

    key = 'Homogeneous_Chunks_Mean_Attenuated_Backscatter_1064'
    params[key] = DataVar(key, data_dict_2d_mcda["homogeneous_chunks_mean_ab_1064"])
    params[key].valid_range = (0, 255)
    params[key].dimensions = ['Profile_ID', 'Altitude']

    key = 'Homogeneous_Chunks_Mean_Attenuated_Scattering_Ratio_532'
    params[key] = DataVar(key, data_dict_2d_mcda["homogeneous_chunks_mean_asr_532"])
    params[key].valid_range = (0, 255)
    params[key].dimensions = ['Profile_ID', 'Altitude']

    # Parameters saved for development
    if save_development_data:
        key = 'Parallel_Attenuated_Backscatter_532'
        params[key] = DataVar(key, data_dict_5kmx180m["Parallel_Attenuated_Backscatter_532"])
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].units = "km-1 sr-1"
        params[key].dimensions = ['Profile_ID', 'Altitude']
    
        key = 'Perpendicular_Attenuated_Backscatter_532'
        params[key] = DataVar(key, data_dict_5kmx180m["Perpendicular_Attenuated_Backscatter_532"])
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].units = "km-1 sr-1"
        params[key].dimensions = ['Profile_ID', 'Altitude']
    
        key = 'Attenuated_Backscatter_1064'
        params[key] = DataVar(key, data_dict_5kmx180m["Attenuated_Backscatter_1064"])
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].units = "km-1 sr-1"
        params[key].dimensions = ['Profile_ID', 'Altitude']

        key = 'Molecular_Parallel_Attenuated_Backscatter_532'
        params[key] = DataVar(key, data_dict_5kmx180m["Molecular_Parallel_Attenuated_Backscatter_532"])
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].units = "km-1 sr-1"
        params[key].dimensions = ['Profile_ID', 'Altitude']
        
        key = 'Noise_Scale_Factor_532_Parallel_AB_domain'        
        params[key] = DataVar(key, data_dict_5kmx180m["Noise_Scale_Factor_532_Parallel_AB_domain"])
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].units = "km-0.5 sr-0.5"
        params[key].dimensions = ['Profile_ID', 'Altitude']
        
        key = 'Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Parallel'        
        params[key] = DataVar(key, data_dict_5kmx180m["Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Parallel"])
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].units = "km-1 sr-1"
        params[key].dimensions = ['Profile_ID', 'Altitude']
        
        key = 'Background_Noise_532_Parallel'        
        params[key] = DataVar(key, data_dict_5kmx180m["Background_Noise_532_Parallel"])
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].units = "km-1 sr-1"
        params[key].dimensions = ['Profile_ID', 'Altitude']
        
        key = 'Shot_Noise_532_Parallel'        
        params[key] = DataVar(key, data_dict_5kmx180m["Shot_Noise_532_Parallel"])
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].units = "km-1 sr-1"
        params[key].dimensions = ['Profile_ID', 'Altitude']

        key = 'Molecular_Perpendicular_Attenuated_Backscatter_532'
        params[key] = DataVar(key, data_dict_5kmx180m["Molecular_Perpendicular_Attenuated_Backscatter_532"])
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].units = "km-1 sr-1"
        params[key].dimensions = ['Profile_ID', 'Altitude']
        
        key = 'Noise_Scale_Factor_532_Perpendicular_AB_domain'        
        params[key] = DataVar(key, data_dict_5kmx180m["Noise_Scale_Factor_532_Perpendicular_AB_domain"])
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].units = "km-0.5 sr-0.5"
        params[key].dimensions = ['Profile_ID', 'Altitude']
        
        key = 'Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Perpendicular'        
        params[key] = DataVar(key, data_dict_5kmx180m["Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Perpendicular"])
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].units = "km-1 sr-1"
        params[key].dimensions = ['Profile_ID', 'Altitude']
        
        key = 'Background_Noise_532_Perpendicular'        
        params[key] = DataVar(key, data_dict_5kmx180m["Background_Noise_532_Perpendicular"])
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].units = "km-1 sr-1"
        params[key].dimensions = ['Profile_ID', 'Altitude']
        
        key = 'Shot_Noise_532_Perpendicular'        
        params[key] = DataVar(key, data_dict_5kmx180m["Shot_Noise_532_Perpendicular"])
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].units = "km-1 sr-1"
        params[key].dimensions = ['Profile_ID', 'Altitude']

        key = 'Molecular_Attenuated_Backscatter_1064'
        params[key] = DataVar(key, data_dict_5kmx180m["Molecular_Attenuated_Backscatter_1064"])
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].units = "km-1 sr-1"
        params[key].dimensions = ['Profile_ID', 'Altitude']
        
        key = 'Noise_Scale_Factor_1064_AB_domain'        
        params[key] = DataVar(key, data_dict_5kmx180m["Noise_Scale_Factor_1064_AB_domain"])
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].units = "km-0.5 sr-0.5"
        params[key].dimensions = ['Profile_ID', 'Altitude']
        
        key = 'Attenuated_Backscatter_Uncertainty_Standard_Deviation_1064'        
        params[key] = DataVar(key, data_dict_5kmx180m["Attenuated_Backscatter_Uncertainty_Standard_Deviation_1064"])
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].units = "km-1 sr-1"
        params[key].dimensions = ['Profile_ID', 'Altitude']
        
        key = 'Background_Noise_1064'        
        params[key] = DataVar(key, data_dict_5kmx180m["Background_Noise_1064"])
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].units = "km-1 sr-1"
        params[key].dimensions = ['Profile_ID', 'Altitude']
        
        key = 'Shot_Noise_1064'        
        params[key] = DataVar(key, data_dict_5kmx180m["Shot_Noise_1064"])
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].units = "km-1 sr-1"
        params[key].dimensions = ['Profile_ID', 'Altitude']

        key = 'Step_532_par'
        params[key] = DataVar(key, np.arange(data_dict_2d_mcda_dev["Parallel_Detection_Flags_532_steps"].shape[0]))
        params[key].fillvalue = FILL_VALUE_SHORT
        params[key].dimensions = ['Step_532_par']

        key = 'Step_532_per'
        params[key] = DataVar(key, np.arange(data_dict_2d_mcda_dev["Perpendicular_Detection_Flags_532_steps"].shape[0]))
        params[key].fillvalue = FILL_VALUE_SHORT
        params[key].dimensions = ['Step_532_per']

        key = 'Step_1064'
        params[key] = DataVar(key, np.arange(data_dict_2d_mcda_dev["Detection_Flags_1064_steps"].shape[0]))
        params[key].fillvalue = FILL_VALUE_SHORT
        params[key].dimensions = ['Step_1064']

        key = 'Parallel_Detection_Flags_532_steps'
        params[key] = DataVar(key, data_dict_2d_mcda_dev["Parallel_Detection_Flags_532_steps"])
        params[key].valid_range = (0, 255)
        params[key].dimensions = ['Step_532_par', 'Profile_ID', 'Altitude']
    
        key = 'Perpendicular_Detection_Flags_532_steps'
        params[key] = DataVar(key, data_dict_2d_mcda_dev["Perpendicular_Detection_Flags_532_steps"])
        params[key].valid_range = (0, 255)
        params[key].dimensions = ['Step_532_per', 'Profile_ID', 'Altitude']
    
        key = 'Detection_Flags_1064_steps'
        params[key] = DataVar(key,  data_dict_2d_mcda_dev["Detection_Flags_1064_steps"])
        params[key].valid_range = (0, 255)
        params[key].dimensions = ['Step_1064', 'Profile_ID', 'Altitude']
    
        key = 'Parallel_Attenuated_Backscatter_532_steps'
        params[key] = DataVar(key, data_dict_2d_mcda_dev["Parallel_Attenuated_Backscatter_532_steps"])
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].units = "km-1 sr-1"
        params[key].dimensions = ['Step_532_par', 'Profile_ID', 'Altitude']
    
        key = 'Perpendicular_Attenuated_Backscatter_532_steps'
        params[key] = DataVar(key, data_dict_2d_mcda_dev["Perpendicular_Attenuated_Backscatter_532_steps"])
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].units = "km-1 sr-1"
        params[key].dimensions = ['Step_532_per', 'Profile_ID', 'Altitude']
    
        key = 'Attenuated_Backscatter_1064_steps'
        params[key] = DataVar(key, data_dict_2d_mcda_dev["Attenuated_Backscatter_1064_steps"])
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].units = "km-1 sr-1"
        params[key].dimensions = ['Step_1064', 'Profile_ID', 'Altitude']
    
    # Filename
    if (SLICE_START == 0 or SLICE_START == None) and (SLICE_END == None) and (SLICE_START_END_TYPE == 'profindex'):
        filename_end = '' # nothing, it is the whole file
    else:
        filename_end = f"_lon_{cal_l1.lon_min:.2f}_{cal_l1.lon_max:.2f}"
    filename = f"CAL_LID_L2_2D_McDA_PSCs-{TYPE_2D_McDA}-{VERSION_2D_McDA.replace('.', '-')}." \
                f"{GRANULE_DATE}{filename_end}"
    
    if filetype == 'HDF':
        hdf_params = {}
        for key, datavar in params.items():
            hdf_params[key] = SDSData(key, datavar.data, datavar.fillvalue)
            hdf_params[key].description = datavar.description
            hdf_params[key].units = datavar.units
            hdf_params[key].dim_labels = datavar.dimensions
        write_hdf(outdata_folder+"/"+filename+".hdf", hdf_params)
    elif filetype == 'netCDF':
        dim_keys = ['Profile_ID', 'Altitude', 'Step_532_par', 'Step_532_per', 'Step_1064']
        nc_params = []
        nc_dims = []
        for key, datavar in params.items():
            if key in dim_keys:
                nc_dim = NetCDFVariable(key, datavar.data)
                nc_dim.long_name = datavar.description
                nc_dim.fillvalue = datavar.fillvalue # might need to check if None if error
                nc_dim.units = datavar.units
                nc_dim.dimensions = datavar.dimensions
                nc_dims.append(nc_dim)
            else:
                nc_param = NetCDFVariable(key, datavar.data)
                nc_param.long_name = datavar.description
                nc_param.fillvalue = datavar.fillvalue # might need to check if None if error
                nc_param.units = datavar.units
                nc_param.dimensions = datavar.dimensions
                nc_params.append(nc_param)
        write_netcdf(outdata_folder+"/"+filename+".nc", nc_dims, nc_params)


def separate_homogeneous_chunks(mask_composite, mask_par532, mask_per532, mask_1064, separation_type):
    """Separate detected feature into homogeneous part based on the detection levels of the 3
    detection channel"""

    if separation_type == "channel": # separate by channel
        mask_homogeneous = np.ma.zeros(mask_composite.shape)
        # mask_homogeneous[ np.bitwise_and(mask_composite, int('000111', 2)) == 1] =  # 1: Clear air
        mask_homogeneous[(np.bitwise_and(mask_composite, int('000111', 2)) == 2) &\
                    (np.bitwise_and(mask_composite, int('111000', 2)) == 8)] = 2 # 2: 532 par only
        mask_homogeneous[(np.bitwise_and(mask_composite, int('000111', 2)) == 2) &\
                    (np.bitwise_and(mask_composite, int('111000', 2)) == 16)] = 3 # 3: 532 per only
        mask_homogeneous[(np.bitwise_and(mask_composite, int('000111', 2)) == 2) &\
                    (np.bitwise_and(mask_composite, int('111000', 2)) == 32)] = 4 # 4: 1064 only
        mask_homogeneous[(np.bitwise_and(mask_composite, int('000111', 2)) == 2) &\
                    (np.bitwise_and(mask_composite, int('111000', 2)) == (8+16))] = 5 # 5: 532 par + 532 per
        mask_homogeneous[(np.bitwise_and(mask_composite, int('000111', 2)) == 2) &\
                    (np.bitwise_and(mask_composite, int('111000', 2)) == (8+32))] = 6 # 6: 532 par + 1064
        mask_homogeneous[(np.bitwise_and(mask_composite, int('000111', 2)) == 2) &\
                    (np.bitwise_and(mask_composite, int('111000', 2)) == (16+32))] = 7 # 7: 532 per + 1064
        mask_homogeneous[(np.bitwise_and(mask_composite, int('000111', 2)) == 2) &\
                    (np.bitwise_and(mask_composite, int('111000', 2)) == (8+16+32))] = 8 # 8: 532 par + 532 per + 1064
        mask_homogeneous[ np.bitwise_and(mask_composite, int('000111', 2)) == 3] = 9
        mask_homogeneous[ np.bitwise_and(mask_composite, int('000111', 2)) == 5] = 10
        mask_homogeneous[ np.bitwise_and(mask_composite, int('000111', 2)) == 7] = 11

    elif separation_type == "best_detection_level": # separate by best detestion level
        # Best detection level mask
        detection_level_masks = []
        for i in np.arange(5) + 1:
            detection_level_masks.append((mask_par532 == i) + (mask_per532 == i) + (mask_1064 == i))

        # Initialization
        mask_homogeneous = np.ma.zeros(mask_par532.shape)

        for i in np.arange(5, 0, -1):
            mask_homogeneous[detection_level_masks[i-1]] = i

    elif separation_type == "all_levels_and_channels": # separate every levels and channels combination
        # Remove flags not in [1,2,3,4,5] like surface = 254, etc.
        mask_par532[mask_par532 > 5] = 0
        mask_per532[mask_per532 > 5] = 0
        mask_1064[mask_1064 > 5] = 0

        mask_homogeneous = mask_par532 + 10*mask_per532 + 100*mask_1064


    return mask_homogeneous


def mask_where_spikes(data, spikes_par, spikes_per):
    
    data[spikes_par == 1] = np.ma.masked
    data[spikes_per == 1] = np.ma.masked
    
    return data


def average_over_homogeneous_chunks(mask_homogeneous, ab_532_par, ab_532_per, ab_1064, sr_532, separation_type):

    # Initialization
    mask_shape = mask_homogeneous.shape
    seen_pixels = np.zeros(mask_shape, dtype=bool)
    ab_532_par_mean = np.ones(mask_shape)*FILL_VALUE_FLOAT
    ab_532_per_mean = np.ones(mask_shape)*FILL_VALUE_FLOAT
    ab_1064_mean = np.ones(mask_shape)*FILL_VALUE_FLOAT
    sr_532_mean = np.ones(mask_shape)*FILL_VALUE_FLOAT

    if separation_type == "channel":
        mask_values = np.arange(7)+2 # 2 to 8
    elif separation_type == "best_detection_level": 
        mask_values = np.arange(5)+1 # 1 to 5
    elif separation_type == "all_levels_and_channels": 
        mask_values = np.arange(1000)+1

    for i in np.arange(mask_shape[0]):
        for j in np.arange(mask_shape[1]):
            if separation_type == "pixel":
                ab_532_par_mean = np.copy(ab_532_par)
                ab_532_per_mean = np.copy(ab_532_per)
                ab_1064_mean = np.copy(ab_1064)
                sr_532_mean = np.copy(sr_532)
            else:
                if not seen_pixels[i, j]:
                    if mask_homogeneous[i, j] in mask_values: 
                        # Count neighbors
                        accessible_pixels = [(i, j)]
                        pattern_pixels = np.zeros(mask_shape, dtype=bool)
                        pattern_pixels[i, j] = True
                        while (len(accessible_pixels) != 0):
                            p = accessible_pixels[0] # 1st pixel of the list
                            accessible_pixels = accessible_pixels[1:] # Remove 1st
                            if not seen_pixels[p]:
                                seen_pixels[p] = True # We note that we see this pixel
                                v = neighbors(mask_shape, p) # Get pixel neighbors
                                # Look for neighbors
                                for voisin in v:
                                    c1 = not seen_pixels[voisin]
                                    c2 = mask_homogeneous[voisin] == mask_homogeneous[i, j]
                                    if c1 and c2:
                                        accessible_pixels.append(voisin)
                                        pattern_pixels[voisin] = True
                        
                        # Compute mean 532 TAB
                        sr_532_mean_feature = np.ma.mean(sr_532[pattern_pixels])
                        sr_532_mean[pattern_pixels] = sr_532_mean_feature

                        # Compute mean 532_par AB
                        ab_532_par_mean_feature = np.ma.mean(ab_532_par[pattern_pixels])
                        ab_532_par_mean[pattern_pixels] = ab_532_par_mean_feature

                        # Compute mean 532_per AB
                        ab_532_per_mean_feature = np.ma.mean(ab_532_per[pattern_pixels])
                        ab_532_per_mean[pattern_pixels] = ab_532_per_mean_feature

                        # Compute mean 1064 AB
                        ab_1064_mean_feature = np.ma.mean(ab_1064[pattern_pixels])
                        ab_1064_mean[pattern_pixels] = ab_1064_mean_feature

                        # If masked replace by fill value
                        try:
                            if ab_532_per_mean_feature.mask:
                                ab_532_per_mean_feature = FILL_VALUE_FLOAT
                        except:
                            pass
                        try:
                            if sr_532_mean_feature.mask:
                                sr_532_mean_feature = FILL_VALUE_FLOAT
                        except:
                            pass


    return ab_532_par_mean, ab_532_per_mean, ab_1064_mean, sr_532_mean


def classify_homogeneous_chunks_with_psc_v2(ab_532_per_mean, sr_532_mean):

    # Initialization
    psc_mask = np.zeros(ab_532_per_mean.shape)

    # Classification
    atb_per_thresh = 2.5e-6
    atb_per_enhanced_nat_thresh = 2e-5
    sr_532_thresh = 1.5
    sr_532_enhanced_nat_thresh = 2
    sr_532_ice_thresh = 3
    sr_532_wave_ice_thresh = 50
    psc_mask[(ab_532_per_mean < atb_per_thresh) & (sr_532_mean >= sr_532_thresh)] = 1 # STS
    psc_mask[(ab_532_per_mean >= atb_per_thresh) & (sr_532_mean >= sr_532_wave_ice_thresh)] = 6 # Wave ice
    psc_mask[(ab_532_per_mean >= atb_per_thresh) & (sr_532_mean >= sr_532_ice_thresh) & (sr_532_mean < sr_532_wave_ice_thresh)] = 4 # Ice
    psc_mask[(ab_532_per_mean >= atb_per_thresh) & (sr_532_mean < sr_532_ice_thresh)] = 2 # NAT
    psc_mask[(ab_532_per_mean >= atb_per_enhanced_nat_thresh) & (sr_532_mean >= sr_532_enhanced_nat_thresh) & (sr_532_mean < sr_532_ice_thresh)] = 5 # Enhanced NAT

    return psc_mask


if __name__ == '__main__':
    tic_main_program = print_time()
    
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    # PARAMETERS
    if False: #len(sys.argv) > 1:
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
        GRANULE_DATE = "2011-06-25T00-11-52ZN" # "2008-07-17T19-15-43ZN"
        VERSION_CAL_LID_L1 = "V4.51"
        TYPE_CAL_LID_L1 = "Standard"
        PREVIOUS_GRANULE = None
        NEXT_GRANULE = None
        SLICE_START_END_TYPE = "longitude" # "profindex" or "longitude" (Use "profindex" if SLICE_START/END = None to process the whole granule)
        SLICE_START = 5.95 # profindex or longitude
        SLICE_END = -150.07 # profindex or longitude
        SAVE_DEVELOPMENT_DATA = False # if True save step by step data
        VERSION_2D_McDA = "V1.2.0"
        TYPE_2D_McDA = "Prototype"
        OUT_FOLDER = "/home/vaillant/codes/projects/2D_McDA_for_PSCs/out/data/"    
        OUT_FILETYPE = 'HDF' # 'HDF' or 'netCDF'
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
            data_dict_2d_mcda["homogeneous_chunks_mask"] = np.ma.ones(data_dict_2d_mcda["Parallel_Detection_Flags_532"].shape) # not used
        else:
            # **************************************
            # *** Separated homogeneous features ***
            print("\n\n############################################################\n"\
                "*****Separated homogeneous features...*****")

            tic_algo = print_time()
        
            data_dict_2d_mcda["homogeneous_chunks_mask"] = \
                    separate_homogeneous_chunks(data_dict_2d_mcda["Composite_Detection_Flags"],
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

        # Mask where spikes
        ab_532_par = mask_where_spikes(data_dict_5kmx180m["Parallel_Attenuated_Backscatter_532"], 
                                       data_dict_2d_mcda_dev["Parallel_Spikes_532"], data_dict_2d_mcda_dev["Perpendicular_Spikes_532"])
        ab_532_per = mask_where_spikes(data_dict_5kmx180m["Perpendicular_Attenuated_Backscatter_532"], 
                                       data_dict_2d_mcda_dev["Parallel_Spikes_532"], data_dict_2d_mcda_dev["Perpendicular_Spikes_532"])
        ab_1064 = mask_where_spikes(data_dict_5kmx180m["Attenuated_Backscatter_1064"], 
                                    data_dict_2d_mcda_dev["Parallel_Spikes_532"], data_dict_2d_mcda_dev["Perpendicular_Spikes_532"])
        sr_532 = mask_where_spikes(data_dict_5kmx180m["Attenuated_Scattering_Ratio_532"], 
                                   data_dict_2d_mcda_dev["Parallel_Spikes_532"], data_dict_2d_mcda_dev["Perpendicular_Spikes_532"])

        data_dict_2d_mcda["homogeneous_chunks_mean_ab_532_par"], \
        data_dict_2d_mcda["homogeneous_chunks_mean_ab_532_per"], \
        data_dict_2d_mcda["homogeneous_chunks_mean_ab_1064"], \
        data_dict_2d_mcda["homogeneous_chunks_mean_asr_532"] = \
            average_over_homogeneous_chunks(data_dict_2d_mcda["homogeneous_chunks_mask"], 
                                            ab_532_par, ab_532_per, ab_1064, sr_532,
                                            separation_type=SEPARATION_TYPE)
        
        data_dict_2d_mcda["homogeneous_chunks_classification"] = \
            classify_homogeneous_chunks_with_psc_v2(data_dict_2d_mcda["homogeneous_chunks_mean_ab_532_per"],
                                                    data_dict_2d_mcda["homogeneous_chunks_mean_asr_532"])
        
        print_elapsed_time(tic_algo)


    # *****************
    # *** Save data ***
    print("\n\n############################################################\n"\
          "*****Save data...*****")
    
    # Create folder to store output data
    granule_date_dict = split_granule_date(GRANULE_DATE)
    outdata_folder = os.path.join(OUT_FOLDER, f"2D_McDA_PSCs.{VERSION_2D_McDA.replace('V', 'v')}",
                                  str(granule_date_dict['year']), f"{granule_date_dict['year']}_"
                                                                  f"{granule_date_dict['month']:02d}_"
                                                                  f"{granule_date_dict['day']:02d}")
    os.makedirs(outdata_folder, exist_ok=True)
    
    # Save the data
    save_data(data_dict_5kmx180m, data_dict_2d_mcda, data_dict_2d_mcda_dev, filetype=OUT_FILETYPE, save_development_data=SAVE_DEVELOPMENT_DATA)

    
    print_time(tic_main_program)
    