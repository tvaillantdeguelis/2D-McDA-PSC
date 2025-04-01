#!/usr/bin/env python
# coding: utf8

"""Main program of 2D-McDA. Takes CLaMS/ZOMM data files to process as input."""

__author__     = "Thibault Vaillant de Guélis"
__version__    = "1.01"
__email__      = "thibault.vaillantdeguelis@outlook.com"
__status__     = "Prototype"

import sys

import numpy as np
from datetime import datetime

from my_modules.readers.netcdf_reader import NetCDFReader
from my_modules.standard_outputs import print_time, print_elapsed_time
from feature_detection import detect_features
from my_modules.calipso_constants import *
from my_modules.writers.hdf_writer import SDSData, write_hdf
from merged_3channels_feature_mask import merged_feature_masks


if __name__ == '__main__':
    tic_main_program = print_time()
    
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    # PARAMETERS
    if len(sys.argv) > 1:
        ZOMM_CLAMS_FILENAME = sys.argv[1]
        ZOMM_CLAMS_PATH = sys.argv[2]
        SAVE_DEVELOPMENT_DATA = sys.argv[3]
        VERSION_2D_McDA = sys.argv[4]
        TYPE_2D_McDA = sys.argv[5]
        OUT_FOLDER = sys.argv[6]
        CNF = float(sys.argv[7])
    else:
        ZOMM_CLAMS_FILENAME = "PSC_ZOMM_CLAMS_BKS_2011d176_0000.nc"
        ZOMM_CLAMS_PATH = "/home/vaillant/codes/projects/2D_McDA_for_PSCs/in/CLaMS_ZOMM/"
        SAVE_DEVELOPMENT_DATA = False # if True save step by step data
        VERSION_2D_McDA = "V1.2.0"
        TYPE_2D_McDA = "Prototype"
        OUT_FOLDER = "/home/vaillant/codes/projects/2D_McDA_for_PSCs/out/data/CLaMS_ZOMM/"
        CNF = 0.00442 # CALIOP noise factor (see Sect. 3.1 in Tritscher et al., 2019), value from Lamont Poole computations
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

    # ********************************
    # *** Configuration parameters ***
    print("\n*****Configuration parameters...*****")

    print("\tZOMM_CLAMS_FILEPATH =", ZOMM_CLAMS_PATH+ZOMM_CLAMS_FILENAME)
    print("\tSAVE_DEVELOPMENT_DATA =", SAVE_DEVELOPMENT_DATA)
    print("\tVERSION_2D_McDA =", VERSION_2D_McDA)
    print("\tTYPE_2D_McDA =", TYPE_2D_McDA)
    print("\tOUT_FOLDER =", OUT_FOLDER)


    # ****************************
    # *** Load CLaMS/ZOMM data ***
    print("\n*****Load CLaMS/ZOMM data...*****")
    
    tic = datetime.now()

    with NetCDFReader(ZOMM_CLAMS_PATH+ZOMM_CLAMS_FILENAME) as data_reader:
        alt = data_reader.get_data('Altitude')[:]
        lat = data_reader.get_data('Latitude')[:]
        lon = data_reader.get_data('Longitude')[:]
        model_b532_par = data_reader.get_data('MODEL_B532_PAR')[:, :].T
        model_b532_per = data_reader.get_data('MODEL_B532_PER')[:, :].T
        model_b1064 = data_reader.get_data('MODEL_B1064')[:, :].T
        model_b532_par_err = data_reader.get_data('MODEL_B532_PAR_ERR')[:, :].T
        model_b532_per_err = data_reader.get_data('MODEL_B532_PER_ERR')[:, :].T
        model_b1064_err = data_reader.get_data('MODEL_B1064_ERR')[:, :].T
        model_b532_ray_par = data_reader.get_data('MODEL_B532_RAY_PAR')[:, :].T
        model_b532_ray_per = data_reader.get_data('MODEL_B532_RAY_PER')[:, :].T
        model_b1064_ray = data_reader.get_data('MODEL_B1064_RAY')[:, :].T
        model_h2o = data_reader.get_data('MODEL_H2O')[:, :].T
        model_hno3 = data_reader.get_data('MODEL_HNO3')[:, :].T
        model_h2oc_ice = data_reader.get_data('MODEL_H2OC_ICE')[:, :].T
        model_hno3c_nat = data_reader.get_data('MODEL_HNO3C_NAT')[:, :].T

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
        detect_features(model_b532_par, model_b532_ray_par, CNF*np.sqrt(model_b532_ray_par), '532_par')

    # Feature detection at 532 nm perpendicular
    # Note: use surface detection at 532 nm parallel
    data_dict_2d_mcda["Perpendicular_Detection_Flags_532"], \
    data_dict_2d_mcda_dev["Perpendicular_Detection_Flags_532_steps"], \
    data_dict_2d_mcda_dev["Perpendicular_Attenuated_Backscatter_532_steps"], \
    data_dict_2d_mcda_dev["Perpendicular_Spikes_532"] =\
        detect_features(model_b532_per, model_b532_ray_per, CNF*np.sqrt(model_b532_ray_per), '532_per')

    # Feature detection at 1064 nm
    data_dict_2d_mcda["Detection_Flags_1064"], \
    data_dict_2d_mcda_dev["Detection_Flags_1064_steps"], \
    data_dict_2d_mcda_dev["Attenuated_Backscatter_1064_steps"], \
    data_dict_2d_mcda_dev["Spikes_1064"] =\
        detect_features(model_b1064, model_b1064_ray, CNF*np.sqrt(model_b1064_ray), '1064')
    

    # *******************************************
    # *** Merged 3 channels feature detection ***
    print("\n\n*****Merged 3 channels feature detection...*****")

    data_dict_2d_mcda["Composite_Detection_Flags"] = \
        merged_feature_masks(data_dict_2d_mcda["Parallel_Detection_Flags_532"],
                                data_dict_2d_mcda["Perpendicular_Detection_Flags_532"],
                                data_dict_2d_mcda["Detection_Flags_1064"])
    

    # *****************************
    # *** Save data in HDF file ***
    print("\n\n############################################################\n"\
          "*****Save data in HDF file...*****")
    
        # Create a dictionary of parameters to save in HDF file
    params = {}
    
    # Add a profile ID to params
    params['prof_ID'] = SDSData('Profile_ID', np.arange(lat.size, dtype=np.int32))
    params['prof_ID'].dim_labels = ['Profile_ID']

    # Add alt to params
    params['alt'] = SDSData('Altitude', alt, FILL_VALUE_FLOAT)
    params['alt'].units = "km"
    params['alt'].dim_labels = ['Altitude']

    # Add lat to params
    params['lat'] = SDSData('Latitude', lat, FILL_VALUE_FLOAT)
    params['lat'].units = "degrees"
    params['lat'].dim_labels = ['Profile_ID']

    # Add lon to params
    params['lon'] = SDSData('Longitude', lon, FILL_VALUE_FLOAT)
    params['lon'].units = "degrees"
    params['lon'].dim_labels = ['Profile_ID']

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

    # Add signals to params
    params['model_b532_par'] =\
        SDSData('Parallel_Attenuated_Backscatter_532', model_b532_par)
    params['model_b532_par'].valid_range = (0, 255)
    params['model_b532_par'].dim_labels = ['Profile_ID', 'Altitude']

    # Add signals to params
    params['model_b532_per'] =\
        SDSData('Perpendicular_Attenuated_Backscatter_532', model_b532_per)
    params['model_b532_per'].valid_range = (0, 255)
    params['model_b532_per'].dim_labels = ['Profile_ID', 'Altitude']
    
    # Add signals to params
    params['model_b1064'] =\
        SDSData('Attenuated_Backscatter_1064', model_b1064)
    params['model_b1064'].valid_range = (0, 255)
    params['model_b1064'].dim_labels = ['Profile_ID', 'Altitude']

    # Add to params
    params['composite_detection_flags'] =\
        SDSData('Composite_Detection_Flags', data_dict_2d_mcda["Composite_Detection_Flags"])
    params['composite_detection_flags'].valid_range = (0, 255)
    params['composite_detection_flags'].dim_labels = ['Profile_ID', 'Altitude']

    # Add signals to params
    params['model_h2o'] =\
        SDSData('H2O', model_h2o)
    params['model_h2o'].valid_range = (0, 255)
    params['model_h2o'].dim_labels = ['Profile_ID', 'Altitude']

    # Add signals to params
    params['model_hno3'] =\
        SDSData('HNO3', model_hno3)
    params['model_hno3'].valid_range = (0, 255)
    params['model_hno3'].dim_labels = ['Profile_ID', 'Altitude']

    # Add signals to params
    params['model_h2oc_ice'] =\
        SDSData('H2OC_ICE', model_h2oc_ice)
    params['model_h2oc_ice'].valid_range = (0, 255)
    params['model_h2oc_ice'].dim_labels = ['Profile_ID', 'Altitude']

    # Add signals to params
    params['model_hno3c_nat'] =\
        SDSData('HNO3C_NAT', model_hno3c_nat)
    params['model_hno3c_nat'].valid_range = (0, 255)
    params['model_hno3c_nat'].dim_labels = ['Profile_ID', 'Altitude']

    # Parameters saved for development
    if SAVE_DEVELOPMENT_DATA:
    
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
    

    # Write in HDF file       
    filename = f"2D_McDA_PSCs-{ZOMM_CLAMS_FILENAME}"
    write_hdf(OUT_FOLDER+"/"+filename, params)
    
    
    print_time(tic_main_program)
    