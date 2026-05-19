#!/usr/bin/env python
# coding: utf8

"""Main program of 2D-McDA. Takes granule to process as input."""

__author__  = "Thibault Vaillant de Guélis"
__email__   = "thibault.vaillantdeguelis@outlook.com"
__version__ = "2.6.0"

import yaml
import sys
from pathlib import Path
import os
from datetime import datetime, timezone
import subprocess

import numpy as np
from scipy.interpolate import interp1d
from pyhdf.SD import SD, SDC

# sys.path.append("/home/vaillant/codes/projects/2D_McDA_PSC/my_modules")
sys.path.append("./my_modules/")
from standard_outputs import print_time, print_elapsed_time
from readers.calipso_reader import CALIOPReader, automatic_path_detection, get_first_profileID_of_chunk, range_from_altitude
from paths import split_granule_date
from calipso_constants import *
from constants import *
from writers.hdf_writer import SDSData, write_hdf
from writers.netcdf_writer import NetCDFVariable, write_netcdf
from calipso_calculator import compute_par_ab532, compute_ab_mol_and_b_mol, \
    nsf_from_V_domain_to_betap_domain, rms_from_P_domain_to_betap_domain, compute_shotnoise, \
    compute_backgroundnoise

from config import NB_PROF_OVERLAP
from feature_detection import detect_features, neighbors
from merged_3channels_feature_mask import merged_feature_masks


def git_version():
    """
    Return a string describing the current Git state of the code.

    The function runs:
        git describe --tags --dirty --always

    Typical returned values:
        - "v1.4.2"                       (exactly on a tag)
        - "v1.4.2-5-g7a3f9c2"             (5 commits after tag v1.4.2)
        - "v1.4.2-5-g7a3f9c2-dirty"       (uncommitted local changes)
        - "g7a3f9c2"                      (no tags available)

    If the code is not inside a Git repository or Git is not available,
    the function returns None.
    """
    try:
        # Call Git to obtain a human-readable description of the current commit
        git_desc = subprocess.check_output(
            ["git", "describe", "--tags", "--dirty", "--always"],
            stderr=subprocess.DEVNULL
        )

        # Convert bytes to string and remove trailing newline
        return git_desc.decode().strip()

    except Exception:
        # Git is not available or the code is not in a Git repository
        return None


def get_full_version():
    """
    Return the best possible version string for this code.

    Priority order:
        1. Use the Git-based version string (tag + commit hash),
           which uniquely identifies the exact code state.
        2. If Git information is unavailable, fall back to the static
           Python version defined in __version__.

    Returned values examples:
        - "v1.4.2"
        - "v1.4.2-5-g7a3f9c2"
        - "v1.4.2-5-g7a3f9c2-dirty"
    """
    git_ver = git_version()

    if git_ver is not None:
        # Git information available: use it as the authoritative version
        return git_ver
    else:
        # Fallback: use the static version defined in the code
        return f"v{__version__}"


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
    pattern = np.array([1.573, 1.345, 1.188, 1.131, 1.188, 1.345]) # This is for 30.1 to 20.2 km. TODO: Adapt depending on range altitude
    FCORR = np.tile(pattern, 1000)
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
        self.standard_name = ''
        self.long_name = ''
        self.axis = ''
        self.description = ''
        self.dimensions = []
        self.valid_range = ()
        self.comment = 'None'
        self.coordinates = ''
        self.flag_values = []
        self.flag_meanings = ''


def save_data(data_dict_5kmx180m, data_dict_2d_mcda, data_dict_2d_mcda_dev, filetype='netCDF', save_development_data=False):

    global_attrs = {

        "title": "CALIOP Level 2 Polar Stratospheric Cloud Mask Data",
        "summary": "CALIOP-derived Level 2 product providing Polar Stratospheric Cloud (PSC) mask and associated optical properties from CALIOP L1B observations combined with ancillary MERRA-2 and MLS data.",
        "product_version": "4.0",
        "Conventions": "CF-1.8",

        "institution": "NASA LaRC",
        "project": "CALIPSO / CALIOP derived products",
        "creator_name": __author__,
        "creator_email": "thibault.vaillantdeguelis@outlook.com",

        "platform": "CALIPSO",
        "instrument": "CALIOP",

        "source": f"CALIOP L1B {VERSION_CAL_LID_L1}, MERRA-2, MLS/Aura",

        "algorithm_name": "2D-McDA-PSC",
        "algorithm_version": get_full_version(),

        "processing_level": "L2",

        "history": "Created from CALIOP L1B V5.00 using 2D-McDA-PSC processing chain",
        "processing_date": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),

        "geospatial_lat_min": -90.0,
        "geospatial_lat_max": 90.0,
        "geospatial_lon_min": -180.0,
        "geospatial_lon_max": 180.0,

        "geospatial_vertical_min": 8.0,
        "geospatial_vertical_max": 31.0,
        "geospatial_vertical_units": "km",

        "spatial_resolution": "5 km horizontal, 180 m vertical",
        "vertical_levels": 121,

        "featureType": "trajectory",

        "time_coverage_description": (
            "Southern Hemisphere: lat > 50° S from May to October; "
            "Northern Hemisphere: lat > 50° N from December to March"
        ),

        "references": (
            "Vaillant de Guélis et al. (2021); "
            "Pitts et al. (2018)"
        )
    }
    
    # Create a dictionary of parameters to save
    params = {}

    key = 'Profile_ID'
    params[key] = DataVar(key, data_dict_5kmx180m["Profile_ID"])
    params[key].dimensions = ['Profile_ID']
    params[key].units = 1
    params[key].valid_range = (1, 228630)
    params[key].long_name = "CALIOP profile identifier"
    params[key].comment = "Unique profile identifier generated sequentially in ground processing for each laser pulse. Profile IDs are guaranteed to be unique within each L1B data file but not over multiple files. For this L2 product, the reported value corresponds to the center profile (8th of 15 consecutive profiles) within the original CALIOP L1B 5-km frame."
    params[key].coordinates = "Latitude Longitude"

    key = 'Profile_Time'
    params[key] = DataVar(key, data_dict_5kmx180m["Profile_Time"])
    params[key].dimensions = ['Profile_ID']
    params[key].units = "seconds since 1993-01-01 00:00:00"
    params[key].valid_range = (4.204e8, 1.072e9)
    params[key].long_name = "Profile time in International Atomic Time"
    params[key].comment = ": Laser firing time for each pulse, given in International Atomic Time (TAI) (i.e., in elapsed seconds from January 1, 1993). For this L2 product, the reported value corresponds to the center profile (8th of 15 consecutive profiles) within the original CALIOP L1B 5-km frame."
    params[key].coordinates = "Latitude Longitude"
    params[key].standard_name = "time"
    params[key].axis = "T"
    
    key = 'Profile_UTC_Time'
    params[key] = DataVar(key, data_dict_5kmx180m["Profile_UTC_Time"])
    params[key].dimensions = ['Profile_ID']
    params[key].units = "Coordinated Universal Time (UTC), formatted as 'yymmdd.ffffffff'"
    params[key].valid_range = (60428.0, 230701.0)
    params[key].long_name = "Profile time in Coordinated Universal Time"
    params[key].comment = "Laser firing time for each pulse, given in Coordinated Universal Time (UTC) and formatted as yymmdd.ffffffff, where yy is a two digit data acquisition year number (06 to 23), mm is a month number (01 to 12), dd is a day number (1 to 31), and ffffffff is the elapsed fraction of the data acquisition day. For this L2 product, the reported value corresponds to the center profile (8th of 15 consecutive profiles) within the original CALIOP L1B 5-km frame."
    params[key].coordinates = "Latitude Longitude"
    params[key].standard_name = "time"
    params[key].axis = "T"

    key = 'Latitude'
    params[key] = DataVar(key, data_dict_5kmx180m["Latitude"])
    params[key].dimensions = ['Profile_ID']
    params[key].units = "degree_north"
    params[key].valid_range = (-90.0, 90.0)
    params[key].fillvalue = FILL_VALUE_FLOAT  
    params[key].long_name = "Geodetic latitude"
    params[key].comment = "Geodetic latitude of the laser footprint on the Earth's surface. For this L2 product, the reported value corresponds to the center profile (8th of 15 consecutive profiles) within the original CALIOP L1B 5-km frame."
    params[key].standard_name = "latitude"
    params[key].axis = "Y"
    
    key = 'Longitude'
    params[key] = DataVar(key, data_dict_5kmx180m["Longitude"])
    params[key].dimensions = ['Profile_ID']
    params[key].units = "degree_east"
    params[key].valid_range = (-180.0, 180.0)
    params[key].fillvalue = FILL_VALUE_FLOAT
    params[key].long_name = "Longitude"
    params[key].comment = "Longitude of the laser footprint on the Earth's surface. For this L2 product, the reported value corresponds to the center profile (8th of 15 consecutive profiles) within the original CALIOP L1B 5-km frame."
    params[key].standard_name = "longitude"
    params[key].axis = "X"
    
    key = 'Altitude'
    params[key] = DataVar(key, data_dict_5kmx180m["Lidar_Data_Altitudes"])
    params[key].dimensions = ['Altitude']
    params[key].units = "km"
    params[key].valid_range = (8.0, 31.0)
    params[key].long_name = "Altitude above mean sea level"
    params[key].comment = "Altitudes at which the Level 2 PSC profile products are reported; consisting of 121 levels between approximately 8.3 and 30.1 km, with an interval of approximately 180 m. The altitudes are a subset of the standard lidar Level 1 profile altitudes."
    
    key = 'Parallel_Detection_Flags_532'
    params[key] = DataVar(key, data_dict_2d_mcda["Parallel_Detection_Flags_532"])
    params[key].dimensions = ['Profile_ID', 'Altitude']
    params[key].units = 1
    params[key].valid_range = (0, 5)
    params[key].fillvalue = FILL_VALUE_BYTE
    params[key].flag_values = [0, 1, 2, 3, 4, 5]
    params[key].flag_meanings = "no_detection detection_level_1 detection_level_2 detection_level_3 detection_level_4 detection_level_5"
    params[key].long_name = "532 nm parallel channel detection flags"
    params[key].comment = "Level of detection (1 to 5) mask for the 532 nm parallel channel from 2D-McDA (Vaillant de Guélis et al., 2021)."
    params[key].coordinates = "Latitude Longitude Profile_Time Altitude"
    
    key = 'Perpendicular_Detection_Flags_532'
    params[key] = DataVar(key, data_dict_2d_mcda["Perpendicular_Detection_Flags_532"])
    params[key].dimensions = ['Profile_ID', 'Altitude']
    params[key].units = 1
    params[key].valid_range = (0, 5)
    params[key].fillvalue = FILL_VALUE_BYTE
    params[key].flag_values = [0, 1, 2, 3, 4, 5]
    params[key].flag_meanings = "no_detection detection_level_1 detection_level_2 detection_level_3 detection_level_4 detection_level_5"
    params[key].long_name = "532 nm perpendicular channel detection flags"
    params[key].comment = "Level of detection (1 to 5) mask for the 532 nm perpendicular channel from 2D-McDA (Vaillant de Guélis et al., 2021)."
    params[key].coordinates = "Latitude Longitude Profile_Time Altitude"

    key = 'Detection_Flags_1064'
    params[key] = DataVar(key, data_dict_2d_mcda["Detection_Flags_1064"])
    params[key].dimensions = ['Profile_ID', 'Altitude']
    params[key].units = 1
    params[key].valid_range = (0, 5)
    params[key].fillvalue = FILL_VALUE_BYTE
    params[key].flag_values = [0, 1, 2, 3, 4, 5]
    params[key].flag_meanings = "no_detection detection_level_1 detection_level_2 detection_level_3 detection_level_4 detection_level_5"
    params[key].long_name = "1064 nm channel detection flags"
    params[key].comment = "Level of detection (1 to 5) mask for the 1064 nm channel from 2D-McDA (Vaillant de Guélis et al., 2021)."
    params[key].coordinates = "Latitude Longitude Profile_Time Altitude"

    if False:
        key = 'Composite_Detection_Flags'
        params[key] = DataVar(key, data_dict_2d_mcda["Composite_Detection_Flags"])
        params[key].comment = "Composite detection mask from the 3 detection channels."
        params[key].valid_range = (0, 255)
        params[key].dimensions = ['Profile_ID', 'Altitude']
    
    if False:
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

    if MAKE_CLASSIFICATION:
        key = 'PSC_Composition'
        params[key] = DataVar(key, data_dict_2d_mcda["homogeneous_chunks_classification"])
        params[key].dimensions = ['Profile_ID', 'Altitude']
        params[key].units = 1
        params[key].valid_range = (-4, 6)
        params[key].fillvalue = FILL_VALUE_BYTE
        params[key].flag_values = [-4, 0, 1, 2, 3, 4, 5, 6]
        params[key].flag_meanings = "likely_tropo no_detection sts nat sbs ice enhanced_nat wave_ice"
        params[key].long_name = "Polar stratospheric cloud composition classification"
        params[key].comment = "PSC composition reports information on the composition of the detected PSC. The composition is determined based on the retrieved lidar optical parameters in terms of attenuated total scattering ratio and perpendicular backscatter at 532 nm averaged over homogeneous chunks."
        params[key].coordinates = "Latitude Longitude Profile_Time Altitude"
    
        if False:
            key = 'Homogeneous_Chunks_Mask'
            params[key] = DataVar(key, data_dict_2d_mcda["homogeneous_chunks_mask"])
            params[key].valid_range = (0, 255)
            params[key].dimensions = ['Profile_ID', 'Altitude']

            key = 'Homogeneous_Chunks_Mean_Parallel_Attenuated_Backscatter_532'
            params[key] = DataVar(key, data_dict_2d_mcda["homogeneous_chunks_mean_ab_532_par"])
            params[key].comment = "532-nm parallel attenuated backscatter signal averaged on homogeneous chunks."
            params[key].fillvalue = FILL_VALUE_FLOAT
            params[key].valid_range = (0, 255)
            params[key].dimensions = ['Profile_ID', 'Altitude']

        key = 'Homogeneous_Chunks_Mean_Particulate_Parallel_Attenuated_Backscatter_532'
        params[key] = DataVar(key, data_dict_2d_mcda["homogeneous_chunks_mean_part_ab_532_par"])
        params[key].dimensions = ['Profile_ID', 'Altitude']
        params[key].units = "1/(km * sr)"
        params[key].valid_range = (-0.1, 3.3)
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].long_name = "Mean 532 nm particulate parallel attenuated backscatter over homogeneous chunks"
        params[key].comment = "Particulate parallel attenuated backscatter at 532 nm averaged over detected homogeneous chunks."
        params[key].coordinates = "Latitude Longitude Profile_Time Altitude"

        if False:
            key = 'Homogeneous_Chunks_Mean_Perpendicular_Attenuated_Backscatter_532'
            params[key] = DataVar(key, data_dict_2d_mcda["homogeneous_chunks_mean_ab_532_per"])
            params[key].comment = "532-nm perpendicular attenuated backscatter signal averaged on homogeneous chunks."
            params[key].fillvalue = FILL_VALUE_FLOAT
            params[key].valid_range = (0, 255)
            params[key].dimensions = ['Profile_ID', 'Altitude']

        key = 'Homogeneous_Chunks_Mean_Particulate_Perpendicular_Attenuated_Backscatter_532'
        params[key] = DataVar(key, data_dict_2d_mcda["homogeneous_chunks_mean_part_ab_532_per"])
        params[key].dimensions = ['Profile_ID', 'Altitude']
        params[key].units = "1/(km * sr)"
        params[key].valid_range = (-0.08, 1.7)
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].long_name = "Mean 532 nm particulate perpendicular attenuated backscatter over homogeneous chunks"
        params[key].comment = "Particulate perpendicular attenuated backscatter at 532 nm averaged over detected homogeneous chunks."
        params[key].coordinates = "Latitude Longitude Profile_Time Altitude"

        if False:
            key = 'Homogeneous_Chunks_Mean_Total_Attenuated_Backscatter_1064'
            params[key] = DataVar(key, data_dict_2d_mcda["homogeneous_chunks_mean_ab_1064"])
            params[key].comment = "1064-nm attenuated backscatter signal averaged on homogeneous chunks."
            params[key].fillvalue = FILL_VALUE_FLOAT
            params[key].valid_range = (0, 255)
            params[key].dimensions = ['Profile_ID', 'Altitude']

        key = 'Homogeneous_Chunks_Mean_Particulate_Total_Attenuated_Backscatter_1064'
        params[key] = DataVar(key, data_dict_2d_mcda["homogeneous_chunks_mean_part_ab_1064"])
        params[key].dimensions = ['Profile_ID', 'Altitude']
        params[key].units = "1/(km * sr)"
        params[key].valid_range = (-0.04, 2.5)
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].long_name = "Mean 1064 nm particulate total attenuated backscatter over homogeneous chunks"
        params[key].comment = "Particulate total attenuated backscatter at 1064 nm averaged over detected homogeneous chunks."
        params[key].coordinates = "Latitude Longitude Profile_Time Altitude"

        key = 'Homogeneous_Chunks_Mean_Total_Attenuated_Scattering_Ratio_532'
        params[key] = DataVar(key, data_dict_2d_mcda["homogeneous_chunks_mean_asr_532"])
        params[key].dimensions = ['Profile_ID', 'Altitude']
        params[key].units = 1
        params[key].valid_range = (0.0, 99.0)
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].long_name = "Mean 532 nm total attenuated scattering ratio over homogeneous chunks"
        params[key].comment = "Total attenuated scattering ratio at 532 nm averaged over detected homogeneous chunks."
        params[key].coordinates = "Latitude Longitude Profile_Time Altitude"

        key = 'PSC_Ice_Mixture_Boundary'
        params[key] = DataVar(key, data_dict_5kmx180m["nat_ice_R_threshold"])
        params[key].dimensions = ['Profile_ID', 'Altitude']
        params[key].units = 1
        params[key].valid_range = (-1.0, 10.0)
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].long_name = "PSC ice mixture boundary scattering ratio"
        params[key].comment = "Value of Total Scattering Ratio at 532 nm (no units) defining the boundary between NAT mixture and ice PSCs, reported for each Level 2 profile at the 121 standard altitudes in the Altitude field. This value is dependent on the amount of available condensable nitric acid and water in the stratosphere (see Pitts et al., 2018 for more detail)."
        params[key].coordinates = "Latitude Longitude Profile_Time Altitude"

        key = 'Homogeneous_Chunks_Mean_PSC_Ice_Mixture_Boundary'
        params[key] = DataVar(key, data_dict_2d_mcda["homogeneous_chunks_mean_nat_ice_R_threshold"])
        params[key].dimensions = ['Profile_ID', 'Altitude']
        params[key].units = 1
        params[key].valid_range = (-1.0, 10.0)
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].long_name = "Mean PSC ice mixture boundary scattering ratio over homogeneous chunks"
        params[key].comment = "PSC Ice Mixture Boundary averaged over detected homogeneous chunks."
        params[key].coordinates = "Latitude Longitude Profile_Time Altitude"

        key = 'Homogeneous_Chunks_Mean_Temperature'
        params[key] = DataVar(key, data_dict_2d_mcda["homogeneous_chunks_mean_temperature"])
        params[key].dimensions = ['Profile_ID', 'Altitude']
        params[key].units = "K"
        params[key].valid_range = (-1.0, 10.0)
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].long_name = "Mean air temperature over homogeneous chunks"
        params[key].comment = "Air temperature averaged over detected homogeneous chunks."
        params[key].coordinates = "Latitude Longitude Profile_Time Altitude"

        key = 'Pressure'
        params[key] = DataVar(key, data_dict_5kmx180m["press"])
        params[key].dimensions = ['Profile_ID', 'Altitude']
        params[key].units = "hPa"
        params[key].valid_range = (1.0, 1000.0)
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].long_name = "Air pressure"
        params[key].comment = "Atmospheric pressure reported for each Level 2 profile at the 121 standard altitudes in the Altitude field. Pressure values are derived from the ancillary meteorological data provided by the MERRA-2."
        params[key].coordinates = "Latitude Longitude Profile_Time Altitude"
        
        key = 'Temperature'
        params[key] = DataVar(key, data_dict_5kmx180m["temp"])
        params[key].dimensions = ['Profile_ID', 'Altitude']
        params[key].units = "K"
        params[key].valid_range = (150.0, 350.0)
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].long_name = "Air temperature"
        params[key].comment = "Temperature reported for each Level 2 profile at the 121 standard altitudes in the Altitude field. Temperature values are interpolated from the ancillary meteorological data provided by the MERRA-2."
        params[key].coordinates = "Latitude Longitude Profile_Time Altitude"

        key = 'Tropopause_Altitude_MERRA2'
        params[key] = DataVar(key, data_dict_5kmx180m["tropopause"])
        params[key].dimensions = ['Profile_ID', ]
        params[key].units = "km"
        params[key].valid_range = (3.0, 25.0)
        params[key].fillvalue = FILL_VALUE_FLOAT
        params[key].long_name = "MERRA-2 blended tropopause altitude"
        params[key].comment = "Mean tropopause height in kilometers above local mean sea level. Tropopause height information is based on the MERRA-2 “blended” tropopause altitudes. The MERRA-2 blended tropopause is the lower (in altitude) of the temperature-based (“thermal”) tropopause and potential vorticity (PV)-based (“dynamic”) tropopause (Bosilovich et al., 2016; Ott et al., 2016)."
        params[key].coordinates = "Latitude Longitude Profile_Time"

        key = 'Pressure_HNO3'
        params[key] = DataVar(key, data_dict_2d_mcda["Pressure_HNO3"])
        params[key].dimensions = ['Pressure_HNO3',]
        params[key].units = "hPa"
        params[key].valid_range = (1.0, 500.0)
        params[key].long_name = "Aura MLS HNO3 pressure levels"
        params[key].comment = "Pressure levels reported for each Aura MLS HNO3 profile."

        key = 'Pressure_H2O'
        params[key] = DataVar(key, data_dict_2d_mcda["Pressure_H2O"])
        params[key].dimensions = ['Pressure_H2O',]
        params[key].units = "hPa"
        params[key].valid_range = (1.0, 500.0)
        params[key].long_name = "Aura MLS H2O pressure levels"
        params[key].comment = "Pressure levels reported for each Aura MLS H2O profile."

        key = 'HNO3_Mixing_Ratio'
        params[key] = DataVar(key, data_dict_5kmx180m["HNO3_Mixing_Ratio"])
        params[key].dimensions = ['Profile_ID', 'Pressure_HNO3']
        params[key].units = 1
        params[key].valid_range = (0.0, 2.0e-8)
        params[key].long_name = "Aura MLS HNO3 mixing ratio"
        params[key].comment = "Profiles of Aura MLS HNO3 (Manney et al., 2015) mixing ratios derived from the publicly available MLS/Aura Level 2 V4 HNO3(https://disc.gsfc.nasa.gov/datasets/ML2HNO3_004/summary) data product. These are reported on their standard vertical pressure grid but have been interpolated horizontally to the CALIOP profile locations along each CALIPSO orbit track."
        params[key].coordinates = "Latitude Longitude Profile_Time Pressure_HNO3"

        key = 'H2O_Mixing_Ratio'
        params[key] = DataVar(key, data_dict_5kmx180m["H2O_Mixing_Ratio"])
        params[key].dimensions = ['Profile_ID', 'Pressure_H2O']
        params[key].units = 1
        params[key].valid_range = (0.0, 2.0e-8)
        params[key].long_name = "Aura MLS H2O mixing ratio"
        params[key].comment = "Profiles of Aura MLS H2O (Lambert et al., 2015) mixing ratios derived from the publicly available  MLS/Aura Level 2 V4 H2O (https://disc.gsfc.nasa.gov/datasets/ML2H2O_004/summary) data product. These are reported on their standard vertical pressure grid but have been interpolated horizontally to the CALIOP profile locations along each CALIPSO orbit track"
        params[key].coordinates = "Latitude Longitude Profile_Time Pressure_H2O"
        

    key = 'Parallel_Attenuated_Backscatter_532'
    params[key] = DataVar(key, data_dict_5kmx180m["Parallel_Attenuated_Backscatter_532"])
    params[key].dimensions = ['Profile_ID', 'Altitude']
    params[key].units = "1/(km * sr)"
    params[key].valid_range = (-0.1, 3.3)
    params[key].fillvalue = FILL_VALUE_FLOAT
    params[key].long_name = "532 nm parallel attenuated backscatter"
    params[key].comment = "Parallel attenuated backscatter at 532 nm reported for each Level 2 profile at the 121 standard altitudes in the Altitude field. The initial values of the CALIOP Level 1 parallel component of the 532-nm total attenuated backscatter averaged to 180-m vertical and 5-km horizontal resolution. Level 1 profiles of the parallel component of the attenuated backscatter are obtained by simple subtraction of the perpendicular component from the total. This variable corresponds to Parallel_Attenuated_Backscatter_532_Initial in PSC Mask V3.00."
    params[key].coordinates = "Latitude Longitude Profile_Time Altitude"
 
    key = 'Perpendicular_Attenuated_Backscatter_532'
    params[key] = DataVar(key, data_dict_5kmx180m["Perpendicular_Attenuated_Backscatter_532"])
    params[key].dimensions = ['Profile_ID', 'Altitude']
    params[key].units = "1/(km * sr)"
    params[key].valid_range = (-0.08, 1.7)
    params[key].fillvalue = FILL_VALUE_FLOAT
    params[key].long_name = "532 nm perpendicular attenuated backscatter"
    params[key].comment = "Perpendicular attenuated backscatter at 532 nm reported for each Level 2 profile at the 121 standard altitudes in the Altitude field. The initial values of the CALIOP Level 1 perpendicular component of the 532-nm total attenuated backscatter averaged to 180-m vertical and 5-km horizontal resolution. This variable corresponds to Perpendicular_Attenuated_Backscatter_532_Initial in PSC Mask V3.00."
    params[key].coordinates = "Latitude Longitude Profile_Time Altitude"

    key = 'Total_Attenuated_Backscatter_1064'
    params[key] = DataVar(key, data_dict_5kmx180m["Attenuated_Backscatter_1064"])
    params[key].dimensions = ['Profile_ID', 'Altitude']
    params[key].units = "1/(km * sr)"
    params[key].valid_range = (-0.04, 2.5)
    params[key].fillvalue = FILL_VALUE_FLOAT
    params[key].long_name = "1064 nm total attenuated backscatter"
    params[key].comment = "Total attenuated backscatter at 1064 nm reported for each Level 2 profile at the 121 standard altitudes in the Altitude field. The values are reported at 180-m vertical and 5-km horizontal resolution."
    params[key].coordinates = "Latitude Longitude Profile_Time Altitude"

    key = 'Total_Attenuated_Scattering_Ratio_532'
    params[key] = DataVar(key, data_dict_5kmx180m["Attenuated_Scattering_Ratio_532"])
    params[key].dimensions = ['Profile_ID', 'Altitude']
    params[key].units = 1
    params[key].valid_range = (0.0, 99.0)
    params[key].fillvalue = FILL_VALUE_FLOAT
    params[key].long_name = "532 nm total attenuated scattering ratio"
    params[key].comment = "Ratio of the total attenuated backscatter at 532 nm to the molecular backscatter at 532 nm, no units, reported for each Level 2 profile at the 121 standard altitudes in the Altitude field. The values are reported at 180-m vertical and 5-km horizontal resolution."
    params[key].coordinates = "Latitude Longitude Profile_Time Altitude"

    key = 'Particulate_Parallel_Attenuated_Backscatter_532'
    params[key] = DataVar(key, data_dict_5kmx180m["Particulate_Parallel_Attenuated_Backscatter_532"])
    params[key].dimensions = ['Profile_ID', 'Altitude']
    params[key].units = "1/(km * sr)"
    params[key].valid_range = (-0.1, 3.3)
    params[key].fillvalue = FILL_VALUE_FLOAT
    params[key].long_name = "532 nm particulate parallel attenuated backscatter"
    params[key].comment = "Particulate parallel attenuated backscatter at 532 nm computed by subtracting the estimated molecular parallel attenuated backscatter at 532 nm to the parallel attenuated backscatter at 532 nm. The values are reported at 180-m vertical and 5-km horizontal resolution."
    params[key].coordinates = "Latitude Longitude Profile_Time Altitude"

    key = 'Particulate_Perpendicular_Attenuated_Backscatter_532'
    params[key] = DataVar(key, data_dict_5kmx180m["Particulate_Perpendicular_Attenuated_Backscatter_532"])
    params[key].dimensions = ['Profile_ID', 'Altitude']
    params[key].units = "1/(km * sr)"
    params[key].valid_range = (-0.08, 1.7)
    params[key].fillvalue = FILL_VALUE_FLOAT
    params[key].long_name = "532 nm particulate perpendicular attenuated backscatter"
    params[key].comment = "Particulate perpendicular attenuated backscatter at 532 nm computed by subtracting the estimated molecular perpendicular attenuated backscatter at 532 nm to the perpendicular attenuated backscatter at 532 nm. The values are reported at 180-m vertical and 5-km horizontal resolution."
    params[key].coordinates = "Latitude Longitude Profile_Time Altitude"

    key = 'Particulate_Total_Attenuated_Backscatter_1064'
    params[key] = DataVar(key, data_dict_5kmx180m["Particulate_Attenuated_Backscatter_1064"])
    params[key].dimensions = ['Profile_ID', 'Altitude']
    params[key].units = "1/(km * sr)"
    params[key].valid_range = (-0.04, 2.5)
    params[key].fillvalue = FILL_VALUE_FLOAT
    params[key].long_name = "1064 nm particulate total attenuated backscatter"
    params[key].comment = "Particulate attenuated backscatter at 1064 nm computed by subtracting the estimated molecular attenuated backscatter at 1064 nm to the attenuated backscatter at 1064 nm. The values are reported at 180-m vertical and 5-km horizontal resolution."
    params[key].coordinates = "Latitude Longitude Profile_Time Altitude"     

    key = 'Molecular_Backscatter_532'
    params[key] = DataVar(key, data_dict_5kmx180m["Molecular_Attenuated_Backscatter_1064"])
    params[key].dimensions = ['Profile_ID', 'Altitude']
    params[key].units = "1/(km * sr)"
    params[key].valid_range = (0.0, 0.1)
    params[key].fillvalue = FILL_VALUE_FLOAT
    params[key].long_name = "532 nm molecular backscatter"
    params[key].comment = "Molecular backscatter at 532 nm reported for each Level 2 profile at the 121 standard altitudes in the Altitude field. The values are reported at 180-m vertical and 5-km horizontal resolution."
    params[key].coordinates = "Latitude Longitude Profile_Time Altitude"  

    key = 'Molecular_Backscatter_1064'
    params[key] = DataVar(key, data_dict_5kmx180m["Molecular_Attenuated_Backscatter_1064"])
    params[key].dimensions = ['Profile_ID', 'Altitude']
    params[key].units = "1/(km * sr)"
    params[key].valid_range = (0.0, 0.1)
    params[key].fillvalue = FILL_VALUE_FLOAT
    params[key].long_name = "1064 nm molecular backscatter"
    params[key].comment = "Molecular backscatter at 1064 nm reported for each Level 2 profile at the 121 standard altitudes in the Altitude field. The values are reported at 180-m vertical and 5-km horizontal resolution."
    params[key].coordinates = "Latitude Longitude Profile_Time Altitude"  
    
    # Parameters saved for development
    if save_development_data:
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
        params[key] = DataVar(key, np.ma.arange(data_dict_2d_mcda_dev["Parallel_Detection_Flags_532_steps"].shape[0], dtype=np.int32))
        params[key].fillvalue = FILL_VALUE_SHORT
        params[key].dimensions = ['Step_532_par']

        key = 'Step_532_per'
        params[key] = DataVar(key, np.ma.arange(data_dict_2d_mcda_dev["Perpendicular_Detection_Flags_532_steps"].shape[0], dtype=np.int32))
        params[key].fillvalue = FILL_VALUE_SHORT
        params[key].dimensions = ['Step_532_per']

        key = 'Step_1064'
        params[key] = DataVar(key, np.ma.arange(data_dict_2d_mcda_dev["Detection_Flags_1064_steps"].shape[0], dtype=np.int32))
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
    if (SLICE_START == 0 or SLICE_START == None) and (SLICE_END == None):
        filename_end = '' # nothing, it is the whole file
    else:
        filename_end = f"_lon_{cal_l1.lon_min:.2f}_{cal_l1.lon_max:.2f}"
    filename = f"CAL_LID_L2_2D_McDA_PSC-{TYPE_2D_McDA_PSC}-{VERSION_2D_McDA_PSC.replace('.', '-').replace('v', 'V', 1)}." \
                f"{GRANULE_DATE}{filename_end}"
    
    if filetype == 'HDF':
        hdf_params = {}
        for key, datavar in params.items():
            hdf_params[key] = SDSData(key, datavar.data, datavar.fillvalue)
            hdf_params[key].comment = datavar.description
            hdf_params[key].units = datavar.units
            hdf_params[key].dim_labels = datavar.dimensions
        write_hdf(outdata_folder+"/"+filename+".hdf", hdf_params)
    elif filetype == 'netCDF':
        dim_keys = ['Profile_ID', 'Altitude', 'Pressure_HNO3', 'Pressure_H2O', 'Step_532_par', 'Step_532_per', 'Step_1064']
        nc_params = []
        nc_dims = []
        for key, datavar in params.items():
            if key in dim_keys:
                nc_dim = NetCDFVariable(key, datavar.data)
                nc_dim.standard_name = datavar.standard_name
                nc_dim.long_name = datavar.long_name
                nc_dim.axis = datavar.axis
                nc_dim.valid_range = datavar.valid_range
                nc_dim.comment = datavar.comment
                nc_dim.coordinates = datavar.coordinates
                nc_dim.fillvalue = datavar.fillvalue # might need to check if None if error
                nc_dim.units = datavar.units
                nc_dim.dimensions = datavar.dimensions
                nc_dims.append(nc_dim)
            else:
                nc_param = NetCDFVariable(key, datavar.data)
                nc_param.standard_name = datavar.standard_name
                nc_param.long_name = datavar.long_name
                nc_dim.axis = datavar.axis
                nc_param.valid_range = datavar.valid_range
                nc_param.comment = datavar.comment
                nc_param.coordinates = datavar.coordinates
                nc_param.flag_values = datavar.flag_values
                nc_param.flag_meanings = datavar.flag_meanings
                nc_param.fillvalue = datavar.fillvalue # might need to check if None if error
                nc_param.units = datavar.units
                nc_param.dimensions = datavar.dimensions
                nc_params.append(nc_param)
        write_netcdf(outdata_folder+"/"+filename+".nc", nc_dims, nc_params, global_attrs=global_attrs)


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


def average_over_homogeneous_chunks(mask_homogeneous, ab_532_par, ab_532_per, ab_1064, sr_532, nat_ice_R_threshold, temperature, separation_type):

    # Initialization
    mask_shape = mask_homogeneous.shape
    seen_pixels = np.zeros(mask_shape, dtype=bool)
    ab_532_par_mean = np.ones(mask_shape)*FILL_VALUE_FLOAT
    ab_532_per_mean = np.ones(mask_shape)*FILL_VALUE_FLOAT
    ab_1064_mean = np.ones(mask_shape)*FILL_VALUE_FLOAT
    sr_532_mean = np.ones(mask_shape)*FILL_VALUE_FLOAT
    nat_ice_R_threshold_mean = np.ones(mask_shape)*FILL_VALUE_FLOAT
    temperature_mean = np.ones(mask_shape)*FILL_VALUE_FLOAT

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
                nat_ice_R_threshold_mean = np.copy(nat_ice_R_threshold)
                temperature_mean = np.copy(temperature)
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

                        # Compute mean NAT/ice threshold
                        nat_ice_R_threshold_mean_feature = np.ma.mean(nat_ice_R_threshold[pattern_pixels])
                        nat_ice_R_threshold_mean[pattern_pixels] = nat_ice_R_threshold_mean_feature

                        # Temperature
                        temperature_mean_feature = np.ma.mean(temperature[pattern_pixels])
                        temperature_mean[pattern_pixels] = temperature_mean_feature

                        # If masked replace by fill value
                        try:
                            if sr_532_mean.mask:
                                sr_532_mean = FILL_VALUE_FLOAT
                        except:
                            pass
                        try:
                            if ab_532_par_mean.mask:
                                ab_532_par_mean = FILL_VALUE_FLOAT
                        except:
                            pass
                        try:
                            if ab_532_per_mean.mask:
                                ab_532_per_mean = FILL_VALUE_FLOAT
                        except:
                            pass
                        try:
                            if ab_1064_mean.mask:
                                ab_1064_mean = FILL_VALUE_FLOAT
                        except:
                            pass
                        try:
                            if nat_ice_R_threshold_mean.mask:
                                nat_ice_R_threshold_mean = FILL_VALUE_FLOAT
                        except:
                            pass
                        try:
                            if temperature_mean.mask:
                                temperature_mean = FILL_VALUE_FLOAT
                        except:
                            pass


    return ab_532_par_mean, ab_532_per_mean, ab_1064_mean, sr_532_mean, nat_ice_R_threshold_mean, temperature_mean


def classify_features(per_detection_flags, asr_mean, ab_p_per_mean, asr_nat_ice, temp, alt, tropopause):

    # Initialization
    psc_mask = np.zeros(ab_p_per_mean.shape)

    # Thresholds
    # ab_p_per_liq_solid = 7.5e-6 # In V3, this threshold changes with horizontal averaging scale
    ab_p_per_nat_enat = 2e-5
    asr_nat_enat = 2
    asr_ice_waveice = 50
    tropo_press_lim = 215 # hPa # Criteria in PSC Mask V3 for "Likely tropospheric" flag
    temp_lim = 200 # K

    # Classification
    # psc_mask[ ab_p_per_mean <  ab_p_per_liq_solid] = 1 # STS
    psc_mask[~(per_detection_flags > 0) & (temp < temp_lim)] = 1 # STS where no enhancement in the perpendicular channel and T° < 200 K
    psc_mask[~(per_detection_flags > 0) & (temp >= temp_lim)] = 3 # SBS where no enhancement in the perpendicular channel and T° ≥ 200 K
    psc_mask[(per_detection_flags > 0) & (asr_mean == np.nan)] = 0 # Not determinable
    psc_mask[(per_detection_flags > 0) & (asr_mean < asr_nat_ice)] = 2 # NAT
    psc_mask[(per_detection_flags > 0) & (asr_mean >= asr_ice_waveice)] = 6 # Wave ice
    psc_mask[(per_detection_flags > 0) & (asr_mean >= asr_nat_ice) & (asr_mean < asr_ice_waveice)] = 4 # Ice
    psc_mask[(per_detection_flags > 0) & (ab_p_per_mean >= ab_p_per_nat_enat) & (asr_mean >= asr_nat_enat) & (asr_mean < asr_nat_ice)] = 5 # Enhanced NAT
    alt_2d = alt[np.newaxis, :]
    tropo_2d = tropopause[:, np.newaxis]
    psc_mask[alt_2d <= tropo_2d] = -4 # Likely tropospheric features
    
    psc_mask[asr_mean == FILL_VALUE_FLOAT] = 0 # No detection

    return psc_mask


def match_profiles(time_ref, time_target, tol=1):
    """
    Match each profile time in 'time_ref' with the closest profile time in 'time_target'.

    Parameters
    ----------
    time_ref : array-like
        Reference time array (e.g., CALIOP L1 Profile_Time at 5-km resolution).
    time_target : array-like
        Target time array to match against (e.g., PSCMask Profile_Time).
        Must be sorted in ascending order.
    tol : float
        Maximum allowed time difference (in same units as time arrays) to consider a match valid (e.g., seconds for Profile_Time).

    Returns
    -------
    idx : ndarray
        Indices in 'time_target' corresponding to the closest match for each element of 'time_ref'.
    valid : ndarray (bool)
        Boolean mask indicating whether the match is within tolerance.
    dt : ndarray
        Absolute time difference between matched profiles.
    """

    # Find insertion indices: where each time_ref would be inserted in time_target
    # to maintain sorted order (points to the "right neighbor")
    idx = np.searchsorted(time_target, time_ref)

    # Ensure indices stay within valid bounds [1, len(time_target)-1]
    # This is required because we will access idx-1 (left neighbor)
    idx = np.clip(idx, 1, len(time_target) - 1)

    # Get left and right neighboring times in time_target
    left = time_target[idx - 1]
    right = time_target[idx]

    # Compare distance to left and right neighbors
    # If left is closer, subtract 1 from idx to select it
    # (True = 1, False = 0 → vectorized operation)
    idx -= (np.abs(time_ref - left) < np.abs(time_ref - right))

    # Compute absolute time difference between matched profiles
    dt = np.abs(time_ref - time_target[idx])

    # Determine which matches are acceptable based on tolerance
    valid = dt < tol

    return idx, valid, dt


if __name__ == "__main__":
    tic_main_program = print_time()

    # Algorithm version (from git)
    VERSION_2D_McDA_PSC = get_full_version()

    # ------------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------------
    if len(sys.argv) != 2:
        raise ValueError("Usage: python main.py <config.yaml>")

    config_file = Path(sys.argv[1])
    if not config_file.exists():
        raise FileNotFoundError(config_file)

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # PARAMETERS
    # ------------------------------------------------------------------
    GRANULE_DATE = config["granule_date"]

    VERSION_CAL_LID_L1 = config["cal_lid_l1"]["version"]
    TYPE_CAL_LID_L1 = config["cal_lid_l1"]["type"]

    SLICE_MODE = config["slice"]["mode"]
    SLICE_START = config["slice"]["start"]
    SLICE_END = config["slice"]["end"]
    LAT_MIN = config["slice"]["lat_min"]
    LAT_MAX = config["slice"]["lat_max"]

    SAVE_DEVELOPMENT_DATA = config["processing"]["save_development_data"]
    PROCESS_UP_TO_40KM = config["processing"]["process_up_to_40km"]
    MAKE_CLASSIFICATION = config["processing"]["make_classification"]
    SEPARATION_TYPE = config["processing"]["feature_separation_type_for_classification"]

    TYPE_2D_McDA_PSC = config["algorithm"]["type"]

    OUT_FOLDER = config["output"]["folder"]
    OUT_FILETYPE = config["output"]["filetype"]

    if MAKE_CLASSIFICATION:
        FOLDER_CAL_LID_L2_PSCMask = "/DATA/LIENS/CALIOP/"
        VERSION_CAL_LID_L2_PSCMask = "V3.00"
        TYPE_CAL_LID_L2_PSCMask = "Standard" # "Standard", "Prov"


    # ********************************
    # *** Configuration parameters ***
    print("\n*****Configuration parameters...*****")
    
    print("\tGRANULE_DATE =", GRANULE_DATE)
    print("\tVERSION_CAL_LID_L1 =", VERSION_CAL_LID_L1)
    print("\tTYPE_CAL_LID_L1 =", TYPE_CAL_LID_L1)
    print("\tSLICE_START_END_TYPE =", SLICE_MODE)
    print("\tSLICE_START =", SLICE_START)
    print("\tSLICE_END =", SLICE_END)
    print("\tLAT_MIN =", LAT_MIN)
    print("\tLAT_MAX =", LAT_MAX)
    print("\tSAVE_DEVELOPMENT_DATA =", SAVE_DEVELOPMENT_DATA)
    print("\tPROCESS_UP_TO_40KM =", PROCESS_UP_TO_40KM)
    print("\tMAKE_CLASSIFICATION =", MAKE_CLASSIFICATION)
    print("\tSEPARATION_TYPE =", SEPARATION_TYPE)
    print("\tVERSION_2D_McDA_PSC =", VERSION_2D_McDA_PSC)
    print("\tTYPE_2D_McDA_PSC =", TYPE_2D_McDA_PSC)
    print("\tOUT_FOLDER =", OUT_FOLDER)
    if MAKE_CLASSIFICATION:
        print("\tFOLDER_CAL_LID_L2_PSCMask =", FOLDER_CAL_LID_L2_PSCMask)
        print("\tVERSION_CAL_LID_L2_PSCMask =", VERSION_CAL_LID_L2_PSCMask)
        print("\tTYPE_CAL_LID_L2_PSCMask =", TYPE_CAL_LID_L2_PSCMask)


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
                          slice_start_end_type=SLICE_MODE,
                          lat_min=LAT_MIN,
                          lat_max=LAT_MAX)

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


    # **************************************************************************
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


    # *******************************************************************************
    # *** Get data between 8.2 km and 30.1 (or 40.0) km at 5-km×180-m resolution  ***
    print("\n*****Get data between 8.2 km and 30.1 (or 40.0) km at 5-km×180-m resolution ...*****")

    tic = datetime.now()

    # ------------------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------------------
    data_dict_5kmx180m = {}
    data_dict_5km_met = {}
    START_INDEX_R5 = LAYER_ALTITUDE_R5_INDEX_RANGE[0] # at 40.0 km
    END_INDEX_R5 = LAYER_ALTITUDE_R5_INDEX_RANGE[1] # at 30.1km
    START_INDEX_R4 = LAYER_ALTITUDE_R4_INDEX_RANGE[0] # at 30.1 km
    END_INDEX_R4 = LAYER_ALTITUDE_R4_INDEX_RANGE[1] # at 20.2km; 55 bins already at 180-m resolution in R4
    START_INDEX_R3 = LAYER_ALTITUDE_R3_INDEX_RANGE[0] # at 20.2km
    END_INDEX_R3 = LAYER_ALTITUDE_R3_INDEX_RANGE[1] - 2 # at 8.2 km; 200 bins of 60-m in R3 => 200-2 = 198 => 198/3 = 66.0 bins of 180 m
    NB_VERT_BINS_TO_AVERAGE_IN_R3 = 3 # resolution 180m on a 60m grid
    NB_HORIZ_BINS_TO_AVERAGE = 15 # resolution 5km on a 333m grid
    if PROCESS_UP_TO_40KM:
        nb_vert_bins_180m_R5 = int((END_INDEX_R5 - START_INDEX_R5 + 1)*300/180) - 1
    else:
        nb_vert_bins_180m_R5 = 0
    nb_vert_bins_180m_R4 = END_INDEX_R4 - START_INDEX_R4 + 1
    nb_vert_bins_180m_R3 = int((END_INDEX_R3 - START_INDEX_R3 + 1)/3)
    
    # Get first profile ID of chunk
    prof_index_first_in_chunk = get_first_profileID_of_chunk(cal_l1.prof_min)
    
    # Get number of 5-km chunks
    nb_chunk_5km = int(data_dict_cal_lid_l1["Latitude"][prof_index_first_in_chunk:].size/NB_HORIZ_BINS_TO_AVERAGE)
    cal_lid_l1_prof_index_range_mult_of_15 = np.arange(prof_index_first_in_chunk, prof_index_first_in_chunk+nb_chunk_5km*NB_HORIZ_BINS_TO_AVERAGE)


    # ------------------------------------------------------------------------------
    # LOW ENERGY SHOTS
    # ------------------------------------------------------------------------------

    LOW_ENERGY_THRESHOLD_532  = 0.08
    LOW_ENERGY_THRESHOLD_1064 = 0.08
    MAX_LOW_ENERGY_PROFILES_IN_REGION_3 = 1 # 3 profiles (1 km) in region 3
    MAX_LOW_ENERGY_PROFILES_IN_REGION_4 = 2 # 5 profiles (5/3 km) in region 4
    MAX_LOW_ENERGY_PROFILES_IN_REGION_5 = 7 # 15 profiles (5 km) in region 5

    low_energy_profile_532  = data_dict_cal_lid_l1["Laser_Energy_532"]  < LOW_ENERGY_THRESHOLD_532
    low_energy_profile_1064 = data_dict_cal_lid_l1["Laser_Energy_1064"] < LOW_ENERGY_THRESHOLD_1064

    n_profiles_total = low_energy_profile_532.size

    bad_block_R3_532  = np.zeros(n_profiles_total, dtype=bool)
    bad_block_R4_532  = np.zeros(n_profiles_total, dtype=bool)
    bad_block_R5_532  = np.zeros(n_profiles_total, dtype=bool)

    bad_block_R3_1064 = np.zeros(n_profiles_total, dtype=bool)
    bad_block_R4_1064 = np.zeros(n_profiles_total, dtype=bool)
    bad_block_R5_1064 = np.zeros(n_profiles_total, dtype=bool)

    # Only consider profiles that belong to full 5-km chunks
    start = prof_index_first_in_chunk
    end   = prof_index_first_in_chunk + nb_chunk_5km * NB_HORIZ_BINS_TO_AVERAGE

    # ------------------------------------------------------------------
    # Region 3: blocks of 3 profiles (1 km)
    # ------------------------------------------------------------------
    for i in range(start, end, 3):

        block_532  = low_energy_profile_532[i:i+3]
        block_1064 = low_energy_profile_1064[i:i+3]

        if block_532.size == 3 and np.sum(block_532) > MAX_LOW_ENERGY_PROFILES_IN_REGION_3:
            bad_block_R3_532[i:i+3] = True

        if block_1064.size == 3 and np.sum(block_1064) > MAX_LOW_ENERGY_PROFILES_IN_REGION_3:
            bad_block_R3_1064[i:i+3] = True

    # ------------------------------------------------------------------
    # Region 4: blocks of 5 profiles (5/3 km)
    # ------------------------------------------------------------------
    for i in range(start, end, 5):

        block_532  = low_energy_profile_532[i:i+5]
        block_1064 = low_energy_profile_1064[i:i+5]

        if block_532.size == 5 and np.sum(block_532) > MAX_LOW_ENERGY_PROFILES_IN_REGION_4:
            bad_block_R4_532[i:i+5] = True

        if block_1064.size == 5 and np.sum(block_1064) > MAX_LOW_ENERGY_PROFILES_IN_REGION_4:
            bad_block_R4_1064[i:i+5] = True

    # ------------------------------------------------------------------
    # Region 5: blocks of 15 profiles (5 km)
    # ------------------------------------------------------------------
    for i in range(start, end, 15):

        block_532  = low_energy_profile_532[i:i+15]
        block_1064 = low_energy_profile_1064[i:i+15]

        if block_532.size == 15 and np.sum(block_532) > MAX_LOW_ENERGY_PROFILES_IN_REGION_5:
            bad_block_R5_532[i:i+15] = True

        if block_1064.size == 15 and np.sum(block_1064) > MAX_LOW_ENERGY_PROFILES_IN_REGION_5:
            bad_block_R5_1064[i:i+15] = True


    # ------------------------------------------------------------------
    # BUILD 5-km MASKS FROM BAD PROFILES
    # ------------------------------------------------------------------

    bad_chunk_R3_532  = np.zeros(nb_chunk_5km, dtype=bool)
    bad_chunk_R4_532  = np.zeros(nb_chunk_5km, dtype=bool)
    bad_chunk_R5_532  = np.zeros(nb_chunk_5km, dtype=bool)

    bad_chunk_R3_1064 = np.zeros(nb_chunk_5km, dtype=bool)
    bad_chunk_R4_1064 = np.zeros(nb_chunk_5km, dtype=bool)
    bad_chunk_R5_1064 = np.zeros(nb_chunk_5km, dtype=bool)

    for c in range(nb_chunk_5km):

        i0 = start + c * NB_HORIZ_BINS_TO_AVERAGE
        i1 = i0 + NB_HORIZ_BINS_TO_AVERAGE

        # --- R5 (15 profiles)
        if np.any(bad_block_R5_532[i0:i1]):
            bad_chunk_R5_532[c] = True

        if np.any(bad_block_R5_1064[i0:i1]):
            bad_chunk_R5_1064[c] = True

        # --- R4 (3 blocks of 5 profiles)
        for k in range(3):

            j0 = i0 + k*5
            j1 = j0 + 5

            if np.any(bad_block_R4_532[j0:j1]):
                bad_chunk_R4_532[c] = True

            if np.any(bad_block_R4_1064[j0:j1]):
                bad_chunk_R4_1064[c] = True

        # --- R3 (5 blocks of 3 profiles)
        for k in range(5):

            j0 = i0 + k*3
            j1 = j0 + 3

            if np.any(bad_block_R3_532[j0:j1]):
                bad_chunk_R3_532[c] = True

            if np.any(bad_block_R3_1064[j0:j1]):
                bad_chunk_R3_1064[c] = True


    # ------------------------------------------------------------------------------
    # 1-D vertical data at 180-m resolution
    # ------------------------------------------------------------------------------
    for key in ["Lidar_Data_Altitudes",]:
        # Initialization
        data_dict_5kmx180m[key] = np.ones(nb_vert_bins_180m_R5+nb_vert_bins_180m_R4+nb_vert_bins_180m_R3, dtype=np.float32)*FILL_VALUE_FLOAT
        if PROCESS_UP_TO_40KM:
            # Add 180 m nb_vert_bins_180m_R5 times from START_INDEX_R4
            data_dict_5kmx180m[key][:nb_vert_bins_180m_R5] = data_dict_cal_lid_l1[key][START_INDEX_R4] + 0.180*(nb_vert_bins_180m_R5 - np.arange(nb_vert_bins_180m_R5))
        # Copy R4 region (20.2 - 30.1 km)
        data_dict_5kmx180m[key][nb_vert_bins_180m_R5:nb_vert_bins_180m_R5+nb_vert_bins_180m_R4] = data_dict_cal_lid_l1[key][START_INDEX_R4:END_INDEX_R4+1]
        # Take middle altitude of 3 60-m vertical bins in R3 region (8.2 - 20.2 km)
        data_dict_5kmx180m[key][nb_vert_bins_180m_R5+nb_vert_bins_180m_R4:] = data_dict_cal_lid_l1[key][START_INDEX_R3+1:END_INDEX_R3:3]

    # ------------------------------------------------------------------------------
    # 2-D AB signal
    # ------------------------------------------------------------------------------

    key_list = [
        "Total_Attenuated_Backscatter_532",
        "Parallel_Attenuated_Backscatter_532",
        "Perpendicular_Attenuated_Backscatter_532",
        "Attenuated_Backscatter_1064",
    ]

    for key in key_list:

        data_dict_5kmx180m[key] = np.ma.ones(
            (nb_chunk_5km, nb_vert_bins_180m_R5+nb_vert_bins_180m_R4+nb_vert_bins_180m_R3)
        ) * FILL_VALUE_FLOAT

        # --- Load full-resolution data first ---
        data_full = np.ma.masked_invalid(data_dict_cal_lid_l1[key])

        # --- Apply LOW-ENERGY MASKS BEFORE 5-km averaging ---

        if "532"  in key:
            data_full[bad_block_R3_532, START_INDEX_R3:END_INDEX_R3+1] = np.ma.masked
            data_full[bad_block_R4_532, START_INDEX_R4:END_INDEX_R4+1] = np.ma.masked
            if PROCESS_UP_TO_40KM:
                data_full[bad_block_R5_532, START_INDEX_R5:END_INDEX_R5+1] = np.ma.masked

        if "1064" in key:
            data_full[bad_block_R3_1064, START_INDEX_R3:END_INDEX_R3+1] = np.ma.masked
            data_full[bad_block_R4_1064, START_INDEX_R4:END_INDEX_R4+1] = np.ma.masked
            if PROCESS_UP_TO_40KM:
                data_full[bad_block_R5_1064, START_INDEX_R5:END_INDEX_R5+1] = np.ma.masked

        # --- Perform 5-km horizontal averaging
        data = data_full[cal_lid_l1_prof_index_range_mult_of_15, :]
        data_15_prof_chunks = data.reshape(nb_chunk_5km,
                                        NB_HORIZ_BINS_TO_AVERAGE,
                                        -1)

        data_5km = np.ma.mean(data_15_prof_chunks, axis=1)

        # --- Vertical processing

        # --- R5 ---
        if PROCESS_UP_TO_40KM:

            alts_R5_300m = data_dict_cal_lid_l1["Lidar_Data_Altitudes"][
                START_INDEX_R5:END_INDEX_R5+2]

            alts_R5_180m = data_dict_5kmx180m["Lidar_Data_Altitudes"][
                :nb_vert_bins_180m_R5]

            data_R5_300m = data_5km[:, START_INDEX_R5:END_INDEX_R5+2]

            f_interp = interp1d(alts_R5_300m,
                                data_R5_300m,
                                kind='linear',
                                axis=1,
                                bounds_error=False,
                                fill_value='extrapolate')

            data_dict_5kmx180m[key][:, :nb_vert_bins_180m_R5] = \
                f_interp(alts_R5_180m)

        # --- R4 ---
        data_dict_5kmx180m[key][:,
            nb_vert_bins_180m_R5:
            nb_vert_bins_180m_R5+nb_vert_bins_180m_R4] = \
            data_5km[:, START_INDEX_R4:END_INDEX_R4+1]

        # --- R3 ---
        R3_data = data_5km[:, START_INDEX_R3:END_INDEX_R3+1]

        R3_3_vert_bins = R3_data.reshape(
            nb_chunk_5km,
            nb_vert_bins_180m_R3,
            NB_VERT_BINS_TO_AVERAGE_IN_R3
        )

        data_dict_5kmx180m[key][:,
            nb_vert_bins_180m_R5+nb_vert_bins_180m_R4:] = \
            np.ma.mean(R3_3_vert_bins, axis=2)

    # ------------------------------------------------------------------------------
    # 1-D horizontal data at 5-km resolution
    # ------------------------------------------------------------------------------
    for key in ["Latitude", "Longitude", "Profile_ID", "Profile_Time", "Profile_UTC_Time"]: #, "Number_Bins_Shift"]: 
        # Take middle (8th) profile of 5-km horizontal bins
        data_dict_5kmx180m[key] = data_dict_cal_lid_l1[key][cal_lid_l1_prof_index_range_mult_of_15][int(NB_HORIZ_BINS_TO_AVERAGE/2)::NB_HORIZ_BINS_TO_AVERAGE]
    
    # ------------------------------------------------------------------------------
    # 2-D data at 5-km×180-m resolution (other than AB signal)
    # ------------------------------------------------------------------------------
    key_list = ["Molecular_Total_Attenuated_Backscatter_532", "Molecular_Parallel_Attenuated_Backscatter_532", "Molecular_Perpendicular_Attenuated_Backscatter_532", "Molecular_Attenuated_Backscatter_1064",
                "Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Parallel", "Attenuated_Backscatter_Uncertainty_Standard_Deviation_532_Perpendicular", "Attenuated_Backscatter_Uncertainty_Standard_Deviation_1064",
                "Attenuated_Scattering_Ratio_532"]
    if SAVE_DEVELOPMENT_DATA:
        key_list = key_list + ["Background_Noise_532_Parallel", "Background_Noise_532_Perpendicular", "Background_Noise_1064", 
                               "Shot_Noise_532_Parallel", "Shot_Noise_532_Perpendicular", "Shot_Noise_1064",
                               "Noise_Scale_Factor_532_Parallel_AB_domain", "Noise_Scale_Factor_532_Perpendicular_AB_domain", "Noise_Scale_Factor_1064_AB_domain"]
    # Average data
    for key in key_list:
        # Initialization
        data_dict_5kmx180m[key] = np.ma.ones((nb_chunk_5km, nb_vert_bins_180m_R5+nb_vert_bins_180m_R4+nb_vert_bins_180m_R3))*FILL_VALUE_FLOAT

        # Horizontal averaging of the whole dataset over 15-profile chunks (5 km resolution)
        data = data_dict_cal_lid_l1[key][cal_lid_l1_prof_index_range_mult_of_15, :]
        data_15_prof_chunks = data.reshape(nb_chunk_5km, NB_HORIZ_BINS_TO_AVERAGE, -1)
        data_5km = np.ma.mean(data_15_prof_chunks, axis=1) 

        # Vertical averaging/interpolation per vertical region on data_5km
        if PROCESS_UP_TO_40KM:
            # R5 vertical interpolation from 300 m to 180 m
            alts_R5_300m = data_dict_cal_lid_l1["Lidar_Data_Altitudes"][START_INDEX_R5:END_INDEX_R5+2]
            alts_R5_180m = data_dict_5kmx180m["Lidar_Data_Altitudes"][:nb_vert_bins_180m_R5]
            data_R5_300m = data_5km[:, START_INDEX_R5:END_INDEX_R5+2]  
            f_interp = interp1d(alts_R5_300m, data_R5_300m, kind='linear', axis=1, bounds_error=False, fill_value='extrapolate')
            data_dict_5kmx180m[key][:, :nb_vert_bins_180m_R5] = f_interp(alts_R5_180m)
        # R4 region: already at 180-m resolution, just copy averaged data in altitude range
        data_dict_5kmx180m[key][:, nb_vert_bins_180m_R5:nb_vert_bins_180m_R5+nb_vert_bins_180m_R4] = data_5km[:, START_INDEX_R4:END_INDEX_R4+1]
        # R3 region: vertical averaging over groups of 3 bins (60m -> 180m)
        R3_data = data_5km[:, START_INDEX_R3:END_INDEX_R3+1] 
        R3_3_vert_bins = R3_data.reshape(nb_chunk_5km, nb_vert_bins_180m_R3, NB_VERT_BINS_TO_AVERAGE_IN_R3)
        data_dict_5kmx180m[key][:, nb_vert_bins_180m_R5+nb_vert_bins_180m_R4:] = np.ma.mean(R3_3_vert_bins, axis=2)

    # Print number of profiles in the granule
    print(f"\tNumber of 5-km profiles to process: {nb_chunk_5km}")

    print_elapsed_time(tic)
    

    # ***********************************
    # *** Compute particulate signals ***
    print("\n\n*****Compute particulate signals...*****")

    tic_algo = print_time()

    data_dict_5kmx180m["Particulate_Parallel_Attenuated_Backscatter_532"] = data_dict_5kmx180m["Parallel_Attenuated_Backscatter_532"] -\
                                                                            data_dict_5kmx180m["Molecular_Parallel_Attenuated_Backscatter_532"]
    data_dict_5kmx180m["Particulate_Perpendicular_Attenuated_Backscatter_532"] = data_dict_5kmx180m["Perpendicular_Attenuated_Backscatter_532"] -\
                                                                                 data_dict_5kmx180m["Molecular_Perpendicular_Attenuated_Backscatter_532"]
    data_dict_5kmx180m["Particulate_Attenuated_Backscatter_1064"] = data_dict_5kmx180m["Attenuated_Backscatter_1064"] -\
                                                                    data_dict_5kmx180m["Molecular_Attenuated_Backscatter_1064"]

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


    # ************************************************
    # *** Mask low energy regions in channel masks ***
    print("\n\n*****Mask low energy regions in channel masks...*****")

    key_list = [
        "Parallel_Detection_Flags_532",
        "Perpendicular_Detection_Flags_532",
        "Detection_Flags_1064"
    ]

    for key in key_list:

        channel_mask = data_dict_2d_mcda[key].copy()

        if "532" in key:
            channel_mask[bad_chunk_R3_532,
                nb_vert_bins_180m_R5+nb_vert_bins_180m_R4:] = np.ma.masked

            channel_mask[bad_chunk_R4_532,
                nb_vert_bins_180m_R5:
                nb_vert_bins_180m_R5+nb_vert_bins_180m_R4] = np.ma.masked

            if PROCESS_UP_TO_40KM:
                channel_mask[bad_chunk_R5_532,
                    :nb_vert_bins_180m_R5] = np.ma.masked
    
        if "1064" in key:
            channel_mask[bad_chunk_R3_1064,
                nb_vert_bins_180m_R5+nb_vert_bins_180m_R4:] = np.ma.masked

            channel_mask[bad_chunk_R4_1064,
                nb_vert_bins_180m_R5:
                nb_vert_bins_180m_R5+nb_vert_bins_180m_R4] = np.ma.masked

            if PROCESS_UP_TO_40KM:
                channel_mask[bad_chunk_R5_1064,
                    :nb_vert_bins_180m_R5] = np.ma.masked

        data_dict_2d_mcda[key] = channel_mask


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

    

    if MAKE_CLASSIFICATION:

        # *************************
        # *** Load PSCMask data ***
        # To get PSC_Ice_Mixture_Boundary, the threshold between NAT and ice computed from MLS observations
        print("\n*****Load PSCMask data...*****")

        tic = datetime.now()

        # Get filename and filepath
        granule_date_dict = split_granule_date(GRANULE_DATE)
        filename_psc = f"CAL_LID_L2_PSCMask-{TYPE_CAL_LID_L2_PSCMask}-{VERSION_CAL_LID_L2_PSCMask.replace('.', '-')}." \
                    f"{granule_date_dict['year']}-{granule_date_dict['month']:02d}-{granule_date_dict['day']:02d}T00-00-00ZN.hdf"
        hdffile = os.path.join(FOLDER_CAL_LID_L2_PSCMask, f"PSCMask.{VERSION_CAL_LID_L2_PSCMask.replace('V', 'v')}",
                            str(granule_date_dict['year']),
                            f"{granule_date_dict['year']}_{granule_date_dict['month']:02d}_"
                            f"{granule_date_dict['day']:02d}",
                            filename_psc)

        # Open HDF file
        print(f"\tGranule path: {hdffile}")
        cal_psc = SD(hdffile, SDC.READ)

        # Find granule section in the daily PSC file
        l1_input_filenames = cal_psc.select("L1_Input_Filenames")[:]
        granule_names = []
        for i_filenames in np.arange(l1_input_filenames.shape[0]):
            granule_name = ''
            if VERSION_CAL_LID_L2_PSCMask in ("V2.00", "V3.00"):
                granule_name_char_indexes = np.arange(26, 47)
            elif VERSION_CAL_LID_L2_PSCMask == "V1.00":
                granule_name_char_indexes = np.arange(27, 48)
            else:
                raise ValueError(f"Define 'granule_name_char_indexes' for VERSION_CAL_LID_L2_PSCMask = {VERSION_CAL_LID_L2_PSCMask}")
            for i_char in granule_name_char_indexes:
                granule_name = granule_name + l1_input_filenames[i_filenames][i_char].decode('UTF-8')
            granule_names.append(granule_name)
        granule_name_index = granule_names.index(GRANULE_DATE)
        granule_start_time = cal_psc.select("L1_Input_Start_Times")[granule_name_index]
        granule_end_time = cal_psc.select("L1_Input_End_Times")[granule_name_index]
        profile_utc_time = cal_psc.select("Profile_UTC_Time")[:]
        granule_start_index = int((np.abs(profile_utc_time - granule_start_time)).argmin())
        granule_end_index = int((np.abs(profile_utc_time - granule_end_time)).argmin())

        # Load PSCMask V3 variables and match 
        psc_v3_ice_nat_threshold = cal_psc.select("PSC_Ice_Mixture_Boundary")[granule_start_index:granule_end_index + 1, :]
        psc_v3_pressure = cal_psc.select("Pressure")[granule_start_index:granule_end_index + 1, :]
        psc_v3_temperature = cal_psc.select("Temperature")[granule_start_index:granule_end_index + 1, :]
        psc_v3_profile_time = cal_psc.select("Profile_Time")[granule_start_index:granule_end_index + 1]
        psc_v3_tropopause = cal_psc.select("Tropopause_Altitude_MERRA2")[granule_start_index:granule_end_index + 1]
        psc_v3_altitude = cal_psc.select("Altitude")[:]
        data_dict_2d_mcda['Pressure_HNO3'] = cal_psc.select("Pressure_HNO3")[:]
        data_dict_2d_mcda['Pressure_H2O'] = cal_psc.select("Pressure_H2O")[:]
        psc_v3_hno3_mix_ratio = cal_psc.select("HNO3_Mixing_Ratio")[granule_start_index:granule_end_index + 1, :]
        psc_v3_h2o_mix_ratio = cal_psc.select("H2O_Mixing_Ratio")[granule_start_index:granule_end_index + 1, :]

        if not np.allclose(psc_v3_altitude, data_dict_5kmx180m["Lidar_Data_Altitudes"]):
            raise ValueError("Altitude grids do not match")
                             
        # Match L1 profile times with PSCMask profile times
        indices, valid, dt = match_profiles(
            data_dict_5kmx180m["Profile_Time"], 
            psc_v3_profile_time
        )

        # Print diagnostic information about matching quality
        print(f"\tMax time difference: {dt.max():.2e} s")
        print(f"\tValid matches: {valid.sum()} / {len(valid)}")

        # Initialize output array with NaNs (missing values)
        n_prof = len(data_dict_5kmx180m["Profile_Time"])
        n_alt = len(data_dict_5kmx180m["Lidar_Data_Altitudes"])
        n_press_hno3 = len(data_dict_2d_mcda['Pressure_HNO3'])
        n_press_h2o = len(data_dict_2d_mcda['Pressure_H2O'])
        psc_v3_ice_nat_threshold_matched = np.full((n_prof, n_alt), np.nan)
        psc_v3_pressure_matched = np.full((n_prof, n_alt), np.nan)
        psc_v3_temperature_matched = np.full((n_prof, n_alt), np.nan)
        psc_v3_tropopause_matched = np.full((n_prof,), np.nan)
        psc_v3_hno3_mix_ratio_matched = np.full((n_prof, n_press_hno3), np.nan)
        psc_v3_h2o_mix_ratio_matched = np.full((n_prof, n_press_h2o), np.nan)

        # Fill only valid matches:
        # For each valid L1 profile, copy the corresponding PSCMask profile
        psc_v3_ice_nat_threshold_matched[valid] = psc_v3_ice_nat_threshold[indices[valid], :]
        data_dict_5kmx180m["nat_ice_R_threshold"] = psc_v3_ice_nat_threshold_matched
        psc_v3_pressure_matched[valid] = psc_v3_pressure[indices[valid], :]
        data_dict_5kmx180m["press"] = psc_v3_pressure_matched
        psc_v3_temperature_matched[valid] = psc_v3_temperature[indices[valid], :]
        data_dict_5kmx180m["temp"] = psc_v3_temperature_matched
        psc_v3_tropopause_matched[valid] = psc_v3_tropopause[indices[valid]]
        data_dict_5kmx180m["tropopause"] = psc_v3_tropopause_matched
        psc_v3_hno3_mix_ratio_matched[valid] = psc_v3_hno3_mix_ratio[indices[valid], :]
        data_dict_5kmx180m["HNO3_Mixing_Ratio"] = psc_v3_hno3_mix_ratio_matched
        psc_v3_h2o_mix_ratio_matched[valid] = psc_v3_h2o_mix_ratio[indices[valid], :]
        data_dict_5kmx180m["H2O_Mixing_Ratio"] = psc_v3_h2o_mix_ratio_matched

        print_elapsed_time(tic)


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


        # ****************************************************
        # *** Apply classification to homogeneous features ***
        print("\n\n############################################################\n"\
            "*****Apply classification to homogeneous features...*****")

        tic_algo = print_time()

        data_dict_2d_mcda["homogeneous_chunks_mean_ab_532_par"], \
        data_dict_2d_mcda["homogeneous_chunks_mean_ab_532_per"], \
        data_dict_2d_mcda["homogeneous_chunks_mean_ab_1064"], \
        data_dict_2d_mcda["homogeneous_chunks_mean_asr_532"], \
        data_dict_2d_mcda["homogeneous_chunks_mean_nat_ice_R_threshold"], \
        data_dict_2d_mcda["homogeneous_chunks_mean_temperature"] = \
            average_over_homogeneous_chunks(data_dict_2d_mcda["homogeneous_chunks_mask"], 
                                            data_dict_5kmx180m["Parallel_Attenuated_Backscatter_532"], 
                                            data_dict_5kmx180m["Perpendicular_Attenuated_Backscatter_532"], 
                                            data_dict_5kmx180m["Attenuated_Backscatter_1064"], 
                                            data_dict_5kmx180m["Attenuated_Scattering_Ratio_532"],
                                            data_dict_5kmx180m["nat_ice_R_threshold"],
                                            data_dict_5kmx180m["temp"],
                                            separation_type=SEPARATION_TYPE)
        
        data_dict_2d_mcda["homogeneous_chunks_mean_part_ab_532_par"], \
        data_dict_2d_mcda["homogeneous_chunks_mean_part_ab_532_per"], \
        data_dict_2d_mcda["homogeneous_chunks_mean_part_ab_1064"], \
        _, _, _ = \
            average_over_homogeneous_chunks(data_dict_2d_mcda["homogeneous_chunks_mask"], 
                                            data_dict_5kmx180m["Particulate_Parallel_Attenuated_Backscatter_532"], 
                                            data_dict_5kmx180m["Particulate_Perpendicular_Attenuated_Backscatter_532"], 
                                            data_dict_5kmx180m["Particulate_Attenuated_Backscatter_1064"], 
                                            data_dict_5kmx180m["Attenuated_Scattering_Ratio_532"],
                                            data_dict_5kmx180m["nat_ice_R_threshold"],
                                            data_dict_5kmx180m["temp"],
                                            separation_type=SEPARATION_TYPE)
        
        data_dict_2d_mcda["homogeneous_chunks_classification"] = \
            classify_features(data_dict_2d_mcda["Perpendicular_Detection_Flags_532"],
                              data_dict_2d_mcda["homogeneous_chunks_mean_asr_532"],
                              data_dict_2d_mcda["homogeneous_chunks_mean_part_ab_532_per"],
                              data_dict_2d_mcda["homogeneous_chunks_mean_nat_ice_R_threshold"],
                              data_dict_2d_mcda["homogeneous_chunks_mean_temperature"],
                              data_dict_5kmx180m["Lidar_Data_Altitudes"],
                              data_dict_5kmx180m["tropopause"])
        
        print_elapsed_time(tic_algo)


    # *****************
    # *** Save data ***
    print("\n\n############################################################\n"\
          "*****Save data...*****")
    
    # Create folder to store output data
    granule_date_dict = split_granule_date(GRANULE_DATE)
    outdata_folder = os.path.join(OUT_FOLDER, f"2D_McDA_PSC.{VERSION_2D_McDA_PSC}",
                                  str(granule_date_dict['year']), f"{granule_date_dict['year']}_"
                                                                  f"{granule_date_dict['month']:02d}_"
                                                                  f"{granule_date_dict['day']:02d}")
    os.makedirs(outdata_folder, exist_ok=True)
    
    # Replace masked data by FILL_VALUE
    key_list = [
        "Parallel_Attenuated_Backscatter_532",
        "Perpendicular_Attenuated_Backscatter_532",
        "Attenuated_Backscatter_1064"
    ]
    for key in key_list:
        data_dict_5kmx180m[key] = data_dict_5kmx180m[key].filled(FILL_VALUE_FLOAT)
    key_list = [
        "Parallel_Detection_Flags_532",
        "Perpendicular_Detection_Flags_532",
        "Detection_Flags_1064"
    ]
    for key in key_list:
        data_dict_2d_mcda[key] = data_dict_2d_mcda[key].filled(FILL_VALUE_BYTE)


    # Save the data
    save_data(data_dict_5kmx180m, data_dict_2d_mcda, data_dict_2d_mcda_dev, filetype=OUT_FILETYPE, save_development_data=SAVE_DEVELOPMENT_DATA)

    
    print_time(tic_main_program)
    