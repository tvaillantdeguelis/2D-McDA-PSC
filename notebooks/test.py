import numpy as np
import os
import sys

from readers.calipso_reader import CALIPSOReader, get_prof_min_max_indexes_from_lon
from paths import split_granule_date

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# PARAMETERS
INDATA_FOLDER = "/DATA/LIENS/CALIOP/"
GRANULE_DATE = "2010-01-18T00-19-57ZN"
VERSION_CAL_LID_L2_PSCMask = "V2.00"
TYPE_CAL_LID_L2_PSCMask = "Standard" # "Standard", "Prov"
SLICE_START_END_TYPE = 'longitude' # 'profindex' (of the PSCMask file) or 'longitude'
SLICE_START = 170.59 # profindex or longitude
SLICE_END = 27.95 # profindex or longitude
EDGES_REMOVAL = 0 # number of prof to remove on both edges of plot
INVERT_XAXIS = False
YMIN = 15
YMAX = 30
FIGURES_PATH = "/home/vaillant/codes/projects/plot_CALIPSO_section/out/figures/"
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


# *******************************
# *** Load PSC mask data file ***
print("\n*****Load PSC mask data file...*****")

# Get filename and filepath
granule_date_dict = split_granule_date(GRANULE_DATE)
filename_psc = f"CAL_LID_L2_PSCMask-{TYPE_CAL_LID_L2_PSCMask}-{VERSION_CAL_LID_L2_PSCMask.replace('.', '-')}." \
                f"{granule_date_dict['year']}-{granule_date_dict['month']:02d}-{granule_date_dict['day']:02d}T00-00-00ZN.hdf"
hdffile = os.path.join(INDATA_FOLDER, f"PSCMask.{VERSION_CAL_LID_L2_PSCMask.replace('V', 'v')}",
                        str(granule_date_dict['year']),
                        f"{granule_date_dict['year']}_{granule_date_dict['month']:02d}_"
                        f"{granule_date_dict['day']:02d}",
                        filename_psc)

# Open HDF file
print(f"\tGranule path: {hdffile}")
cal_psc = CALIPSOReader(hdffile)

# Find granule section in the daily PSC file
l1_input_filenames = cal_psc.get_data("L1_Input_Filenames")
granule_names = []
for i_filenames in np.arange(l1_input_filenames.shape[0]):
    granule_name = ''
    if VERSION_CAL_LID_L2_PSCMask == "V2.00":
        granule_name_char_indexes = np.arange(26, 47)
    elif VERSION_CAL_LID_L2_PSCMask == "V1.00":
        granule_name_char_indexes = np.arange(27, 48)
    else:
        sys.exit(f"Define 'granule_name_char_indexes' for VERSION_CAL_LID_L2_PSCMask = {VERSION_CAL_LID_L2_PSCMask}")
    for i_char in granule_name_char_indexes:
        granule_name = granule_name + l1_input_filenames[i_filenames][i_char].decode('UTF-8')
    granule_names.append(granule_name)
granule_name_index = granule_names.index(GRANULE_DATE)
granule_start_time = cal_psc.get_data("L1_Input_Start_Times")[granule_name_index]
granule_end_time = cal_psc.get_data("L1_Input_End_Times")[granule_name_index]
profile_utc_time = cal_psc.get_data("Profile_UTC_Time")
granule_start_index = (np.abs(profile_utc_time - granule_start_time)).argmin()
granule_end_index = (np.abs(profile_utc_time - granule_end_time)).argmin()
print('granule_start_index:', granule_start_index)
print('granule_end_index:', granule_end_index)


# Get prof_min and prof_max from longitudes
lat_granule = cal_psc.get_data("Latitude")[granule_start_index:granule_end_index+1]
lon_granule = cal_psc.get_data("Longitude")[granule_start_index:granule_end_index+1]
if SLICE_START_END_TYPE == 'longitude':
    prof_min_granule, prof_max_granule = get_prof_min_max_indexes_from_lon(lon_granule, SLICE_START, SLICE_END) 
    prof_min = prof_min_granule + granule_start_index
    prof_max = prof_max_granule + granule_start_index
else:
    prof_min = SLICE_START
    prof_max = SLICE_END
    
# Print lat/lon of min and max prof indices
print(f"\tFrom min profile index {prof_min:d} "
        f"(lat = {lat_granule[prof_min_granule]:.2f} / lon = {lon_granule[prof_min_granule]:.2f}) "
        f"to max profile index {prof_max:d} "
        f"(lat = {lat_granule[prof_max_granule]:.2f} / lon = {lon_granule[prof_max_granule]:.2f})")

# Load 2D-McDA parameters
data_dict_cal_psc = {}
cal_psc_keys = [
    "Latitude",
    "Longitude",
    "Profile_UTC_Time",
    "Altitude",
    "PSC_Feature_Mask",
    "PSC_Composition"
]
for key in cal_psc_keys:
    data_dict_cal_psc[key] = cal_psc.get_data(key, prof_min, prof_max, 'profindex')

lat = data_dict_cal_psc["Latitude"] 
lon = data_dict_cal_psc["Longitude"]
alt = data_dict_cal_psc["Altitude"]