#!/bin/bash

# This program launches 2D-McDA on a given list of granules (between SLICE_START and SLICE_END)

# Run plot_calipso_section.py
run_2d_mcda () {
jobname="2D-McDA_$1"
echo -e "\njobname=$jobname"
sbatch --job-name=$jobname \
    --error=./sbatch_out/${jobname}.e \
    --output=./sbatch_out/${jobname}.o \
    --export=GRANULE_DATE=$1,\
VERSION_CAL_LID_L1=$2,\
TYPE_CAL_LID_L1=$3,\
PREVIOUS_GRANULE=$4,\
NEXT_GRANULE=$5,\
SLICE_START_END_TYPE=$6,\
SLICE_START=$7,SLICE_END=$8,\
SAVE_DEVELOPMENT_DATA=$9,\
VERSION_2D_McDA=${10},\
TYPE_2D_McDA=${11},\
OUT_FOLDER=${12} 2D_McDA.sbatch
}

granule_date="2018-07-08T04-07-14ZN"
version_cal_lid_l1="V4.51"
type_cal_lid_l1="Standard"
previous_granule="None"
next_granule="None"
slice_start_end_type="longitude" # "profindex" or "longitude"
slice_start=-50.00 # profindex or longitude
slice_end=-140.00 # profindex or longitude
save_development_data="False" # if "True" save step by step data
version_2d_mcda="V1.4.1"
type_2d_mcda="Prototype"
out_folder="/home/vaillant/workspace/data/2D_McDA_for_PSCs/"
run_2d_mcda $granule_date $version_cal_lid_l1 $type_cal_lid_l1 $previous_granule $next_granule\
            $slice_start_end_type $slice_start $slice_end $save_development_data $version_2d_mcda $type_2d_mcda $out_folder