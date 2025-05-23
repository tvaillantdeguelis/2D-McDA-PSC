#!/bin/bash

#SBATCH --job-name=scheduler
#SBATCH --time=3-00:00:00
#SBATCH -o "./out/slurm/2D_McDA_PSC_run.out"

# Run this scipt with: 'sbatch 2D_McDA_PSC_run.sh'

# This program launches 2D-McDA on every granules found between 'START_DATE' and 'END_DATE'.

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# CONFIGURATION
VERSION_CAL_LID_L1="V4.51"
TYPE_CAL_LID_L1="Standard"
SLICE_START_END_TYPE="latminmax"
SLICE_START="None"
SLICE_END="None"
START_DATE="2010-01-01"
END_DATE="2010-01-31" # included
LAT_MIN=50
LAT_MAX="None"
SAVE_DEVELOPMENT_DATA="False" # if "True" save step by step data
VERSION_2D_McDA_PSC="V1.2.1"
TYPE_2D_McDA_PSC="Prototype"
OUT_FILETYPE='HDF' # 'HDF' or 'netCDF'
#-----------------------------------------------------------------------
DAYNIGHT_FLAG="ZN" # "ZN", "ZD", or ""
DATA_FOLDER_L1_HEAD="/DATA/LIENS/CALIOP/CAL_LID_L1."
OUT_FOLDER="/work_users/vaillant/data/2D_McDA_PSC/"
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

# Initialization
mkdir -p out/slurm # Ensure the directory exists and avoids job submission failure
shopt -s nullglob # Prevent glob pattern from expanding to itself
currentdate=$START_DATE
data_file_l1_head="CAL_LID_L1-${TYPE_CAL_LID_L1}-${VERSION_CAL_LID_L1//[.]/-}"
EXTENSION=".hdf"
GRANULE_DATE_SIZE=21
MAX_NB_JOBS=101

# Loop on each day between 'START_DATE' and 'END_DATE'
while [ "$(date -d "$currentdate" +%s)" -le "$(date -d "$END_DATE" +%s)" ]; do

    # Get current year, month, and day
    year="${currentdate:0:4}"
    month="${currentdate:5:2}"
    day="${currentdate:8:2}"

    # Get the corresponding L1 data folder
    data_folder_l1="${DATA_FOLDER_L1_HEAD}${VERSION_CAL_LID_L1//[V]/v}/${year}/${year}_${month}_${day}/"
    echo "Processing folder: $data_folder_l1"

    # If this folder does not exist
    if [ ! -d "$data_folder_l1" ]; then
        # Show message that the folder does not exist
        echo -e "\n$data_folder_l1 does not exist"

    # If this folder exists
    else
        # Loop on granules in this folder
        file_l1_list=( "${data_folder_l1}${data_file_l1_head}"*"${DAYNIGHT_FLAG}${EXTENSION}" )
        for file_l1 in "${file_l1_list[@]}"; do

            # Get granule time
            granule_date=${file_l1:(-GRANULE_DATE_SIZE - ${#EXTENSION}):${GRANULE_DATE_SIZE}}

            # Wait if max number of jobs reached
            jobs_count=$(squeue -u vaillant | tail -n +2 | wc -l)
            while (( jobs_count >= MAX_NB_JOBS )); do
                sleep 3
                jobs_count=$(squeue -u vaillant | tail -n +2 | wc -l)
            done

            # Run 2D-McDA
            jobname="2D-McDA-PSC_${granule_date}"
            echo -n "Launching job $jobname:"
            sbatch --job-name="$jobname" \
                   --error=./out/slurm/"${jobname}.e" \
                   --output=./out/slurm/"${jobname}.o" \
                   --export="GRANULE_DATE=$granule_date,\
VERSION_CAL_LID_L1=$VERSION_CAL_LID_L1,\
TYPE_CAL_LID_L1=$TYPE_CAL_LID_L1,\
PREVIOUS_GRANULE=$previous_granule_date,\
NEXT_GRANULE=$next_granule_date,\
SLICE_START_END_TYPE=$SLICE_START_END_TYPE,\
SLICE_START=$SLICE_START,\
SLICE_END=$SLICE_END,\
LAT_MIN=$LAT_MIN,\
LAT_MAX=$LAT_MAX,\
SAVE_DEVELOPMENT_DATA=$SAVE_DEVELOPMENT_DATA,\
VERSION_2D_McDA_PSC=$VERSION_2D_McDA_PSC,\
TYPE_2D_McDA_PSC=$TYPE_2D_McDA_PSC,\
OUT_FOLDER=$OUT_FOLDER,\
OUT_FILETYPE=$OUT_FILETYPE" \
       2D_McDA_PSC.sbatch
        done
    fi

    currentdate=$(date -I -d "$currentdate + 1 day")
done
