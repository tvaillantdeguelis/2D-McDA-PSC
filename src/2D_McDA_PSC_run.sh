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
START_DATE="2006-06-12"
END_DATE="2023-06-30" # included
SAVE_DEVELOPMENT_DATA="False" # if "True" save step by step data
TYPE_2D_McDA_PSC="Prototype"
OUT_FILETYPE='netCDF' # 'HDF' or 'netCDF'
PROCESS_UP_TO_40KM="True"
#-----------------------------------------------------------------------
DAYNIGHT_FLAG="ZN" # "ZN", "ZD", or ""
DATA_FOLDER_L1_HEAD="/DATA/LIENS/CALIOP/CAL_LID_L1."
OUT_FOLDER="/work_users/vaillant/data/2D_McDA_PSC/"
MAX_NB_JOBS=101
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

# Initialization
mkdir -p out/slurm # Ensure the directory exists and avoids job submission failure
shopt -s nullglob # Prevent glob pattern from expanding to itself
currentdate=$START_DATE
data_file_l1_head="CAL_LID_L1-${TYPE_CAL_LID_L1}-${VERSION_CAL_LID_L1//[.]/-}"
EXTENSION_CAL_LID_L1=".hdf"
GRANULE_DATE_SIZE=21

# Loop on each day between 'START_DATE' and 'END_DATE'
while [ "$(date -d "$currentdate" +%s)" -le "$(date -d "$END_DATE" +%s)" ]; do

    # Get current year, month, and day
    year="${currentdate:0:4}"
    month="${currentdate:5:2}"
    day="${currentdate:8:2}"

    # Decide which hemisphere to process
    LAT_MIN=""
    LAT_MAX=""

    case "$month" in
        05|06|07|08|09|10)
            LAT_MIN="None"
            LAT_MAX=-50
            ;;
        12|01|02|03)
            LAT_MIN=50
            LAT_MAX="None"
            ;;
        *)
            currentdate=$(date -I -d "$currentdate + 1 day")
            continue
            ;;
    esac

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
        file_l1_list=( "${data_folder_l1}${data_file_l1_head}"*"${DAYNIGHT_FLAG}${EXTENSION_CAL_LID_L1}" )
        for file_l1 in "${file_l1_list[@]}"; do

            # Get granule time
            granule_date=${file_l1:(-GRANULE_DATE_SIZE - ${#EXTENSION_CAL_LID_L1}):${GRANULE_DATE_SIZE}}

            # Wait if max number of jobs reached
            jobs_count=$(squeue -u vaillant | tail -n +2 | wc -l)
            while (( jobs_count >= MAX_NB_JOBS )); do
                sleep 3
                jobs_count=$(squeue -u vaillant | tail -n +2 | wc -l)
            done

            # Create YAML params file
            params_file="src/2D_McDA_PSC_params_tmp/2D_McDA_PSC_params_${granule_date}.yaml"

            sed \
            -e "s|__GRANULE_DATE__|$granule_date|g" \
            -e "s|__VERSION_CAL_LID_L1__|$VERSION_CAL_LID_L1|g" \
            -e "s|__TYPE_CAL_LID_L1__|$TYPE_CAL_LID_L1|g" \
            -e "s|__SLICE_START_END_TYPE__|$SLICE_START_END_TYPE|g" \
            -e "s|__SLICE_START__|$SLICE_START|g" \
            -e "s|__SLICE_END__|$SLICE_END|g" \
            -e "s|__LAT_MIN__|$LAT_MIN|g" \
            -e "s|__LAT_MAX__|$LAT_MAX|g" \
            -e "s|__SAVE_DEVELOPMENT_DATA__|$SAVE_DEVELOPMENT_DATA|g" \
            -e "s|__PROCESS_UP_TO_40KM__|$PROCESS_UP_TO_40KM|g" \
            -e "s|__TYPE_2D_McDA_PSC__|$TYPE_2D_McDA_PSC|g" \
            -e "s|__OUT_FOLDER__|$OUT_FOLDER|g" \
            -e "s|__OUT_FILETYPE__|$OUT_FILETYPE|g" \
            configs/2D_McDA_PSC_params_template.yaml > "$params_file"

            # Run 2D-McDA
            jobname="2D-McDA-PSC_${granule_date}"
            job_year="${granule_date:0:4}"
            log_dir="./out/slurm/${job_year}"
            mkdir -p "$log_dir"
            echo -n "[$(date)] Launching job ${jobname}: "
            sbatch --job-name="$jobname" \
                   --error="${log_dir}/${jobname}.e" \
                   --output="${log_dir}/${jobname}.o" \
                   --export=PARAMS_FILE="$params_file" \
       2D_McDA_PSC.sbatch
        done
    fi

    currentdate=$(date -I -d "$currentdate + 1 day")
done
