#!/bin/bash

# This program launches 2D-McDA on every granules found between 'START_DATE' and 'END_DATE'.

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# CONFIGURATION
VERSION_CAL_LID_L1="V4.10"
TYPE_CAL_LID_L1="Standard"
START_DATE=2010-06-01
END_DATE=2010-06-01 # included
SAVE_DEVELOPMENT_DATA="False" # if "True" save step by step data
VERSION_2D_McDA="V1.01"
TYPE_2D_McDA="Prototype"
#-----------------------------------------------------------------------
# ICARE path
DATA_FOLDER_L1_HEAD="/DATA/LIENS/CALIOP/CAL_LID_L1."
OUT_FOLDER="/work_users/vaillant/data/2D_CALIOP/2D_McDA/"
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

# Initialization
currentdate="$START_DATE"
previous_file_l1="None"
current_file_l1="None"
next_file_l1="None"
SLICE_START="None"
SLICE_END="None"
SLICE_START_END_TYPE="profindex"
data_file_l1_head="CAL_LID_L1-${TYPE_CAL_LID_L1}-${VERSION_CAL_LID_L1//[.]/-}"
extension=".hdf"
granule_date_size=21

# Loop on each day between 'START_DATE' and 'END_DATE'
while [ $(date -I -d "$currentdate - 1 day") != "$END_DATE" ]; do

    # Get current year, month, and day
    year="${currentdate:0:4}"
    month="${currentdate:5:2}"
    day="${currentdate:8:2}"

    # Get the corresponding L1 data folder
    data_folder_l1="${DATA_FOLDER_L1_HEAD}${VERSION_CAL_LID_L1//[V]/v}/${year}/${year}_${month}_${day}/"

    # If this folder does not exist
    if [ ! -d $data_folder_l1 ]; then
        # Show message that the folder does not exist
        echo -e "\n$data_folder_l1 does not exist"

    # If this folder exists
    else
        # Loop on granules in this folder
        for file_L1 in ${data_folder_l1}${data_file_l1_head}*${extension}; do

            previous_file_l1=$current_file_l1
            current_file_l1=$next_file_l1
            next_file_l1=$file_L1

            if [[ $previous_file_l1 != "None" ]]; then
                # Get granule time
                pos_start_time=$((${#previous_file_l1}-${#extension}-${granule_date_size}))
                                # ${#x} gives nb char in x
                previous_granule_date=${previous_file_l1:$pos_start_time:${granule_date_size}}
            else
                previous_granule_date="None"
            fi

            if [[ $next_file_l1 != "None" ]]; then
                # Get granule time
                pos_start_time=$((${#next_file_l1}-${#extension}-${granule_date_size}))
                                # ${#x} gives nb char in x
                next_granule_date=${next_file_l1:$pos_start_time:${granule_date_size}}
            else
                next_granule_date="None"
            fi

            if [[ $current_file_l1 != "None" ]]; then
                # Get granule time
                pos_start_time=$((${#current_file_l1}-${#extension}-${granule_date_size}))
                                # ${#x} gives nb char in x
                granule_date=${current_file_l1:$pos_start_time:${granule_date_size}}

                echo $granule_date $previous_granule_date $next_granule_date
                # Run 2D-McDA
                jobname="2D-McDA_${granule_date}"
                echo -e "\njobname=$jobname"
                sbatch --job-name=$jobname \
                       --error=./sbatch_out/${jobname}.e \
                       --output=./sbatch_out/${jobname}.o \
                       --export=GRANULE_DATE=$granule_date,VERSION_CAL_LID_L1=$VERSION_CAL_LID_L1,\
TYPE_CAL_LID_L1=$TYPE_CAL_LID_L1,PREVIOUS_GRANULE=$previous_granule_date,NEXT_GRANULE=$next_granule_date,\
SLICE_START_END_TYPE=$SLICE_START_END_TYPE,SLICE_START=$SLICE_START,SLICE_END=$SLICE_END,\
SAVE_DEVELOPMENT_DATA=$SAVE_DEVELOPMENT_DATA,VERSION_2D_McDA=$VERSION_2D_McDA,\
TYPE_2D_McDA=$TYPE_2D_McDA,OUT_FOLDER=$OUT_FOLDER 2D_McDA.sbatch
            fi
        done
    fi

    currentdate=$(date -I -d "$currentdate + 1 day")
done

# Process last granule
previous_file_l1=$current_file_l1
current_file_l1=$next_file_l1
previous_granule_date=${previous_file_l1:$pos_start_time:${granule_date_size}}
next_granule_date="None"
granule_date=${current_file_l1:$pos_start_time:${granule_date_size}}

echo $granule_date $previous_granule_date $next_granule_date

# Run 2D-McDA
jobname="2D-McDA_${granule_date}"
echo -e "\njobname=$jobname"
sbatch --job-name=$jobname \
       --error=./sbatch_out/${jobname}.e \
       --output=./sbatch_out/${jobname}.o \
       --export=GRANULE_DATE=$granule_date,VERSION_CAL_LID_L1=$VERSION_CAL_LID_L1,\
TYPE_CAL_LID_L1=$TYPE_CAL_LID_L1,PREVIOUS_GRANULE=$previous_granule_date,NEXT_GRANULE=$next_granule_date,\
SLICE_START_END_TYPE=$SLICE_START_END_TYPE,SLICE_START=$SLICE_START,SLICE_END=$SLICE_END,\
SAVE_DEVELOPMENT_DATA=$SAVE_DEVELOPMENT_DATA,VERSION_2D_McDA=$VERSION_2D_McDA,\
TYPE_2D_McDA=$TYPE_2D_McDA,OUT_FOLDER=$OUT_FOLDER 2D_McDA.sbatch
