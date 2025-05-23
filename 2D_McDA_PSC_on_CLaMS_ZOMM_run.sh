#!/bin/bash

# This program launches 2D-McDA on a list of CLaMS_ZOMM files.

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# CONFIGURATION
ZOMM_CLAMS_PATH="/home/vaillant/codes/projects/2D_McDA_for_PSCs/in/CLaMS_ZOMM/"
SAVE_DEVELOPMENT_DATA=false # if True save step by step data
VERSION_2D_McDA="V1.01"
TYPE_2D_McDA="Prototype"
OUT_FOLDER="/home/vaillant/codes/projects/2D_McDA_for_PSCs/out/data/CLaMS_ZOMM/"    
CNF=0.00442 # CALIOP noise factor (see Sect. 3.1 in Tritscher et al., 2019), value from Lamont Poole computations
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

launch_2d_mcda () {
    ZOMM_CLAMS_FILENAME="PSC_ZOMM_CLAMS_BKS_"$1"_v1.nc"
    jobname="2D-McDA_"$1
    echo -e "jobname=$jobname"
    sbatch --job-name=$jobname \
            --error=./sbatch_out/${jobname}.e \
            --output=./sbatch_out/${jobname}.o \
            --export=ZOMM_CLAMS_FILENAME="$ZOMM_CLAMS_FILENAME",ZOMM_CLAMS_PATH="$2",SAVE_DEVELOPMENT_DATA="$3",VERSION_2D_McDA="$4",TYPE_2D_McDA="$5",OUT_FOLDER="$6",CNF="$7" 2D_McDA_on_CLaMS_ZOMM.sbatch
}

for ((i = 0 ; i < 15 ; i++)); do
	printf -v granule "2011d176_%04d" $i
    launch_2d_mcda $granule $ZOMM_CLAMS_PATH $SAVE_DEVELOPMENT_DATA $VERSION_2D_McDA $TYPE_2D_McDA $OUT_FOLDER $CNF
done
