#!/bin/bash

# This program launches plots of 2D-McDA processed CLaMS_ZOMM files.

launch_plot () {
    FILENAME_2D_McDA_PSCs_ZOMM_CLAMS="2D_McDA_PSCs-PSC_ZOMM_CLAMS_BKS_"$1"_v1.nc"
    jobname="plot_"$1
    echo -e "jobname=$jobname"
    sbatch --job-name=$jobname \
            --error=./out/slurm/${jobname}.e \
            --output=./out/slurm/${jobname}.o \
            --export=FILENAME_2D_McDA_PSCs_ZOMM_CLAMS="$FILENAME_2D_McDA_PSCs_ZOMM_CLAMS" plot_detection_masks_CLaMS_ZOMM.sbatch
}

for ((i = 0 ; i < 15 ; i++)); do
	printf -v granule "2011d176_%04d" $i
    launch_plot $granule
done
