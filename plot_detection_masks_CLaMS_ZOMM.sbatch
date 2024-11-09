#!/bin/bash -l

#SBATCH --time=1:00:00
#SBATCH --mem=1000

# Activer Python virtual environment
source ~/.zshrc
mamba activate 2D_McDA_PSC_v1.0

# Run program
./plot_detection_masks_CLaMS_ZOMM.py $FILENAME_2D_McDA_PSCs_ZOMM_CLAMS
