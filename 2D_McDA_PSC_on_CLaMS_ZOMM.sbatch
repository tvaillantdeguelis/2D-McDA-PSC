#!/bin/bash -l

#SBATCH --time=1:00:00
#SBATCH --mem=1000

# Activer Python virtual environment
mamba activate 2D_McDA_for_PSCs_env001

# Run program
./2D_McDA_on_CLaMS_ZOMM.py $ZOMM_CLAMS_FILENAME $ZOMM_CLAMS_PATH $SAVE_DEVELOPMENT_DATA $VERSION_2D_McDA $TYPE_2D_McDA $OUT_FOLDER $CNF
