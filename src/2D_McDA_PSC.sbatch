#!/bin/bash -l

#SBATCH --time=24:00:00
#SBATCH --mem=1000

# Activate Python environment
eval "$(/work_users/vaillant/mambaforge/bin/mamba shell hook --shell bash)"
if ! mamba activate 2D_McDA_PSC; then
    echo "Failed to activate environment" >&2
    exit 1
fi

# PYTHONPATH
export PYTHONPATH="/home/vaillant/codes/projects/2D_McDA_PSC/my_modules/"

# Safety check
if [ -z "$PARAMS_FILE" ]; then
    echo "PARAMS_FILE is not set" >&2
    exit 1
fi

# Run algorithm
./2D_McDA_PSC.py "$PARAMS_FILE"
