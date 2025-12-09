#!/bin/bash -l

#SBATCH --time=24:00:00
#SBATCH --mem=1000

# Activer Python virtual environment
eval "$(/work_users/vaillant/mambaforge/bin/mamba shell hook --shell bash)"
if ! mamba activate 2D_McDA_PSC; then
    echo "Failed to activate environment" >&2
    exit 1
fi

# Export PYTHONPATH
export PYTHONPATH="/home/vaillant/codes/projects/2D_McDA_PSC/my_modules/"

# Run program
./2D_McDA_PSC.py "$GRANULE_DATE" "$VERSION_CAL_LID_L1" "$TYPE_CAL_LID_L1" \
                 "$SLICE_START_END_TYPE" "$SLICE_START" "$SLICE_END" \
                 "$LAT_MIN" "$LAT_MAX" \
                 "$SAVE_DEVELOPMENT_DATA" "$VERSION_2D_McDA_PSC" \
                 "$TYPE_2D_McDA_PSC" "$OUT_FOLDER" "$OUT_FILETYPE" "$PROCESS_UP_TO_40KM"
