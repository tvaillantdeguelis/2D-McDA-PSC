#!/bin/bash

FOLDER="/home/vaillant/codes/projects/2D_McDA_PSC/out/data/2D_McDA_PSC.v1.2.1/2010/"
FIGURES_PATH="/home/vaillant/codes/projects/2D_McDA_PSC/out/figures/quicklooks/"

# Find all .hdf files and process them
find "$FOLDER" -type f -name "CAL_LID_L2_2D_McDA_PSC-Prototype*.hdf" | sort | while read -r filepath; do
    # Extract the filename from the path
    filename=$(basename "$filepath")

    # Extract the granule using parameter expansion and pattern matching
    # Remove the prefix and suffix
    granule=${filename#CAL_LID_L2_2D_McDA_PSC-Prototype-V1-2-1.}
    granule=${granule%.hdf}

    # Call the Python script with the granule
    echo "$granule"
    ./plot_detection_masks.py "$granule" "$FIGURES_PATH"
done