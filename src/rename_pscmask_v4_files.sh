#!/bin/bash

VERSION="2.4.2"

# Convert version to file pattern format: VX-X-X
VERSION_TAG="V$(echo "$VERSION" | tr '.' '-')"

ROOT_DIR="/home/vaillant/codes/projects/2D_McDA_PSC/out/data/2D_McDA_PSC.v${VERSION}"

find "$ROOT_DIR" -type f -name "CAL_LID_L2_2D_McDA_PSC-Prototype-${VERSION_TAG}*.nc" | while IFS= read -r file
do
    dir=$(dirname "$file")
    base=$(basename "$file")

    # Extract granule datetime
    datetime=$(echo "$base" | sed -E "s/^CAL_LID_L2_2D_McDA_PSC-Prototype-${VERSION_TAG}\.([0-9T:-]+ZN)_lon_.*/\1/")

    newname="CAL_LID_L2_PSCMask-Standard-V4-00.${datetime}.nc"

    echo "Renaming:"
    echo "  $file"
    echo "  -> $dir/$newname"

    mv "$file" "$dir/$newname"
done