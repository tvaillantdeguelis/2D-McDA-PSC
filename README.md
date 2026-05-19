# 2D-McDA-PSC

Two-dimensional and multi-channel feature detection algorithm for the CALIPSO lidar measurements.

## Overview

The main Python program is:

```src/2D_McDA_PSC.py```

It can be executed:

* Locally, to process a single granule slice defined in the parameters inside the script.

* On a computing cluster, to process all granules between a start and end date, using the SLURM job script:

```src/2D_McDA_PSC_run.sh```

## 1. Download the program

Go to https://github.com/tvaillantdeguelis/2D-McDA-PSC/ and click the green **Code** button near the top right. In the dropdown menu, click **Download ZIP**. Save the ZIP file anywhere on your computer and extract it.

On the same page, click on "my_modules", then click **Download ZIP**. Add the downloaded files under the folder "my_modules" of the project.

## 2. Install Conda

To run the program, **if you don't already have a Python environment**, you can install Miniconda.
https://www.anaconda.com/docs/getting-started/miniconda/install

After installation:

Windows: open Anaconda Prompt

macOS/Linux: open a terminal

## 3. Create the Conda environment

Create a Conda environment. For example, create an environment named 2D_McDA_PSC in python 3.13:

```conda create -n 2D_McDA_PSC python=3.13```

Activate the environment:

```conda activate 2D_McDA_PSC```

Install the packages required for this program:

```conda install numpy numba scipy matplotlib pyhdf netcdf4```

## 4. Define the paths where the CALIOP files are stored

In my_modules/paths.py define the paths where the CALIOP files are stored for the 'hostname' machine.

```
elif hostname == 'argo':  # change by the name of your machine
    # Head paths
    CALIOP_DATA_HEAD_PATH = "/SCF10/Data_Archive/CALIPSO/"  # change by the path to CALIPSO data on your machine
    # Tail paths format
    CALIOP_DATA_TAIL_PATH_FMT['L1'] = "LID_L1.-{data_type}-{version}/{year:d}/{month:02d}/"  # change according the the subfolders organisation on your machine ('data_type' and 'version' are values set for parameters TYPE_CAL_LID_L1 and VERSION_CAL_LID_L1 in 2D_McDA_PSC_run.sh or 2D_McDA_PSC.py)
```

## 5. Running the program locally on a single granule

Program parameters are configured through a YAML configuration file.

### 5.1. Create a run parameter file

Copy the template to a new file:

```cp src/2D_McDA_PSC_params_template.yaml src/2D_McDA_PSC_params_run.yaml```

Edit `src/2D_McDA_PSC_params_run.yaml` to set the granule you want to process:

```
granule_date: "2010-01-18T00-19-57ZN"

cal_lid_l1:
  version: "V5.00"
  type: "Standard"

slice:
  mode: "latminmax"
  start: null
  end: null
  lat_min: 50
  lat_max: null

processing:
  save_development_data: false
  process_up_to_40km: false # PSC_Ice_Mixture_Boundary in PSCMask V3 only up to 30 km, classification will break if set to 'true'
  make_classification: true
  feature_separation_type_for_classification: "all_levels_and_channels" # "pixel", "channel", "best_detection_level", or "all_levels_and_channels"

algorithm:
  type: "Prototype"

output:
  folder: "/home/vaillant/codes/projects/2D_McDA_PSC/out/data/"
  filetype: "netCDF"
```

You can modify these values for any granule or slice you want.

### 5.2. Run the script with the YAML file

```
python src/2D_McDA_PSC.py src/2D_McDA_PSC_params_run.yaml
```

This allows you to run a single granule without modifying the Python code.

## 6. Running the program on a computing cluster (SLURM) on many granules

To process all granules between two dates, use the SLURM job script:

```src/2D_McDA_PSC_run.sh```

## 6.1. Configuration

Edit the run script:

```
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# CONFIGURATION
VERSION_CAL_LID_L1="V5.00"
TYPE_CAL_LID_L1="Standard"
SLICE_MODE="latminmax"
SLICE_START="null"
SLICE_END="null"
START_DATE="2008-01-01"
END_DATE="2011-12-31" # included
SAVE_DEVELOPMENT_DATA="false" # if "true" save step by step data
TYPE_2D_McDA_PSC="Prototype"
OUT_FILETYPE='netCDF' # 'HDF' or 'netCDF'
PROCESS_UP_TO_40KM="false"
MAKE_CLASSIFICATION="true"
FEATURE_SEPARATION_TYPE_FOR_CLASSIFICATION="all_levels_and_channels" # "pixel", "channel", "best_detection_level", or "all_levels_and_channels"
#-----------------------------------------------------------------------
DAYNIGHT_FLAG="ZN" # "ZN", "ZD", or ""
DATA_FOLDER_L1_HEAD="/DATA/LIENS/CALIOP/CAL_LID_L1."
OUT_FOLDER="/work_users/vaillant/data/2D_McDA_PSC/"
MAX_NB_JOBS=101
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
```

The script generates a YAML parameter file for each granule automatically.

## 6.2. Submit the job

From the project directory, run:

```sbatch src/2D_McDA_PSC_run.sh```


The cluster will automatically run the program on every granule in the date interval.