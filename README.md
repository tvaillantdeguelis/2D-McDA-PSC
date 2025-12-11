# 2D-McDA-PSC

Two-dimensional and multi-channel feature detection algorithm for the CALIPSO lidar measurements.

## Overview

The main Python program is:

```src/2D_McDA_PSC.py```

It can be executed:

Locally, to process a single granule slice defined in the parameters inside the script.

On a computing cluster, to process all granules between a start and end date, using the SLURM job script:

```src/2D_McDA_PSC_run.sh```

## 1. Download the program

Go to https://github.com/tvaillantdeguelis/2D-McDA-PSC/ and click the green **Code** button near the top right. In the dropdown menu, click **Download ZIP**. Save the ZIP file anywhere on your computer and extract it.

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

You can run the main Python script directly:

```python src/2D_McDA_PSC.py```

The parameters controlling what is processed (granule slice, dates, etc.) are located in the script immediately after `if __name__ == '__main__':`:

```ma
if __name__ == '__main__':
    tic_main_program = print_time()
    
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    # PARAMETERS
    if len(sys.argv) > 1:
        ...
    else:
        GRANULE_DATE = "2011-06-25T00-11-52ZN"
        VERSION_CAL_LID_L1 = "V4.51"
        TYPE_CAL_LID_L1 = "Standard"
        SLICE_START_END_TYPE = "latminmax" # "profindex", "longitude", "latminmax" (Use "profindex" if SLICE_START/END = None to process the whole granule)
        SLICE_START = None # 170.68 # profindex or longitude
        SLICE_END = None # 27.93 # profindex or longitude
        LAT_MIN = None # with SLICE_START_END_TYPE = "latminmax"
        LAT_MAX = -50 # SLICE_START_END_TYPE = "latminmax"
        SAVE_DEVELOPMENT_DATA = False # if True save step by step data
        VERSION_2D_McDA_PSC = "V1.4.1"
        TYPE_2D_McDA_PSC = "Prototype"
        OUT_FOLDER = "/home/vaillant/codes/projects/2D_McDA_PSC/out/data/"    
        OUT_FILETYPE = 'netCDF' # 'HDF' or 'netCDF'
        PROCESS_UP_TO_40KM = True
```

Modify these values before running the script to process the desired granule slice.

## 6. Running the program on a computing cluster (SLURM) on many granules

To process all granules between two dates, use the SLURM job script:

```src/2D_McDA_PSC_run.sh```

## 6.1. Configure the date range

Open the file and edit:

```
START_DATE="YYYY-MM-DD"
END_DATE="YYYY-MM-DD"
```

## 6.2. Submit the job

From the project directory, run:

```sbatch src/2D_McDA_PSC_run.sh```


The cluster will automatically run the program on every granule in the date interval.