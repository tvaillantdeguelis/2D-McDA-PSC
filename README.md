# 2D-McDA-PSC

Two-dimensional and multi-channel feature detection algorithm for the CALIPSO lidar measurements.

## Overview

The environment required to run the program is defined in the file:

```environment.yml```

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

Create the environment:

```conda env create -f environment.yml```

This will download and install all required software.
The installation may take several minutes.

## 4. Activate the environment

Before running the program, activate the environment:

```conda activate 2D_McDA_PSC```

Your terminal will now use the correct Python version and libraries.

## 5. Running the program locally

You can run the main Python script directly:

```python src/2D_McDA_PSC.py```

The parameters controlling what is processed (granule slice, dates, etc.) are located in the script immediately after:

```
if __name__ == '__main__':
    tic_main_program = print_time()
    
    # <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
    # PARAMETERS
    if len(sys.argv) > 1:
        ...
    else:
```

Modify these values before running the script to process the desired granule slice.

## 6. Running the program on a computing cluster (SLURM)

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