#!/bin/sh

# Get environment name (first line of environment.yml: "name: var_name")
first_line=$(head -n 1 environment.yml)
env_name=${first_line#*: } # remove everything up to a colon and space

# Create Python environment
echo "Creating $env_name Python virtual environment..."
mamba env create -f environment.yml # also works with 'conda' instead of 'mamba'
