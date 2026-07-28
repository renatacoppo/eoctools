# Eocene tools

Eocene Tools is a Python package for processing and analysing output from Earth System Model (ESM) simulations, with an initial focus on EC-Earth experiments of Eocene climates.

## Repository structure
eocene_tools/
    common/
        utils.py
        yaml.py
    remapping/
        remap.py
        config_remap.yml
        run_remap.job
    data_analysis/

## Requirements
You need:
- a Python environment with the project dependencies installed
- acces to EC-Earth4 simulation outputs.
- access to the DeepMIP dataset
- external tools required by the selected workflow, especially `cdo` for several operations

## Installation

Clone the repository:
git clone https://github.com/renatacoppo/eoctools
cd eocene-tools

Create the conda environment:
mamba env create -f environment.yml
or
conda env create -f environment.yml

Activate the environment:
conda activate eoctools

Install the package in editable mode:
pip install -e .

## Dependencies

Main dependencies include:

Python ≥ 3.9
CDO
xarray
PyYAML
ecCodes

See environment.yml for the complete software environment.

## Running the remapping workflow

The remapping workflow produces reproducible workflows to extract variables from model output, compute annual means, remap data to common grids of 2ºx2º (if needed), and saves them in a new folder. 

It produces three main files:
- atm_annual_mean_{first_year}-{last_year}.nc  
- oce_annual_mean_{first_year}-{last_year}.nc
- moc_annual_mean_{first_year}-{last_year}.nc

The main entry point is:

```bash
./remapping/remap.py
```

## Command-line options

```text
-c, --config   Path to the YAML configuration file. Required.
```

Example:

```bash
./eoctools/remapping/remap.py -c eoctools/remapping/config_remap.yml
```

or, on an HPC system, submit the provided batch script:

```bash
sbatch eoctools/remapping/run_remap.job
```

## Configuration file
The remapping workflow is configured through a YAML file specifying:

- cdo_path: location of the CDO executable
- base_output: output directory
- experiments: input experiments path
- atmospheric variables (configured for cdo)
- ocean variables (configured for cdo)
- MOC variables (configured for xarray)
- filename filters (for selecting the file type from the simulation output)

## Running the data_analysis workflow



