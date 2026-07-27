#!/usr/bin/env python
"""Command-line interface for running the remaping of the eocene_prospoccesing workflow."""

import os
import glob
import re
import tempfile
import common.yaml as yaml
import argparse
from common import load_yaml
import xarray as xr
from cdo import Cdo


def extract_year(filename):
    """
    Extract first year from filenames:
    """

    match = re.search(r"(\d{4})-\d{4}", filename)

    if match:
        return int(match.group(1))

    raise ValueError(
        f"Could not extract year from {filename}"
    )

def process_dataset(cdo,input_folder,output_folder,filename_filter,variables,remap_method,output_prefix,exclude_filter=None,grid="r180x90"):
    """
    Select variables, calculate annual means, remap,
    concatenate all years and write one NetCDF file.

    Parameters
    ----------
    input_folder : str
        Folder containing monthly files.

    output_folder : str
        Where the final file is written.

    filename_filter : str
        String identifying input files.

    variables : str
        Variables separated by commas.

    remap_method : str
        "remapcon" or "remapbil".

    output_prefix : str
        Final filename prefix.
        Example: "atm" or "oce"

    exclude_filter : str
        Files containing this string are ignored.

    grid : str
        CDO target grid.
        r180x90 = 2° x 2°
    """

    os.makedirs(output_folder, exist_ok=True)

    files = sorted(
        glob.glob(
            os.path.join(
                input_folder,
                f"*{filename_filter}*.nc"
            )
        )
    )

    if exclude_filter:

        files = [
            f for f in files
            if exclude_filter not in os.path.basename(f)
        ]

    if len(files) == 0:
        print("No files found.")
        return

    print(f"Found {len(files)} files for {output_prefix}")

    annual_list = []
    years = []

    for infile in files:

        filename = os.path.basename(infile)
        print(f"Processing {filename}")

        year = extract_year(filename)
        years.append(year)

        # temporary files
        tmp_file = tempfile.NamedTemporaryFile(
            suffix=".nc",
            delete=False
        ).name

        cdo_command = (
            f"-{remap_method},{grid} "
            f"-yearmean "
            f"-selname,{variables} "
            f"{infile}"
        )

        cdo.copy(
            input=cdo_command,
            output=tmp_file
        )
        with xr.open_dataset(tmp_file) as ds:

            annual = ds.load()

        annual_list.append(annual)

        # remove temporary file
        os.remove(tmp_file)

    final = xr.concat(
        annual_list,
        dim="time_counter"
    )

    first_year = min(years)
    last_year = max(years)

    output_file = os.path.join(
        output_folder,
        f"{output_prefix}_annual_mean_{first_year}-{last_year}.nc"
    )

    if os.path.exists(output_file):
        print(f"Already exists: {output_file}")
        return

    final.to_netcdf(output_file)

    print(
        f"\n Saved: {output_file}"
    )

def annual_mean_moc(input_folder, output_folder,
                    filename_filter, variables):
    """
    Compute annual means for zonal-mean diagnostics (msftyz), and save a single output file.

    Parameters
    ----------
    input_folder : str
        Folder containing the original NEMO monthly files.

    output_folder : str
        Folder where the final file will be written.

    filename_filter : str
        Pattern identifying the desired files.
        
    variables : list[str]
        Variables to retain.
    """

    os.makedirs(output_folder, exist_ok=True)

    files = sorted(
        glob.glob(os.path.join(input_folder, f"*{filename_filter}*.nc"))
    )

    if len(files) == 0:
        print("No matching files found.")
        return

    annual_list = []

    years = []

    for file in files:

        print(f"Processing {os.path.basename(file)}")

        with xr.open_dataset(file) as ds:

            annual = (
                ds[variables]
                .mean(dim="time_counter", keep_attrs=True)
                .expand_dims(
                    time_counter=[ds.time_counter.values[0]]
                )
            )

            # Load the small annual dataset into memory so
            # the file can be closed immediately.
            annual.load()

            annual_list.append(annual)

        # Extract year from filename
        match = re.search(r"(\d{4})-(\d{4})", os.path.basename(file))
        if match:
            years.append(int(match.group(1)))

    # Concatenate all annual means
    final = xr.concat(annual_list, dim="time_counter")

    first_year = min(years)
    last_year = max(years)

    var_string = "_".join(variables)

    output_file = os.path.join(
        output_folder,
        f"moc_annual_mean_{first_year}-{last_year}.nc"
    )

    final.to_netcdf(output_file)

    print(f"\n Saved {output_file}")


def main(config_file):

    with open(config_file) as f:
        config = yaml.safe_load(f)

    cdo = Cdo(cdo=config["cdo_path"])

    base_output = config["base_output"]
    experiments = config["experiments"]

    atm_vars = config["variables"]["atmosphere"]
    oce_vars = config["variables"]["ocean"]
    moc_vars = config["variables"]["moc"]

    filters = config["filters"]

    for exp_name, input_base in experiments.items():

        print(f"\n================ {exp_name} ================\n")

        # Moc
        annual_mean_moc(
            input_folder=os.path.join(input_base, "nemo"),
            output_folder=os.path.join(base_output, exp_name, "processed"),
            filename_filter=filters["moc"],
            variables=moc_vars
        ) 

        # Atmosphere
        process_dataset(
            cdo=cdo,
            input_folder=os.path.join(
                input_base,
                "oifs"
            ),
            output_folder=os.path.join(
                base_output,
                exp_name,
                "processed"
            ),
            filename_filter=filters["atm"],
            variables=atm_vars,
            remap_method="remapcon",
            output_prefix="atm",
            exclude_filter=filters["atm_exclude"]
        )

        # Ocean
        process_dataset(
            cdo=cdo,
            input_folder=os.path.join(
                input_base,
                "nemo"
            ),
            output_folder=os.path.join(
                base_output,
                exp_name,
                "processed"
            ),
            filename_filter=filters["oce"],
            variables=oce_vars,
            remap_method="remapbil",
            output_prefix="oce"
        )



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to YAML config file"
    )

    args = parser.parse_args()
    config = load_yaml(args.config)

    main(args.config)