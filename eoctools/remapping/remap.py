#!/usr/bin/env python
"""Command-line interface for running the remaping of the eocene_prospoccesing workflow."""

import os
import glob
import re
import tempfile
import argparse
import xarray as xr
from cdo import Cdo

from eoctools.common.yaml import load_yaml

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

def process_dataset(cdo,
                    input_folder,
                    output_folder,
                    filename_filter,
                    variables,
                    remap_method,
                    output_prefix,
                    experiment_name,
                    exclude_filter=None,
                    grid="r180x90",
                    overwrite=False
                    ):
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

    overwrite : bool
        If True, an existing output file is replaced.
        If False (default), it is left untouched.
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

    # Output filename is known from the input filenames, so check it
    # before doing any (expensive) processing
    years = [extract_year(os.path.basename(f)) for f in files]

    output_file = os.path.join(
        output_folder,
        f"{experiment_name}_{output_prefix}_annual_mean_{min(years)}-{max(years)}.nc"
    )

    if os.path.exists(output_file):
        if not overwrite:
            print(f"Already exists: {output_file} (use --overwrite to replace it)")
            return
        print(f"Overwriting: {output_file}")

    annual_list = []

    for infile in files:

        filename = os.path.basename(infile)
        print(f"Processing {filename}")

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
        with xr.open_dataset(
            tmp_file,
            use_cftime=True,
            decode_times=True
        ) as ds:

            annual = ds.load()

        annual_list.append(annual)

        # remove temporary file
        os.remove(tmp_file)

    final = xr.concat(
        annual_list,
        dim="time_counter"
    )

    final.to_netcdf(
        output_file,
        encoding={
            "time_counter": {
                "calendar": "gregorian",
                "units": "days since 1850-01-01"
            }
        }
    )

    print(
        f"\n Saved: {output_file}"
    )

def annual_mean_moc(input_folder, 
                    output_folder,
                    filename_filter, 
                    variables,
                    experiment_name,
                    overwrite=False
                    ):
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

    overwrite : bool
        If True, an existing output file is replaced.
        If False (default), it is left untouched.
    """

    os.makedirs(output_folder, exist_ok=True)

    files = sorted(
        glob.glob(os.path.join(input_folder, f"*{filename_filter}*.nc"))
    )

    if len(files) == 0:
        print("No matching files found.")
        return

    years = [extract_year(os.path.basename(f)) for f in files]

    output_file = os.path.join(
        output_folder,
        f"{experiment_name}_moc_annual_mean_{min(years)}-{max(years)}.nc"
    )

    if os.path.exists(output_file):
        if not overwrite:
            print(f"Already exists: {output_file} (use --overwrite to replace it)")
            return
        print(f"Overwriting: {output_file}")

    annual_list = []

    for file in files:

        print(f"Processing {os.path.basename(file)}")

        with xr.open_dataset(
            file,
            use_cftime=True
        ) as ds:

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

    # Concatenate all annual means
    final = xr.concat(annual_list, dim="time_counter")

    final.to_netcdf(
        output_file,
        encoding={
            "time_counter": {
                "calendar": "gregorian",
                "units": "days since 1850-01-01"
            }
        }
    )

    print(f"\n Saved {output_file}")


def main(config_file, product="all", overwrite=False):

    
    config = load_yaml(config_file)

    cdo = Cdo(cdo=config["cdo_path"])

    base_output = config["base_output"]
    experiments = config["experiments"]

    atm_vars = config["variables"]["atmosphere"]
    oce_vars = config["variables"]["ocean"]
    moc_vars = config["variables"]["moc"]

    filters = config["filters"]

    for exp_name, input_base in experiments.items():

        print(f"\n================ {exp_name} ================\n")

        # MOC
        if product in ("moc", "all"):
            annual_mean_moc(
                input_folder=os.path.join(input_base, "nemo"),
                output_folder=os.path.join(base_output, exp_name, "processed"),
                filename_filter=filters["moc"],
                variables=moc_vars,
                experiment_name=exp_name,
                overwrite=overwrite
            )

        # Atmosphere
        if product in ("atm", "all"):
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
                experiment_name=exp_name,
                exclude_filter=filters["atm_exclude"],
                overwrite=overwrite
            )

        # Ocean
        if product in ("oce", "all"):
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
                output_prefix="oce",
                experiment_name=exp_name,
                overwrite=overwrite
            )



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to YAML config file"
    )

    parser.add_argument(
        "-p",
        "--product",
        choices=["atm", "oce", "moc", "all"],
        default="all",
        help="Select which dataset to process: atm, oce, moc, or all (default)"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files that already exist (default: skip them)"
    )

    args = parser.parse_args()

    main(
        args.config,
        product=args.product,
        overwrite=args.overwrite
        )