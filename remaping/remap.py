import os
import yaml
from cdo import Cdo
import argparse


def remap_files(cdo, input_folder, output_folder, file_filter):
    """
    Remap netCDF files to 1°x1° grid using nearest neighbor.
    """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):

        if filename.endswith(".nc") and file_filter in filename:

            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, f"remapped_{filename}")

            if not os.path.exists(output_file):

                cdo.remapnn("r180x90", input=input_file, output=output_file)
                print(f"Remapped: {filename}")

            else:
                print(f"Skipped: {filename}")


def extract_variables(input_folder, output_folder, variables,
                      cdo, filename_filter=None, exclude_filter=None):

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):

        if not filename.endswith(".nc"):
            continue

        if filename_filter and filename_filter not in filename:
            continue

        if exclude_filter and exclude_filter in filename:
            continue

        infile = os.path.join(input_folder, filename)
        outfile = os.path.join(output_folder, f"small_{filename}")

        if not os.path.exists(outfile):

            cdo.selname(variables, input=infile, output=outfile)
            print(f"Extracted vars from {filename}")

        else:
            print(f"Skipped: {filename}")


def main(config_file):

    with open(config_file) as f:
        config = yaml.safe_load(f)

    cdo = Cdo(cdo=config["cdo_path"])

    base_output = config["base_output"]
    experiments = config["experiments"]

    atm_vars = config["variables"]["atmosphere"]
    oce_vars = config["variables"]["ocean"]

    filters = config["filters"]

    for exp_name, input_base in experiments.items():

        print(f"\n===== {exp_name} =====\n")

        # ATM remap
        remap_files(
            cdo,
            input_folder=os.path.join(input_base, "oifs"),
            output_folder=os.path.join(base_output, exp_name, "remapped/oifs"),
            file_filter=filters["atm_remap"]
        )

        # OCE remap
        remap_files(
            cdo,
            input_folder=os.path.join(input_base, "nemo"),
            output_folder=os.path.join(base_output, exp_name, "remapped/nemo"),
            file_filter=filters["oce_remap"]
        )

        # ATM variables
        extract_variables(
            input_folder=os.path.join(base_output, exp_name, "remapped/oifs"),
            output_folder=os.path.join(base_output, exp_name, "variables/atm"),
            variables=atm_vars,
            filename_filter=filters["atm_filename"],
            exclude_filter=filters["atm_exclude"],
            cdo=cdo
        )

        # OCE variables
        extract_variables(
            input_folder=os.path.join(base_output, exp_name, "remapped/nemo"),
            output_folder=os.path.join(base_output, exp_name, "variables/oce"),
            variables=oce_vars,
            cdo=cdo
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

    main(args.config)