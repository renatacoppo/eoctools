#!/usr/bin/env python
"""Command-line interface for running eoctools diagnistics"""

import argparse
from diagnostics import Diagnostics

def main(config_file, diagnostics, experiment=None, no_plots=False):
    diag=Diagnostics(config_file)

    print("\n=============== GENERAL INFORMATION ===============\n")

    print("Experiments:")
    print(diag.experiment_names)

    print("\nCO2 levels:")
    print(diag.co2_levels)

    print("\nReference:")
    print(diag.reference)

    plot = not no_plots

    # ==========================================================
    # TAS diagnostics
    # ==========================================================
    if "tas" in diagnostics:
        print("\n========== TAS ==========\n")

    diag.tas.run(
        plot=plot,
        plot_experiment=experiment
    )

    # ==========================================================
    # PR diagnostics
    # ==========================================================
    if "pr" in diagnostics:
        print("\n========== PR ==========\n")
    
    diag.pr.run(
        plot=plot,
        plot_experiment=experiment
    )

    # ==========================================================
    # TOA diagnostics
    # ==========================================================
    if "toa" in diagnostics:
        print("\n========== TOA ==========\n")

    diag.toa.run(
        plot=plot
    )

    # ==========================================================
    # SST diagnostics
    # ==========================================================
    if "sst" in diagnostics:
        print("\n========== SST ==========\n")

    diag.tos.run(
        plot=plot,
        plot_experiment=experiment
    )

    # --------------------------------------------------
    # SOS
    # --------------------------------------------------
    if "sos" in diagnostics:
        print("\n================ SOS ================\n")

        diag.sos.run(
            plot=plot,
            plot_experiment=experiment
        )

    # ==========================================================
    # DeepMIP diagnostics
    # ==========================================================

    diag.tas.run(plot=False)
    diag.tos.run(plot=False)

    diag.deepmip.run(
        plot=plot,
        own_results=diag.tas.global_mean,
        own_co2_levels=diag.co2_levels,
        own_reference=diag.reference,
        own_sst_results=diag.tos.global_mean,
        own_sst_gradient=diag.tos.dT,
    )

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(
            description="Run eoctools climate diagnostics."
        )

        parser.add_argument(
            "-c",
            "--config",
            required=True,
            help="Path to diagnostics YAML configuration file"
        )

        parser.add_argument(
            "--tas",
            action="store_true",
            help="Run surface air temperature diagnostics."
        )

        parser.add_argument(
            "--pr",
            action="store_true",
            help="Run precipitation diagnostics"
        )

        parser.add_argument(
            "--toa",
            action="store_true",
            help="Run radiation diagnostics."
        )

        parser.add_argument(
            "--sst",
            action="store_true",
            help="Run sea-surface temperature diagnostics"
        )

        parser.add_argument(
            "--sos",
            action="store_true",
            help="Run sea-surface salinity diagnostics."
        )

        parser.add_argument(
            "--deepmip",
            action="store_true",
            help="Run DeepMIP comparison diagnostics"
        )

        parser.add_argument(
            "--all",
            action="store_true",
            help="Run all diagnostics"
        )

        parser.add_argument(
            "-e",
            "--experiment",
            default=None,
            help="Experiment to use for single-experiment plots"
        )

        parser.add_argument(
            "--no-plots",
            action="store_true",
            help="Calculate diagnostics without generating plots"
        )

        args=parser.parse_args()

        if args.all:
            diagnostics = [
                "tas",
                "pr",
                "toa",
                "sst",
                "sos",
                "deepmip"
            ]
        else:
            diagnostics=[]

            if args.tas:
                diagnostics.append("tas")

            if args.pr:
                diagnostics.append("pr")

            if args.toa:
                diagnostics.append("toa")

            if args.sst:
                diagnostics.append("sst")

            if args.sos:
                diagnostics.append("sos")

            if args.deepmip:
                diagnostics.append("deepmip")

        if not diagnostics:
            parser.error(
                "No diagnostics selected. Use --all or one of --tas, --pr, --toa, --sst, --sos, --deepmip"
            )

        main(
            config_file=args.config,
            diagnostics=diagnostics,
            experiment=args.experiment,
            no_plots=args.no_plots
        )

