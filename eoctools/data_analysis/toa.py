import xarray as xr
import numpy as np
from utils import plot_gregory, global_toa_ts, global_mean, global_surface_net_radiation, plot_radiation_balance

class TOADiagnostics:
    def __init__(self, atm, reference, plot_dirs, comparison_plot_dir):

        self.atm = atm
        self.reference = reference
        self.plot_dirs = plot_dirs
        self.comparison_plot_dir = comparison_plot_dir

        self.toa = {}
        self.tas = {}

        self.sfc = {}
        self.toa_sfc_difference = {}

    def run(self, plot=True, plot_experiment=None):

        self._calculate()
        
        if plot:
            self._plot(plot_experiment=plot_experiment)

        return self
    
    def _calculate(self):
        # Global mean TAS time series
        self.tas = {
            k: global_mean(ds["tas"] - 273.15)
            for k, ds in self.atm.items()
        }
        
        # Net TOA
        self.toa = {
            k: global_toa_ts(ds)
            for k, ds in self.atm.items()
        }

        # Net surface radiation
        self.sfc = {
            k: global_surface_net_radiation(ds)
            for k, ds in self.atm.items()
        }

        # TOA - SFC radiation imbalance
        self.toa_sfc_difference = {
            k: self.toa[k] - self.sfc[k]
            for k in self.atm
        }

    def _plot(self, plot_experiment=None):

        # Default: plot the reference experiment
        if plot_experiment is None:
            plot_experiment = self.reference

        exp = plot_experiment

        # Check that experiment exists
        if exp not in self.atm:
            raise ValueError(
                f"Unknown experiment '{exp}'. "
                f"Available experiments: {list(self.atm.keys())}"
            )

        # Directory for this experiment's TAS plots
        plot_dir = self.comparison_plot_dir / "toa"

        plot_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # --------------------------------------------------
        # Create filename from experiment directories
        # --------------------------------------------------

        experiment_dirs = [
            self.plot_dirs[exp].parent.name
            for exp in self.atm.keys()
        ]

        experiment_string = "_".join(experiment_dirs)

        save = (
            plot_dir
            / f"gregory_all_experiments_{experiment_string}.png"
        )

        # Gregory plot for all experiments (the years plotted depend on the years loaded on atm)
        plot_gregory(
            tas_series=self.tas,
            toa_series=self.toa,
            title="Gregory plots – All experiments",
            save=save
        )

        # TOA - SFC radiation balance
        save_balance = (
            plot_dir
            / f"toa_sfc_radiations_{experiment_string}.png"
        )

        plot_radiation_balance(
            toa=self.toa,
            sfc=self.sfc,
            difference=self.toa_sfc_difference,
            title="TOA - SFC radiation balance",
            save=save_balance
        )