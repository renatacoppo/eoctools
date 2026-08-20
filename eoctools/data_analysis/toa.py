import xarray as xr
import numpy as np
from utils import plot_gregory, global_toa_ts

class TOADiagnostics:
    def __init__(self, atm, tas, reference, plot_dirs):

        self.atm = atm
        self.tas= tas
        self.reference = reference
        self.plot_dirs = plot_dirs

        self.tos = {}

    def run(self, plot=True, plot_experiment=None):

        self._calculate()
        
        if plot:
            self._plot(plot_experiment=plot_experiment)

        return self
    
    def _calculate(self):
        #Define TOA
        self.toa = {
            k: global_toa_ts(ds)
            for k, ds in self.atm.items()
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
        plot_dir = self.plot_dirs[exp] / "toa"

        plot_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        # Gregory plot for all experiments (the years plotted depend on the years loaded on atm)
        plot_gregory(
            tas_series=self.tas.tas_ts,
            toa_series=self.toa,
            title="Gregory plots – All experiments",
            save=plot_dir / "gregory_all_experiments.png"
        )