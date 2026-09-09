import numpy as np
import cmocean
from utils import (prepare_pr, 
                   global_mean, 
                   zonal_mean,
                   plot_reference_anomalies, 
                   plot_mean_map, 
                   plot_zonal_mean, 
                   plot_zonal_anomalies, 
                   plot_zonal_time,
                   plot_global_mean_vs_logco2)

class PRDiagnostics:
    """
    Calculate and plot precipitation diagnostics for multiple experiments.

    Parameters:
    atm : dict[str, xarray.Dataset]
        Atmospheric datasets indexed by experiment name.

    reference : str
        Name of the reference experiment used to calculate anomalies.

    co2_levels_log2 : dict [str, float]
        Base-2 logarithm of the atmospheric CO2 concentration associated with each experiment.

    plot_dirs : dict[str, pathlib.Path]
        Output directories associated with each experiment.

    Attributes:
    pr : dict[str, xarrat.DataArray]
        Precipitation fields converted to mm day-1.

    annual_mean : dict[str, xarray.DataArray]
        Mean precipitation fields over the analysis period.

    anomaly : dict[str, xarray.DataArray]
        Mean precipitation anomalies relative to the reference experiment.

    zonal_mean : dict[str, xarray.DataArray]
        Zonal means  of the mean precipitation fields.

    zonal_anom : dict[str, xarray.DataArray]
        Zonal precipitation anomalies relative to the reference experiment.

    zonal_ts : dict[str, xarray.DataArray]
        Time-dependent zonal mean precipitation.

    global_mean : dict[str, xarray.DataArray]
        Global mean precipitation averaged over the analysis period.

    anom : dict[str, xarray.DataArray]
        Time-dependent precipitation anomalies relative to the reference zonal climatology.

    anom_zonal : dict[str, xarray.DataArray]
        Zonal mean of the time-dependent precipitation anomalies.
    """
    def __init__(self, atm, reference, co2_levels_log2, plot_dirs):
        """
        Initialize the precipitation diagnostics.
        """

        # Input data and configuration
        self.atm = atm
        self.reference = reference
        self.co2_levels_log2 = co2_levels_log2
        self.plot_dirs = plot_dirs

        # Precipitation diagnostics
        self.pr = {}
        self.annual_mean = {}
        self.anomaly = {}

        # Reference-based time-dependent anomalies
        self.pi_zonal = {}
        self.pi_2d = {}
        self.anom = {}
        self.anom_zonal = {}

        # Zonal diagnostics
        self.zonal_mean = {}
        self.zonal_anom = {}
        self.zonal_ts = {}

        # Global diagnostics
        self.global_mean = {}



    def run(self, plot=True, plot_experiment=None):
        """
        Run all precipitation diagnostics.
        The method first calculated the precipitation diagnostics and, optionally, produces the associated figures.

        Parameters:
        plot : bool, default= True
            If True, generate diagnostic figures.

        plot_experiment : str, optional
            Experiment used for diagnostics that display a single experiment.
            If None, the reference experiment is used.

        Returns:
        PRDiagnostics
            The diagnostics object containng the calculated results.
        """

        self._calculate()
        
        if plot:
            self._plot(plot_experiment=plot_experiment)

        return self
    
    def _calculate(self):
        """
        Calculate precipitation diagnostics for all experiments.
        The calculations include mean fields, anomalies relative to the reference experiment,
        zonal means, zonal time series, and global mean precipitation.
        """
             
        # Define Pr, convert precipitation to mm day-1
        self.pr = {
            k: prepare_pr(ds["pr"])
            for k, ds in self.atm.items()
        }

        # Mean precipitation fields
        # Mean Pr over the complete analysis period
        self.annual_mean = {
            k: v.mean("time_counter")
            for k, v in self.pr.items()
        }

        # Spatial precipitation anomalies relative to reference experiment.
        self.anomaly = {
            k: self.annual_mean[k] - self.annual_mean[self.reference]
                for k in self.annual_mean
                if k != self.reference
        }

        # Time-dependent anomalies relative to the reference zonal field
        # Zonal mean precipitation of the reference experiment
        # This retains the time dimension and therefore represents the time-dependent
        # zonal precipitation structure of the reference simulation.
        self.pi_zonal = zonal_mean(self.pr[self.reference])

        # Time-dependent anomalies relative to reference
        # Broadcast the reference zonal precipitation field to the full latitude-longitude grid of 
        # each perturbation experiment.
        pi_2d = {
            k: self.pi_zonal.broadcast_like(v) 
            for k, v in self.pr.items() 
            if k != self.reference
        }
        
        # Time-dependent precipitation anomalies relative to the reference zonal precipitation field.
        self.anom = {
            k: self.pr[k] - pi_2d[k] 
            for k in pi_2d
        }

        # Zonal meann of the time-dependent anomalies
        self.anom_zonal = {
            k: v.mean("lon") 
            for k, v in self.anom.items()
        }

        # Zonal mean diagnostics
        # Zonal mean of annual mean Pr field
        self.zonal_mean = {
            k: zonal_mean(v)
            for k, v in self.annual_mean.items()
        }

        #Zonal mean precipitation anomalies relative to reference experiment
        self.zonal_anom = {
            k: self.zonal_mean[k] - self.zonal_mean[self.reference]
            for k in self.zonal_mean
            if k != self.reference
        }

        # Time-dependent zonal mean precipitation
        self.zonal_ts = {
        k: zonal_mean(v)
        for k, v in self.pr.items()
        }

        #Global mean precipitation averaged over the analysis period
        self.global_mean = {
            k: global_mean(v).mean("time_counter")
            for k, v in self.pr.items()
        }


    def _plot(self, plot_experiment=None):
        """
        Generate precipitation diagnostic figures.

        Parameters:
        plot_experiment : str, optional
            Experiment used for plots that display a single experiment.
            If None, the reference experiment is used.

        Raises:
        ValueError
            If the requested experiment is not available.
        """

        # Default: plot the reference experiment
        if plot_experiment is None:
            plot_experiment = self.reference

        exp = plot_experiment

        # Check that the requested experiment exists
        if exp not in self.atm:
            raise ValueError(
                f"Unknown experiment '{exp}'. "
                f"Available experiments: {list(self.atm.keys())}"
            )

        # Create the output directory foor precipitation diagnostics
        plot_dir = self.plot_dirs[exp] / "pr"

        plot_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Mean annual pr for one experiment
        plot_mean_map(
            field=self.annual_mean[exp],
            experiment=exp,
            title="Mean annual precipitation",
            levels=np.arange(0, 20.5, 0.5),
            cmap=cmocean.cm.rain,
            colorbar_label="Precipitation (mm/day)",
            save=plot_dir / f"pr_{exp}.png",
            )

        # Mean annual pr for the reference experiment and precipitation anomalies
        plot_reference_anomalies(
            reference=self.annual_mean[self.reference],
            anomalies=self.anomaly,
            reference_name=self.reference,
            absolute_title="Mean annual precipitation",
            anomaly_title="Precipitation anomaly",
            absolute_levels=np.arange(0, 20.5, 0.5),
            anomaly_levels=np.arange(-10, 10.5, 0.5),
            absolute_cmap=cmocean.cm.rain,
            anomaly_cmap=cmocean.cm.balance,
            absolute_label="Precipitation (mm/day)",
            anomaly_label="Precipitation anomaly (mm/day)",
            save=plot_dir / "pr_maps.png",
            )

        # Zonal mean annual pr for one experiment
        plot_zonal_mean(
            field=self.zonal_mean[exp],
            experiment=exp,
            title="Zonal mean precipition",
            ylabel="Zonal mean PR (mm/day)",
            save=plot_dir / f"pr_zonal_mean_{exp}.png"
            )

        # Zonal mean annual pr anomalies
        plot_zonal_anomalies(
            anomalies=self.zonal_anom,
            title="Zonal-mean precipitation anomaly",
            ylabel="Zonal-mean Pr anomaly (mm/day)",
            save=plot_dir / "pr_zonal_anom.png"
        )

        # Zonal precipitation evolution for one experiment
        plot_zonal_time(
            field=self.zonal_ts[exp],
            experiment=exp,
            title="Zonal mean precipitation evolution",
            cmap=cmocean.cm.rain,
            colorbar_label="Precipitation (mm/day)",
            levels=20,
            save=plot_dir / f"pr_zonal_time_{exp}.png"
        )


        # Global mean pr vs CO2 for multiple experiments
        plot_global_mean_vs_logco2(
            values=self.global_mean,
            logco2_levels=self.co2_levels_log2,
            ylabel="Global mean precipitation (mm/day)",
            xlabel="log₂(CO₂)",
            title="Global mean precipitation vs CO₂",
            ylim=(2.5, 5),
            xlim=(-0.1, 3),
            save=plot_dir / "global_mean_pr_logco2.png"
        )
