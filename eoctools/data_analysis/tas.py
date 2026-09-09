import numpy as np
import cmocean
from utils import global_mean, zonal_mean, polar_mean 
from utils import plot_reference_anomalies, plot_mean_map, calculate_ecs, plot_zonal_mean 
from utils import plot_zonal_anomalies, plot_zonal_time, plot_global_mean_vs_co2, plot_global_timeseries

class TASDiagnostics:
    """
    Calculate and plot surface air temperature (TAS) diagnostics.
    The class operates on atmospheric datasets from multiple experiments.
    Temperature is converted from Kelvin to degrees Celsius before diagnostics are calculated.

    Diagnostics include spatial means, anomalies, zonal means, global means, temperature time series,
    polar amplification, and equilibrium climate sensitivity.

    Parameters
    atm : dict[str, xarray.Dataset]
        Dictionary containing atmospheric datasets for each experiment.

    reference : str
        Name of the reference experiment used to calculate anomalies.

    co2_levels : dict[str, float]
        Dictionary mapping experiment names to their atmospheric CO2 concentration relative to the pre-industrial value.

    plot_dirs : dict[str, pathlib.Path]
        Dictionary mapping experiment names to their base plot directories.
    """
    def __init__(self, atm, reference, co2_levels, plot_dirs):
        """
        Initialize the TAS diactnostics container.
        The constructor stores the input datasets and initializes empty dictionaries that will later 
        contain the calculated diagnostics.
        """

        # Input data and experiment configuration
        self.atm = atm
        self.reference = reference
        self.co2_levels = co2_levels
        self.plot_dirs = plot_dirs

        # Temperature fiels
        # Full TAS fields in degrees Celcius
        self.tas = {}

        # Mean TAS field over the analysis period
        self.annual_mean = {}

        # Spatial TAS anomalies relative to the reference experiment
        self.anomaly = {}

        # Zonal diagnostics
        # Time-mean zonal temperature profiles
        self.zonal_mean = {}

        # Zonal temperature anomalies rerlative to the reference
        self.zonal_anom = {}

        # Latitude-time evolution of zonal mean TAS
        self.zonal_ts = {}

        # Global diagnostics
        # Time-mean global surface air temperature
        self.global_mean = {}
        
        # Annual global mean TAS time series
        self.tas_ts = {}

        # Polar amplification diagnostics
        # Temperature change over the polar regions
        self.polar_warming = {}

        #Global mean temperature change
        self.global_warming = {}

        # Ratio of polar warming to global warming
        self.polar_amp = {}

        # Equilibrium climate sensitivity
        self.ECS = None

    def run(self, plot=True, plot_experiment=None):
        """
        Run all TAS diagnostics.
        The method first calculates all temperature diagnostics and then, optionally, generates the corresponding figures.
        """
        # Calculate all TAS diagnostics
        self._calculate()
        
        # Generate figures if requested
        if plot:
            self._plot(plot_experiment=plot_experiment)

        return self
    
    def _calculate(self):
        """
        Calculate all surface air temperature diagnostics.
        The calculations are performed in the following order:
        1. Convert TAS from Kelvin to degrees Celcius.
        2. Calculate mean temperature fields over the analysis period.
        3. Calculate spatial anomalies relative to the reference experiment.
        4. Calculate zonal mean temperature diagnostics.
        5. Calculate global mean temperature diagnostics.
        6. Calculate polar and global warming.
        7. Calculate polar amplification.
        8. Estimate equilibrium climate sensitivity.
        """
             
        #1. TAS from Kelvin to degrees Celcius.
        self.tas = {
            k: ds["tas"] - 273.15
            for k, ds in self.atm.items()
        }
             
        #2. Mean temperature fields over the analysis period.
        self.annual_mean = {
            k: v.mean("time_counter")
            for k, v in self.tas.items()
        }

        #3. Spatial temperature anomalies relative to the reference experiment.
        self.anomaly = {
            k: self.annual_mean[k] - self.annual_mean[self.reference]
                for k in self.annual_mean
                if k != self.reference
        }

        #4. Zonal mean temperature diagnostics.
        self.zonal_mean = {
            k: zonal_mean(v)
            for k, v in self.annual_mean.items()
        }

        #5. Zonal temperature anomalies relative to the reference experiment
        self.zonal_anom = {
            k: self.zonal_mean[k] - self.zonal_mean[self.reference]
            for k in self.zonal_mean
            if k != self.reference
        }

        #6. Latitude-time evolution
        # Calculate the zonal mean for every year, preserving the time dimension 
        # to show the evolution of temperature with latitude.
        self.zonal_ts = {
            k: zonal_mean(v)
            for k, v in self.tas.items()
        }

        #7. Global mean TAS
        # First calculate the area-weighted global mean for each year, then
        # average over the analysis period.
        self.global_mean = {
            k: global_mean(v).mean("time_counter")
            for k, v in self.tas.items()
        }

        # Preserve the annual global mean time series for transient temperature
        # evolution diagnostics.
        self.tas_ts = {
            k: global_mean(v)
            for k, v in self.tas.items()
        }

        #7. Calculate polar and global warming.
        # Warming is defined relative to the reference experiment.
        self.polar_warming = {
            k: (
                polar_mean(self.annual_mean[k]) 
                - polar_mean(self.annual_mean[self.reference])
            )
            for k in self.annual_mean
            if k!= self.reference
        }
        # Global mean temperature change
        self.global_warming = {
            k: (
                global_mean(self.annual_mean[k])
                - global_mean(self.annual_mean[self.reference])
            )
            for k in self.annual_mean
            if k!= self.reference
        }
        #8. Polar amplification
        # Defined as the ratio between polar warming and global warming.
        # Valueas greater than 1 indicate that the polar regions warm faster than
        # the global mean.
        self.polar_amp = {
            k: self.polar_warming[k]/self.global_warming[k]
            for k in self.global_warming
        }

        #9. Equilibrium Climate Sensitivity.
        # ECS is estimated from the relationship between global mean
        # temperature and log2(CO2 concentration).
        # The resulting slope represents the temperature response to one doubling of
        # atmospheric CO2.
        self.ECS = calculate_ecs(
            self.global_mean, self.co2_levels
        )

    def _plot(self, plot_experiment=None):
        """
        Generate figures for the TAS diagnostics.
        Some diagnostics show all experiments simultaneously, while others display a detailed
        map or latitude-time evolution for one selected experiment.
        """
        # Select experiment for single-experiment diagnostics.
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

        # Create output directory for this experiment's TAS plots
        plot_dir = self.plot_dirs[exp] / "tas"

        plot_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Mean annual TAS for one experiment
        plot_mean_map(
            field=self.annual_mean[exp],
            experiment=exp,
            title="Mean annual surface air temperature",
            levels=np.arange(-40, 51, 6),
            cmap=cmocean.cm.thermal,
            colorbar_label="Surface Air Temperature (°C)",
            save=plot_dir / f"tas_{exp}.png",
            )

        # Mean annual tas for the reference experiment and anomalies for the other experiments
        plot_reference_anomalies(
            reference=self.annual_mean[self.reference],
            anomalies=self.anomaly,
            reference_name=self.reference,
            absolute_title="Mean annual surface air temperature",
            anomaly_title="Surface air temperature anomaly",
            absolute_levels=np.arange(-40, 51, 6),
            anomaly_levels=np.arange(-40, 41, 1),
            absolute_cmap=cmocean.cm.thermal,
            anomaly_cmap=cmocean.cm.balance,
            absolute_label="Surface Air Temperature (°C)",
            anomaly_label="Temperature anomaly (°C)",
            save=plot_dir / "tas_maps.png",
            )

        # Zonal mean annual TAS for one experiment
        plot_zonal_mean(
            field=self.zonal_mean[exp],
            experiment=exp,
            title="Zonal mean surface air temperature",
            ylabel="Zonal mean TAS (°C)",
            save=plot_dir / f"tas_zonal_mean_{exp}.png"
            )

        # Zonal mean annual TAS anomalies
        plot_zonal_anomalies(
            anomalies=self.zonal_anom,
            title="Zonal-mean surface air temperature anomaly",
            ylabel="Zonal-mean TAS anomaly (°C)",
            save=plot_dir / "tas_zonal_anom.png"
        )

        # Zonal mean TAS time series for one experiment
        plot_zonal_time(
            field=self.zonal_ts[exp],
            experiment=exp,
            title="Zonal mean surface air temperature evolution",
            cmap=cmocean.cm.thermal,
            colorbar_label="Surface Air Temperature (°C)",
            levels=20,
            save=plot_dir / f"tas_zonal_time_{exp}.png"
        )


        # Global mean TAS vs CO2 for multiple experiments
        plot_global_mean_vs_co2(
            values=self.global_mean,
            co2_levels=self.co2_levels,
            ecs=self.ECS,
            polar_amp=self.polar_amp,
            ylabel="Global mean TAS (°C)",
            title = "Global mean TAS (°C) vs CO2 concentration",
            ylim=(10.5, 40),
            xlim=(0, 7),
            save=plot_dir / "global_mean_tas.png"
        )

        # Global mean TAS time series
        plot_global_timeseries(
            series=self.tas_ts,
            ylabel="Global mean TAS (°C)",
            title="Annual global mean TAS",
            save=plot_dir / "global_mean_tas_timeseries.png"
        )
