import numpy as np
import cmocean
from utils import global_mean, zonal_mean, polar_mean 
from utils import plot_reference_anomalies, plot_mean_map, calculate_ecs, plot_zonal_mean 
from utils import plot_zonal_anomalies, plot_zonal_time, plot_global_mean_vs_co2, plot_global_timeseries

class TASDiagnostics:
    def __init__(self, atm, reference, co2_levels, plot_dirs):

        self.atm = atm
        self.reference = reference
        self.co2_levels = co2_levels
        self.plot_dirs = plot_dirs

        self.tas = {}
        self.annual_mean = {}
        self.anomaly = {}

        self.zonal_mean = {}
        self.zonal_anom = {}
        self.zonal_ts = {}

        self.global_mean = {}
        self.tas_ts = {}

        self.polar_warming = {}
        self.global_warming = {}
        self.polar_amp = {}

        self.ECS = None

    def run(self, plot=True, plot_experiment=None):

        self._calculate()
        
        if plot:
            self._plot(plot_experiment=plot_experiment)

        return self
    
    def _calculate(self):
             
        #Define TAS
        self.tas = {
            k: ds["tas"] - 273.15
            for k, ds in self.atm.items()
        }
             
        #Mean TAS over analysis period
        self.annual_mean = {
            k: v.mean("time_counter")
            for k, v in self.tas.items()
        }

        #Spatial anomalies relative to reference
        self.anomaly = {
            k: self.annual_mean[k] - self.annual_mean[self.reference]
                for k in self.annual_mean
                if k != self.reference
        }

        #Zonal mean
        self.zonal_mean = {
            k: zonal_mean(v)
            for k, v in self.annual_mean.items()
        }

        #Zonal anomalies
        self.zonal_anom = {
            k: self.zonal_mean[k] - self.zonal_mean[self.reference]
            for k in self.zonal_mean
            if k != self.reference
        }

        #Zonal time series
        self.zonal_ts = {
            k: zonal_mean(v)
            for k, v in self.tas.items()
        }

        #Global mean
        self.global_mean = {
            k: global_mean(v).mean("time_counter")
            for k, v in self.tas.items()
        }

        #Global mean time series
        self.tas_ts = {
            k: global_mean(v)
            for k, v in self.tas.items()
        }

        #Polar amplification
        self.polar_warming = {
            k: (
                polar_mean(self.annual_mean[k]) 
                - polar_mean(self.annual_mean[self.reference])
            )
            for k in self.annual_mean
            if k!= self.reference
        }

        self.global_warming = {
            k: (
                global_mean(self.annual_mean[k])
                - global_mean(self.annual_mean[self.reference])
            )
            for k in self.annual_mean
            if k!= self.reference
        }

        self.polar_amp = {
            k: self.polar_warming[k]/self.global_warming[k]
            for k in self.global_warming
        }

        #Equilibrium Climate Sensitivity
        self.ECS = calculate_ecs(
            self.global_mean, self.co2_levels
        )

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

        plot_global_timeseries(
            series=self.tas_ts,
            ylabel="Global mean TAS (°C)",
            title="Annual global mean TAS",
            save=plot_dir / "global_mean_tas_timeseries.png"
        )
