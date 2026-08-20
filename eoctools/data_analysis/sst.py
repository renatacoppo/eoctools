import numpy as np
import cmocean
from utils import global_mean, zonal_mean 
from utils import high_lat_mean_sst, trop_mean_sst 
from utils import plot_reference_anomalies, plot_mean_map, plot_zonal_mean 
from utils import plot_zonal_anomalies, plot_global_mean_vs_co2, plot_sst_gradient_vs_global_mean

class SSTDiagnostics:
    def __init__(self, oce, co2_levels, reference, plot_dirs):

        self.oce = oce
        self.reference = reference
        self.co2_levels = co2_levels
        self.plot_dirs = plot_dirs

        self.tos = {}
        self.annual_mean = {}
        self.pi = {}
        self.ocean_mask_pi = {}
        self.pi_zonal = {}
        self.anomaly = {}
        #self.pi_2d = {}

        self.zonal_mean = {}
        self.zonal_anom = {}
        self.zonal_ts = {}

        self.global_mean = {}
        self.ts = {}
        self.trop = {}
        self.high = {}
        self.dT = {}

    def run(self, plot=True, plot_experiment=None):

        self._calculate()
        
        if plot:
            self._plot(plot_experiment=plot_experiment)

        return self
    
    def _calculate(self):
             
        #Define SST
        self.tos = {
            k: ds["tos"]
            for k, ds in self.oce.items()
        }
             
        #Mean TAS over analysis period
        self.annual_mean = {
            k: v.mean("time_counter")
            for k, v in self.tos.items()
        }

        #Reference annual mean SST
        self.pi = self.annual_mean[self.reference]

        #Ocean mask of reference experiment
        self.ocean_mask_pi = self.pi.notnull()

        #Latitude wheights
        weights = np.cos(np.deg2rad(self.pi.lat))
        weights = weights / weights.mean()

        #Zonal meann of reference SST
        self.pi_zonal = (
            self.pi
            .where(self.ocean_mask_pi)
            .weighted(weights)
            .mean("lon")
        )

        #SST anomalies relative to reference zonal mean
        self.anomaly = {}
        for k in self.annual_mean:
            if k == self.reference:
                continue
                    
            #Broadcast reference zonal mean to 2D
            tos_pi_2d = (
                self.pi_zonal
                .broadcast_like(self.annual_mean[k])
                .where(self.annual_mean[k].notnull())
            )

            #SST anomaly
            self.anomaly[k] = (
                self.annual_mean[k] - tos_pi_2d
            )

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
            for k, v in self.tos.items()
        }

        #Global mean
        self.global_mean = {
            k: global_mean(v).mean("time_counter")
            for k, v in self.tos.items()
        }

        #Global mean time series
        self.ts = {
            k: global_mean(v)
            for k, v in self.tos.items()
        }

        # Computing tropics SSTs, high latitudes SSTs and the difference between both
        #Tropics
        self.trop = {
            k: trop_mean_sst(v).values 
            for k, v in self.annual_mean.items()
        } 
        
        #High latitudes
        self.high = {
            k: high_lat_mean_sst(v).values 
            for k, v in self.annual_mean.items()
        } 
        
        #Meridional SST gradient
        self.dT = {
            k: self.trop[k] - self.high[k] 
            for k in self.annual_mean.keys()
        } 

    def _plot(self, plot_experiment=None):

        # Default: plot the reference experiment
        if plot_experiment is None:
            plot_experiment = self.reference

        exp = plot_experiment

        # Check that experiment exists
        if exp not in self.oce:
            raise ValueError(
                f"Unknown experiment '{exp}'. "
                f"Available experiments: {list(self.oce.keys())}"
            )

        # Directory for this experiment's SST plots
        plot_dir = self.plot_dirs[exp] / "sst"

        plot_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Mean annual SST for one experiment
        plot_mean_map(
            field=self.annual_mean[exp],
            experiment=exp,
            title="Mean annual sea surface temperature",
            levels=np.arange(-40, 51, 6),
            cmap=cmocean.cm.thermal,
            colorbar_label="Sea Surface Temperature (°C)",
            save=plot_dir / f"sst_{exp}.png",
            )

        # Mean annual tas for the reference experiment and anomalies for the other experiments
        plot_reference_anomalies(
            reference=self.annual_mean[self.reference],
            anomalies=self.anomaly,
            reference_name=self.reference,
            absolute_title="Mean annual sea surface temperature",
            anomaly_title="Sea surface temperature anomaly",
            absolute_levels=np.arange(-5, 41, 6),
            anomaly_levels=np.arange(-30, 30.5, 2),
            absolute_cmap=cmocean.cm.thermal,
            anomaly_cmap=cmocean.cm.balance,
            absolute_label="Sea surface Temperature (°C)",
            anomaly_label="Temperature anomaly (°C)",
            save=plot_dir / "sst_maps.png",
            )

        # Zonal mean annual SST for one experiment
        plot_zonal_mean(
            field=self.zonal_mean[exp],
            experiment=exp,
            title="Zonal mean sea surface temperature",
            ylabel="Zonal mean SST (°C)",
            save=plot_dir / f"sst_zonal_mean_{exp}.png"
            )

        # Zonal mean annual SST anomalies
        plot_zonal_anomalies(
            anomalies=self.zonal_anom,
            title="Zonal-mean sea surface temperature anomaly",
            ylabel="Zonal-mean SST anomaly (°C)",
            save=plot_dir / "sst_zonal_anom.png"
        )

        # Global mean TAS vs CO2 for multiple experiments
        plot_global_mean_vs_co2(
            values=self.global_mean,
            co2_levels=self.co2_levels,
            ylabel="Global mean SST (°C)",
            title = "Global mean SST (°C) vs CO2 concentration",
            ylim=(10.5, 36),
            xlim=(0, 7),
            save=plot_dir / "global_mean_sst.png"
        )

        plot_sst_gradient_vs_global_mean(
            global_mean_sst=self.global_mean,
            meridional_gradient=self.dT,
            reference=self.reference,
            xlabel="Global mean SST (°C)",
            ylabel="Tropics – High latitude SST (°C)",
            title="Meridional SST gradient vs global mean SST",
            save=plot_dir / "sst_gradient_vs_global_mean.png"
        )