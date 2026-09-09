import numpy as np
import cmocean
from utils import global_mean, zonal_mean 
from utils import plot_reference_anomalies, plot_mean_map 

class SOSDiagnostics:
    def __init__(self, oce, reference, plot_dirs):

        self.oce = oce
        self.reference = reference
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

    def run(self, plot=True, plot_experiment=None):

        self._calculate()
        
        if plot:
            self._plot(plot_experiment=plot_experiment)

        return self
    
    def _calculate(self):
             
        #Define SST
        self.sos = {
            k: ds["sos"]
            for k, ds in self.oce.items()
        }
             
        #Mean TAS over analysis period
        self.annual_mean = {
            k: v.mean("time_counter")
            for k, v in self.sos.items()
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
            sos_pi_2d = (
                self.pi_zonal
                .broadcast_like(self.annual_mean[k])
                .where(self.annual_mean[k].notnull())
            )

            #SST anomaly
            self.anomaly[k] = (
                self.annual_mean[k] - sos_pi_2d
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
            for k, v in self.sos.items()
        }

        #Global mean
        self.global_mean = {
            k: global_mean(v).mean("time_counter")
            for k, v in self.sos.items()
        }

        #Global mean time series
        self.ts = {
            k: global_mean(v)
            for k, v in self.sos.items()
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

        # Directory for this experiment's Salinity plots
        plot_dir = self.plot_dirs[exp] / "sos"

        plot_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Mean annual Salinity for one experiment
        plot_mean_map(
            field=self.annual_mean[exp],
            experiment=exp,
            title="Mean annual salinity",
            levels=np.arange(30.5, 36.5, 0.5),
            cmap=cmocean.cm.haline,
            colorbar_label="Salinity (1e-3)",
            save=plot_dir / f"sos_{exp}.png",
            )

        # Mean annual tas for the reference experiment and anomalies for the other experiments
        plot_reference_anomalies(
            reference=self.annual_mean[self.reference],
            anomalies=self.anomaly,
            reference_name=self.reference,
            absolute_title="Mean annual salinity",
            anomaly_title="Salinity anomaly",
            absolute_levels=np.arange(30.5, 36.5, 0.5),
            anomaly_levels=np.arange(-5, 5.5, 0.5),
            absolute_cmap=cmocean.cm.haline,
            anomaly_cmap=cmocean.cm.balance,
            absolute_label="Salinity (1e-3)",
            anomaly_label="Salinity anomaly (1e-3)",
            save=plot_dir / "sos_maps.png",
            )
