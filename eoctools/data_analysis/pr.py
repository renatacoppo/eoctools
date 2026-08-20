import numpy as np
import cmocean
from utils import prepare_pr, global_mean, zonal_mean
from utils import plot_reference_anomalies, plot_mean_map, plot_zonal_mean 
from utils import plot_zonal_anomalies, plot_zonal_time
from utils import plot_global_mean_vs_logco2

class PRDiagnostics:
    def __init__(self, atm, reference, co2_levels_log2, plot_dirs):

        self.atm = atm
        self.reference = reference
        self.co2_levels_log2 = co2_levels_log2
        self.plot_dirs = plot_dirs

        self.pr = {}
        self.annual_mean = {}
        self.anomaly = {}

        self.pi_zonal = {}
        self.pi_2d = {}
        self.anom = {}
        self.anom_zonal = {}

        self.zonal_mean = {}
        self.zonal_anom = {}
        self.zonal_ts = {}

        self.global_mean = {}
        self.ts = {}



    def run(self, plot=True, plot_experiment=None):

        self._calculate()
        
        if plot:
            self._plot(plot_experiment=plot_experiment)

        return self
    
    def _calculate(self):
             
        #Define Pr
        self.pr = {
            k: prepare_pr(ds["pr"])
            for k, ds in self.atm.items()
        }
             
        #Mean Pr over analysis period
        self.annual_mean = {
            k: v.mean("time_counter")
            for k, v in self.pr.items()
        }

        #Spatial anomalies relative to reference
        self.anomaly = {
            k: self.annual_mean[k] - self.annual_mean[self.reference]
                for k in self.annual_mean
                if k != self.reference
        }

        #Reference zonal mean
        self.pi_zonal = zonal_mean(self.pr[self.reference])

        #Time-dependent anomalies relative to reference
        pi_2d = {
            k: self.pi_zonal.broadcast_like(v) 
            for k, v in self.pr.items() 
            if k != self.reference
        }
        self.anom = {
            k: self.pr[k] - pi_2d[k] 
            for k in pi_2d
        }
        self.anom_zonal = {
            k: v.mean("lon") 
            for k, v in self.anom.items()
        }

        #Zonal mean of annual mean Pr
        self.zonal_mean = {
            k: zonal_mean(v)
            for k, v in self.annual_mean.items()
        }

        #Zonal anomalies relative to reference
        self.zonal_anom = {
            k: self.zonal_mean[k] - self.zonal_mean[self.reference]
            for k in self.zonal_mean
            if k != self.reference
        }

        #Global mean
        self.global_mean = {
            k: global_mean(v).mean("time_counter")
            for k, v in self.pr.items()
        }

        #Zonal mean time series
        self.zonal_ts = {
        k: zonal_mean(v)
        for k, v in self.pr.items()
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

        # Directory for this experiment's pr plots
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

        # Mean annual pr for the reference experiment and anomalies for the other experiments
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

        # Zonal mean pr time series for one experiment
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
            ylim=(2.5, 4.4),
            xlim=(-0.1, 3),
            save=plot_dir / "global_mean_pr_logco2.png"
        )
