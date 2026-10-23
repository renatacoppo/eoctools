import numpy as np
import cmocean
from utils import global_mean, zonal_mean 
from utils import plot_reference_anomalies, plot_mean_map 

class SOSDiagnostics:
    """
    Calculate and plot sea surface salinity (SOS) diagnostics.
    The class operates on ocean datasets from multiple experiments.
    Salinity is used in the model's native units (1e-3, i.e. psu / g kg-1).

    Diagnostics include time-mean maps, anomalies relative to the zonal mean of the
    reference experiment, zonal means, global means and global mean time series.

    Parameters
    oce : dict[str, xarray.Dataset]
        Dictionary containing ocean datasets (with a "sos" variable) for each experiment.

    reference : str
        Name of the reference experiment used to calculate anomalies.

    plot_dirs : dict[str, pathlib.Path]
        Dictionary mapping experiment names to their base plot directories.
    """
    def __init__(self, oce, reference, plot_dirs):
        """
        Initialize the SOS diagnostics container.
        The constructor stores the input datasets and initializes empty dictionaries that will later
        contain the calculated diagnostics.
        """

        # Input data and experiment configuration
        self.oce = oce
        self.reference = reference
        self.plot_dirs = plot_dirs

        # Salinity fields
        # Full SOS fields (note: _calculate stores them in self.sos, not self.tos)
        self.tos = {}

        # Time-mean SOS field over the analysis period (psu)
        self.annual_mean = {}

        # Time-mean SOS field of the reference experiment (psu)
        self.pi = {}

        # Ocean mask (True over ocean points) of the reference experiment
        self.ocean_mask_pi = {}

        # Zonal mean SOS profile of the reference experiment (psu)
        self.pi_zonal = {}

        # SOS anomalies relative to the reference zonal mean (psu)
        self.anomaly = {}
        #self.pi_2d = {}

        # Zonal diagnostics
        # Time-mean zonal salinity profiles
        self.zonal_mean = {}

        # Zonal salinity anomalies relative to the reference
        self.zonal_anom = {}

        # Latitude-time evolution of zonal mean SOS
        self.zonal_ts = {}

    def run(self, plot=True, plot_experiment=None, save=True):
        """
        Run all SOS diagnostics.
        The method first calculates all salinity diagnostics and then, optionally, generates the corresponding figures.

        Parameters:
        plot : bool, default=True
            If True, generate diagnostic figures.

        plot_experiment : str, optional
            Experiment used for single-experiment plots (mean map) and for
            selecting the plot directory. If None, the reference experiment is used.

        save : bool, default=True
            If True, save the generated figures to disk. If False, figures are
            displayed but not saved.

        Returns:
        SOSDiagnostics
            The diagnostics object containing all calculated results.
        """
        # Calculate all SOS diagnostics
        self._calculate()

        # Generate figures if requested
        if plot:
            self._plot(plot_experiment=plot_experiment, save=save)

        return self
    
    def _calculate(self):
        """
        Calculate all sea surface salinity diagnostics.
        The calculations are performed in the following order:
        1. Extract the SOS field of each experiment.
        2. Calculate mean salinity fields over the analysis period.
        3. Select the time-mean salinity of the reference experiment and its ocean mask.
        4. Calculate the cos(latitude)-weighted zonal mean of the reference salinity.
        5. Calculate spatial anomalies relative to the reference zonal mean.
        6. Calculate zonal mean salinity profiles.
        7. Calculate zonal mean anomalies relative to the reference experiment.
        8. Calculate the latitude-time evolution of zonal mean salinity.
        9. Calculate global mean salinity and its annual time series.
        """

        #1. Extract SOS (psu) from each experiment
        self.sos = {
            k: ds["sos"]
            for k, ds in self.oce.items()
        }

        #2. Mean SOS over the analysis period
        self.annual_mean = {
            k: v.mean("time_counter")
            for k, v in self.sos.items()
        }

        #3. Reference time-mean SOS
        self.pi = self.annual_mean[self.reference]

        # Ocean mask of the reference experiment (land points are NaN)
        self.ocean_mask_pi = self.pi.notnull()

        #4. Zonal mean of the reference SOS
        # Latitude weights (cos(lat)), normalised to a mean of 1
        weights = np.cos(np.deg2rad(self.pi.lat))
        weights = weights / weights.mean()

        # Weighted mean over longitude, restricted to ocean points
        self.pi_zonal = (
            self.pi
            .where(self.ocean_mask_pi)
            .weighted(weights)
            .mean("lon")
        )

        #5. SOS anomalies relative to the reference zonal mean.
        # Each non-reference experiment is compared with the reference zonal mean
        # profile, not with the full 2D reference field. The reference is a
        # pre-industrial simulation while the other experiments are Eocene
        # simulations with a different continental configuration, so a
        # point-by-point comparison of the 2D fields would not be meaningful.
        self.anomaly = {}
        for k in self.annual_mean:
            if k == self.reference:
                continue

            # Broadcast the reference zonal mean to 2D, masked to this experiment's ocean points
            sos_pi_2d = (
                self.pi_zonal
                .broadcast_like(self.annual_mean[k])
                .where(self.annual_mean[k].notnull())
            )

            # SOS anomaly (psu)
            self.anomaly[k] = (
                self.annual_mean[k] - sos_pi_2d
            )

        #6. Zonal mean SOS profiles
        self.zonal_mean = {
            k: zonal_mean(v)
            for k, v in self.annual_mean.items()
        }

        #7. Zonal SOS anomalies relative to the reference experiment
        self.zonal_anom = {
            k: self.zonal_mean[k] - self.zonal_mean[self.reference]
            for k in self.zonal_mean
            if k != self.reference
        }

        #8. Latitude-time evolution.
        # Zonal mean for every time step, preserving the time dimension.
        self.zonal_ts = {
            k: zonal_mean(v)
            for k, v in self.sos.items()
        }

        #9. Global mean SOS.
        # First calculate the global mean for each time step, then
        # average over the analysis period.
        self.global_mean = {
            k: global_mean(v).mean("time_counter")
            for k, v in self.sos.items()
        }

        # Global mean SOS time series
        self.ts = {
            k: global_mean(v)
            for k, v in self.sos.items()
        }


    def _plot(self, plot_experiment=None, save=True):
        """
        Generate figures for the SOS diagnostics.
        Produces a mean salinity map for one selected experiment and a panel with the
        reference mean salinity and the anomalies of the other experiments.
        If save is True, all figures are saved in the "sos" subdirectory of the selected experiment's plot directory.

        Parameters:
        plot_experiment : str, optional
            Experiment used for single-experiment plots. If None, the
            reference experiment is used.

        save : bool, default=True
            If True, save the figures to disk. If False, figures are only displayed.
        """
        # Select experiment for single-experiment diagnostics.
        # Default: plot the reference experiment.
        if plot_experiment is None:
            plot_experiment = self.reference

        exp = plot_experiment

        # Check that experiment exists
        if exp not in self.oce:
            raise ValueError(
                f"Unknown experiment '{exp}'. "
                f"Available experiments: {list(self.oce.keys())}"
            )

        # Create output directory for this experiment's SOS plots
        plot_dir = self.plot_dirs[exp] / "sos"

        # Only create the directory if we are actually saving files
        if save:
            plot_dir.mkdir(
                parents=True,
                exist_ok=True
            )

        # Mean annual SOS for one experiment
        plot_mean_map(
            field=self.annual_mean[exp],
            experiment=exp,
            title="Mean annual salinity",
            levels=np.arange(30.5, 36.5, 0.5),
            cmap=cmocean.cm.haline,
            colorbar_label="Salinity (1e-3)",
            save=plot_dir / f"sos_{exp}.png" if save else None,
            )

        # Mean annual SOS for the reference experiment and anomalies (relative to the
        # reference zonal mean) for the other experiments
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
            save=plot_dir / "sos_maps.png" if save else None,
            )
