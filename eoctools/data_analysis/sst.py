import numpy as np
import cmocean
from utils import global_mean, zonal_mean 
from utils import high_lat_mean_sst, trop_mean_sst 
from utils import plot_reference_anomalies, plot_mean_map, plot_zonal_mean 
from utils import plot_zonal_anomalies, plot_global_mean_vs_co2, plot_sst_gradient_vs_global_mean

class SSTDiagnostics:
    """
    Calculate and plot sea surface temperature (SST) diagnostics.
    The class operates on ocean datasets from multiple experiments.
    SST is read from the 'tos' variable and used as stored in the datasets
    (plots label it in degrees Celsius, °C); no unit conversion is applied.

    Diagnostics include spatial means, anomalies relative to the reference zonal mean,
    zonal means, global means, SST time series, tropical and high-latitude mean SSTs,
    and the meridional SST gradient.

    Parameters
    oce : dict[str, xarray.Dataset]
        Dictionary containing ocean datasets (with a 'tos' variable) for each experiment.

    co2_levels : dict[str, float]
        Dictionary mapping experiment names to their atmospheric CO2 concentration relative to the pre-industrial value.

    reference : str
        Name of the reference experiment used to calculate anomalies.

    plot_dirs : dict[str, pathlib.Path]
        Dictionary mapping experiment names to their base plot directories.
    """
    def __init__(self, oce, reference, co2_levels, plot_dirs):
        """
        Initialize the SST diagnostics container.
        The constructor stores the input datasets and initializes empty dictionaries that will later
        contain the calculated diagnostics.
        """

        # Input data and experiment configuration
        self.oce = oce
        self.reference = reference
        self.co2_levels = co2_levels
        self.plot_dirs = plot_dirs

        # Temperature fields
        # Full SST fields (°C)
        self.tos = {}

        # Time-mean SST field over the analysis period (°C)
        self.annual_mean = {}

        # Time-mean SST field of the reference experiment (°C)
        self.pi = {}

        # Ocean mask (True where SST is defined) of the reference experiment
        self.ocean_mask_pi = {}

        # Zonal mean SST profile of the reference experiment (°C)
        self.pi_zonal = {}

        # Spatial SST anomalies relative to the reference zonal mean (°C)
        self.anomaly = {}
        #self.pi_2d = {}

        # Zonal diagnostics
        # Time-mean zonal SST profiles
        self.zonal_mean = {}

        # Zonal SST anomalies relative to the reference
        self.zonal_anom = {}

        # Latitude-time evolution of zonal mean SST
        self.zonal_ts = {}

        # Global and regional diagnostics
        # Time-mean global SST (°C)
        self.global_mean = {}

        # Annual global mean SST time series
        self.ts = {}

        # Tropical mean SST (°C)
        self.trop = {}

        # High-latitude mean SST (°C)
        self.high = {}

        # Meridional SST gradient: tropical minus high-latitude SST (°C)
        self.dT = {}

    def run(self, plot=True, plot_experiment=None):
        """
        Run all SST diagnostics.
        The method first calculates all SST diagnostics and then, optionally, generates the corresponding figures.

        Parameters:
        plot : bool, default=True
            If True, generate diagnostic figures.

        plot_experiment : str, optional
            Experiment used for single-experiment plots (maps, zonal mean)
            and for selecting the plot directory.
            If None, the reference experiment is used.

        Returns:
        SSTDiagnostics
            The diagnostics object containing all calculated results.
        """
        # Calculate all SST diagnostics
        self._calculate()

        # Generate figures if requested
        if plot:
            self._plot(plot_experiment=plot_experiment)

        return self

    def _calculate(self):
        """
        Calculate all sea surface temperature diagnostics.
        The calculations are performed in the following order:
        1. Extract SST ('tos') from each dataset.
        2. Calculate mean SST fields over the analysis period.
        3. Calculate the zonal mean SST of the reference experiment over its ocean points.
        4. Calculate spatial anomalies relative to the reference zonal mean.
        5. Calculate zonal mean SST profiles.
        6. Calculate zonal mean anomalies relative to the reference experiment.
        7. Calculate the latitude-time evolution of zonal mean SST.
        8. Calculate global mean SST and its annual time series.
        9. Calculate tropical and high-latitude mean SST and the meridional SST gradient.
        """

        #1. Extract SST
        self.tos = {
            k: ds["tos"]
            for k, ds in self.oce.items()
        }

        #2. Mean SST over the analysis period
        self.annual_mean = {
            k: v.mean("time_counter")
            for k, v in self.tos.items()
        }

        #3. Reference zonal mean SST.
        # Reference annual mean SST
        self.pi = self.annual_mean[self.reference]

        # Ocean mask of the reference experiment
        self.ocean_mask_pi = self.pi.notnull()

        # Latitude weights (normalised cos(latitude)).
        # Note: they do not affect a mean taken over longitude only.
        weights = np.cos(np.deg2rad(self.pi.lat))
        weights = weights / weights.mean()

        # Zonal mean of reference SST, restricted to reference ocean points
        self.pi_zonal = (
            self.pi
            .where(self.ocean_mask_pi)
            .weighted(weights)
            .mean("lon")
        )

        #4. SST anomalies relative to the reference zonal mean.
        # Anomalies are taken against the reference zonal mean rather than the
        # reference 2D field. The reference is a pre-industrial simulation while
        # the other experiments are Eocene simulations with a different
        # continental configuration, so a point-by-point comparison of the 2D
        # fields would not be meaningful. Using the zonal mean also gives every
        # ocean point of an experiment a reference value, even where the
        # reference experiment has land.
        self.anomaly = {}
        for k in self.annual_mean:
            if k == self.reference:
                continue

            # Broadcast reference zonal mean to 2D, masked to this experiment's ocean points
            tos_pi_2d = (
                self.pi_zonal
                .broadcast_like(self.annual_mean[k])
                .where(self.annual_mean[k].notnull())
            )

            # SST anomaly (°C)
            self.anomaly[k] = (
                self.annual_mean[k] - tos_pi_2d
            )

        #5. Zonal mean SST profiles
        self.zonal_mean = {
            k: zonal_mean(v)
            for k, v in self.annual_mean.items()
        }

        #6. Zonal SST anomalies relative to the reference experiment
        self.zonal_anom = {
            k: self.zonal_mean[k] - self.zonal_mean[self.reference]
            for k in self.zonal_mean
            if k != self.reference
        }

        #7. Latitude-time evolution.
        # Zonal mean for every year, preserving the time dimension.
        self.zonal_ts = {
            k: zonal_mean(v)
            for k, v in self.tos.items()
        }

        #8. Global mean SST.
        # First calculate the area-weighted global mean for each year, then
        # average over the analysis period.
        self.global_mean = {
            k: global_mean(v).mean("time_counter")
            for k, v in self.tos.items()
        }

        # Annual global mean SST time series
        self.ts = {
            k: global_mean(v)
            for k, v in self.tos.items()
        }

        #9. Tropical SST, high-latitude SST and the difference between them.
        # Tropical mean SST (31°S-31°N), stored as a numpy value
        self.trop = {
            k: trop_mean_sst(v).values
            for k, v in self.annual_mean.items()
        }

        # High-latitude mean SST (average of 60°N-90°N and 90°S-60°S), stored as a numpy value
        self.high = {
            k: high_lat_mean_sst(v).values
            for k, v in self.annual_mean.items()
        }

        # Meridional SST gradient: tropics minus high latitudes (°C)
        self.dT = {
            k: self.trop[k] - self.high[k] 
            for k in self.annual_mean.keys()
        } 

    def _plot(self, plot_experiment=None):
        """
        Generate figures for the SST diagnostics.
        Some diagnostics show all experiments simultaneously, while others display a detailed
        map or zonal profile for one selected experiment.
        All figures are saved in the "sst" subdirectory of the selected experiment's plot directory.

        Parameters:
        plot_experiment : str, optional
            Experiment used for single-experiment plots. If None, the
            reference experiment is used.
        """
        # Select experiment for single-experiment diagnostics.
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

        # Create output directory for this experiment's SST plots
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

        # Mean annual SST for the reference experiment and anomalies for the other experiments
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

        # Zonal mean annual SST anomalies for all non-reference experiments
        plot_zonal_anomalies(
            anomalies=self.zonal_anom,
            title="Zonal-mean sea surface temperature anomaly",
            ylabel="Zonal-mean SST anomaly (°C)",
            save=plot_dir / "sst_zonal_anom.png"
        )

        # Global mean SST vs CO2 for all experiments
        plot_global_mean_vs_co2(
            values=self.global_mean,
            co2_levels=self.co2_levels,
            ylabel="Global mean SST (°C)",
            title = "Global mean SST (°C) vs CO2 concentration",
            ylim=(10.5, 40),
            xlim=(0, 7),
            save=plot_dir / "global_mean_sst.png"
        )

        # Meridional SST gradient (tropics minus high latitudes) vs global mean SST for all experiments
        plot_sst_gradient_vs_global_mean(
            global_mean_sst=self.global_mean,
            meridional_gradient=self.dT,
            reference=self.reference,
            xlabel="Global mean SST (°C)",
            ylabel="Tropics – High latitude SST (°C)",
            title="Meridional SST gradient vs global mean SST",
            save=plot_dir / "sst_gradient_vs_global_mean.png"
        )