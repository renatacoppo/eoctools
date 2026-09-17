import numpy as np
import matplotlib.pyplot as plt

from utils import (
    global_mean,
    plot_moc_mean,
    plot_moc_timeseries,
    plot_amoc_gregory
)

class AMOCDiagnostics:
    """
    Diagnostics for the meridional overturning circulation (MOC).

    Parameters
    ----------
    moc : dict[str, xarray.Dataset]
        MOC datasets for each experiment. The datasets are expected to
        contain the ``msftyz`` meridional overturning streamfunction.

    reference : str
        Name of the reference experiment.

    plot_dirs : dict[str, pathlib.Path]
        Output directory for each experiment's plots.
    """

    def __init__(
        self,
        moc,
        moc_full,
        atm,
        reference,
        plot_dirs,
        comparison_plot_dir,
        skip_years=0,
        colors=None,
        experiment_labels=None
    ):
        """
        Initialize MOC diagnostics.
        """

        self.moc = moc
        self.moc_full = moc_full
        self.atm = atm

        self.reference = reference
        self.plot_dirs = plot_dirs
        self.comparison_plot_dir = comparison_plot_dir
        self.skip_years = skip_years

        self.colors = colors or {}
        self.experiment_labels = experiment_labels or {}

        # Annual global mean surface air temperature
        self.tas = {}

    # ==============================================================
    # Main interface
    # ==============================================================

    def run(
        self,
        nyears=None,
        lat_range=(38, 50),
        depth_range=(500, 2000),
        basin=1,
        moving_mean=None,
        f_years=None,
        plot=True,
        save=True,
    ):
        """
        Run MOC diagnostics.

        Parameters
        ----------
        nyears : int, optional
            Number of final years to use for the mean AMOC structure.
            If ``None``, all available years are used.

        lat_range : tuple of float, optional
            Latitude range used to calculate the AMOC maximum.

        depth_range : tuple of float, optional
            Depth range used to calculate the AMOC maximum.

        basin : int, optional
            Basin index used for the Atlantic basin.

        plot : bool, optional
            Whether to generate plots.

        save : bool, optional
            Whether to save the generated figures.

        Returns
        -------
        dict
            Dictionary containing calculated AMOC diagnostics.
        """

        # Calculate TAS using the same spin-up selection
        # as the AMOC time series.
        self._calculate_tas()

        results = {}

        for exp in self.moc:

            # ------------------------------------------------------
            # Calculate mean AMOC structure
            # ------------------------------------------------------

            amoc_mean = self._calculate_mean_amoc(
                self.moc[exp],
                nyears=nyears,
                basin=basin,
            )

            # ------------------------------------------------------
            # Calculate AMOC strength time series
            # ------------------------------------------------------

            amoc_timeseries = self._calculate_amoc_timeseries(
                self.moc_full[exp],
                lat_range=lat_range,
                depth_range=depth_range,
                basin=basin,
            )

            results[exp] = {
                "mean": amoc_mean,
                "timeseries": amoc_timeseries,
            }

            # ------------------------------------------------------
            # Plot
            # ------------------------------------------------------

        if plot:

            self._plot(
                results,
                f_years=f_years,
                moving_mean=moving_mean,
                save=save,
            )

        return results

    # ==============================================================
    # Calculations
    # ==============================================================

    def _calculate_mean_amoc(
        self,
        ds,
        nyears=None,
        basin=1,
    ):
        """
        Calculate the time-mean AMOC streamfunction.

        Parameters
        ----------
        ds : xarray.Dataset
            MOC dataset.

        nyears : int, optional
            Number of final years to average.

        basin : int
            Basin index.

        Returns
        -------
        xarray.DataArray
            Time-mean Atlantic meridional overturning streamfunction.
        """

        amoc = ds["msftyz"].sel(basin=basin)

        if nyears is not None:

            amoc = amoc.isel(
                time_counter=slice(-nyears, None)
            )

        return amoc.mean("time_counter").squeeze()

    def _calculate_amoc_timeseries(
        self,
        ds,
        lat_range=(38, 50),
        depth_range=(500, 2000),
        basin=1,
    ):
        """
        Calculate the annual maximum AMOC strength.

        The AMOC is restricted to the specified latitude and depth
        ranges before calculating the maximum over depth and latitude.

        Parameters
        ----------
        ds : xarray.Dataset
            MOC dataset.

        lat_range : tuple of float
            Latitude range.

        depth_range : tuple of float
            Depth range.

        basin : int
            Basin index.

        Returns
        -------
        xarray.DataArray
            Annual maximum AMOC strength.
        """

        amoc = ds["msftyz"].sel(
            basin=basin,
            depthw=slice(*depth_range),
        )

        amoc = amoc.where(
            (amoc.nav_lat > lat_range[0])
            & (amoc.nav_lat < lat_range[1])
        )

        if self.skip_years > 0:
            amoc = amoc.isel(
                time_counter=slice(self.skip_years, None)
            )

        amoc = amoc.resample(
            time_counter="YS"
        ).mean()

        # Reduce all spatial dimensions
        spatial_dims = [
            dim for dim in amoc.dims
            if dim != "time_counter"
        ]

        amoc_max = amoc.max(
            dim=spatial_dims,
            skipna=True
        ). squeeze()

        return amoc_max
    
    def _calculate_tas(self):
        """
        Calculate annual global mean surface air temperature.

        The same initial years removed from the AMOC time series are
        removed here to ensure that TAS and AMOC represent the same
        simulation years.
        """

        atm = {
            exp: ds.isel(
                time_counter=slice(self.skip_years, None)
            )
            for exp, ds in self.atm.items()
        }

        self.tas = {
            exp: global_mean(
                ds["tas"] - 273.15
            )
            for exp, ds in atm.items()
        }

    # ==============================================================
    # Plotting
    # ==============================================================

    def _plot(
        self,
        results,
        f_years=None,
        moving_mean=None,
        save=True,
    ):
        """
        Generate MOC plots for one experiment.
        
        The mean AMOC structure is plotted separately for each
        experiment, while all AMOC time series are combined into
        one figure.
        """

        # Create output directory
        # These diagnostics compare multiple experiments, so figures are stored in the common comparison directory
        # rather than inside an individual experiment directory.
        plot_dir = self.comparison_plot_dir / "amoc"

        # Only create the directory if we are actually saving files
        if save:
            plot_dir.mkdir(
                parents=True,
                exist_ok=True
            )


        # Create filename from experiment directories
        # The experiment directory names are included so that comparison figures remain identifiable when multiple simulation 
        # sets are analysed

        experiment_dirs = [
            self.plot_dirs[exp].parent.name
            for exp in self.moc.keys()
        ]

        experiment_string = "_".join(experiment_dirs)

        if self.skip_years > 0:
            years_string = f"after_skip{self.skip_years}"
        else:
            years_string = "all_years"
        
        # AMOC mean plots for every experiment
        for exp, data in results.items():

            experiment_plot_dir = self.plot_dirs[exp] / "amoc"

            # Only create the directory if we are actually saving files
            if save:
                experiment_plot_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )
                
                save_moc_mean = (
                    experiment_plot_dir
                    / f"moc_mean_{exp}_{years_string}.png"
                )
            else:
                save_moc_mean = None

            plot_moc_mean(
                data["mean"],
                experiment=exp,
                experiment_label=self.experiment_labels.get(exp,exp),
                save_path=save_moc_mean if save else None,
                title=f"AMOC mean - {exp}"
            )

        # AMOC time series
        amoc_timeseries = {
            exp: data ["timeseries"]
            for exp, data in results.items()
        }

        if save:
            save_moc_timeseries = (
                plot_dir
                / f"moc_timeseries_{years_string}_{experiment_string}.png"
            )
        else:
            save_moc_timeseries = None

        plot_moc_timeseries(
            amoc_timeseries,
            colors=self.colors,
            experiment_labels=self.experiment_labels,
            plot_dir=self.comparison_plot_dir,
            moving_mean=moving_mean,
            title=f"Annual maximum AMOC strength",
            save_path=save_moc_timeseries if save else None
        )

        # AMOC-TAS Gregory plot
        tas_series = {
        exp: self.tas[exp]
        for exp in results
        }

        if save:
            save_amoc_gregory = (
                plot_dir
                / f"amoc_tas_gregory_{years_string}_{experiment_string}.png"
            )
        else:
            save_amoc_gregory = None

        plot_amoc_gregory(
            tas_series=tas_series,
            amoc_series=amoc_timeseries,
            experiment_labels=self.experiment_labels,
            colors=self.colors,
            f_years=f_years,
            title=(
                "AMOC–TAS relationship"
                if f_years is None
                else f"AMOC–TAS relationship – first {f_years} years"
            ),
            save=save_amoc_gregory,
        )
