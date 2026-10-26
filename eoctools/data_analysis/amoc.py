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
    Calculate and plot Atlantic meridional overturning circulation (AMOC) diagnostics.
    The class operates on ocean MOC datasets and atmospheric datasets from multiple experiments.

    Diagnostics include the time-mean AMOC streamfunction, the annual maximum AMOC strength
    time series, and the relationship between AMOC strength and global mean surface air
    temperature (TAS). ``msftyz`` is already in Sv in the simulation output, so no unit
    conversion is applied. All input data (MOC and atmosphere) are annual, so one time
    step corresponds to one year.

    Parameters
    moc : dict[str, xarray.Dataset]
        Dictionary containing MOC datasets for each experiment, used for the time-mean AMOC
        structure. The datasets are expected to contain the ``msftyz`` meridional overturning
        streamfunction with a ``basin`` dimension.

    moc_full : dict[str, xarray.Dataset]
        Dictionary containing the full-length MOC datasets for each experiment, used for the
        AMOC strength time series.

    atm : dict[str, xarray.Dataset]
        Dictionary containing atmospheric datasets for each experiment. The datasets are
        expected to contain ``tas`` in Kelvin.

    reference : str
        Name of the reference experiment.

    plot_dirs : dict[str, pathlib.Path]
        Dictionary mapping experiment names to their base plot directories.

    comparison_plot_dir : pathlib.Path
        Directory for figures that compare multiple experiments.

    skip_years : int, default=0
        Number of initial time steps (e.g. spin-up years) removed from the AMOC time series
        and from the TAS time series.

    colors : dict[str, str], optional
        Dictionary mapping experiment names to plot colors.

    experiment_labels : dict[str, str], optional
        Dictionary mapping experiment names to display labels used in the plots.
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
        Initialize the AMOC diagnostics container.
        The constructor stores the input datasets and plotting options and initializes
        an empty dictionary that will later contain the TAS time series.
        """

        # Input data
        self.moc = moc
        self.moc_full = moc_full
        self.atm = atm

        # Experiment configuration and output directories
        self.reference = reference
        self.plot_dirs = plot_dirs
        self.comparison_plot_dir = comparison_plot_dir
        self.skip_years = skip_years

        # Plot styling; fall back to empty mappings if not provided
        self.colors = colors or {}
        self.experiment_labels = experiment_labels or {}

        # Global mean surface air temperature time series (°C), filled by _calculate_tas
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
        Run all AMOC diagnostics.
        The method calculates the global mean TAS, then for each experiment the time-mean
        AMOC structure and the annual maximum AMOC strength time series, and finally,
        optionally, generates the corresponding figures.

        Parameters:
        nyears : int, optional
            Number of final time steps (years) averaged for the mean AMOC structure.
            If None, all available time steps are used.

        lat_range : tuple of float, default=(38, 50)
            Latitude range (degrees north) in which the AMOC maximum is searched.

        depth_range : tuple of float, default=(500, 2000)
            Depth range (m) in which the AMOC maximum is searched.

        basin : int, default=1
            Index of the Atlantic basin in the ``basin`` dimension of ``msftyz``.

        moving_mean : int, optional
            Window length (years) of the moving mean added to the AMOC time series plot.
            If None, no moving mean is plotted.

        f_years : int, optional
            Number of initial years shown in the AMOC-TAS plot.
            If None, all years are shown.

        plot : bool, default=True
            If True, generate diagnostic figures.

        save : bool, default=True
            If True, save the generated figures to disk.

        Returns:
        dict[str, dict]
            Dictionary mapping each experiment name to a dictionary with keys
            "mean" (time-mean AMOC streamfunction) and "timeseries"
            (annual maximum AMOC strength).
        """

        # Calculate TAS using the same spin-up selection
        # as the AMOC time series
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
            # (uses the full-length dataset, moc_full)
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

        # ----------------------------------------------------------
        # Plot (all experiments at once, after the loop)
        # ----------------------------------------------------------

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
        The calculation is performed in the following order:
        1. Select the Atlantic basin from the overturning streamfunction.
        2. Optionally keep only the final nyears time steps.
        3. Average over time.

        Note: skip_years is not applied here; nyears selects from the end of the record.

        Parameters:
        ds : xarray.Dataset
            MOC dataset containing ``msftyz``.

        nyears : int, optional
            Number of final time steps (years) to average. If None, all time steps are used.

        basin : int, default=1
            Index of the Atlantic basin in the ``basin`` dimension.

        Returns:
        xarray.DataArray
            Time-mean Atlantic meridional overturning streamfunction (depth-latitude).
        """

        #1. Atlantic basin streamfunction
        amoc = ds["msftyz"].sel(basin=basin)

        #2. Keep only the final nyears time steps
        if nyears is not None:

            amoc = amoc.isel(
                time_counter=slice(-nyears, None)
            )

        #3. Time mean (squeeze drops leftover singleton dimensions)
        return amoc.mean("time_counter").squeeze()

    def _calculate_amoc_timeseries(
        self,
        ds,
        lat_range=(38, 50),
        depth_range=(500, 2000),
        basin=1,
    ):
        """
        Calculate the annual maximum AMOC strength time series.
        The calculation is performed in the following order:
        1. Select the Atlantic basin and restrict the streamfunction to the depth range.
        2. Mask values outside the latitude range.
        3. Remove the first skip_years time steps.
        4. Resample to annual means.
        5. Take the maximum over all spatial dimensions for each year.

        Parameters:
        ds : xarray.Dataset
            MOC dataset containing ``msftyz``.

        lat_range : tuple of float, default=(38, 50)
            Latitude range (degrees north); bounds are exclusive.

        depth_range : tuple of float, default=(500, 2000)
            Depth range (m) on the ``depthw`` coordinate.

        basin : int, default=1
            Index of the Atlantic basin in the ``basin`` dimension.

        Returns:
        xarray.DataArray
            Annual maximum AMOC strength as a function of time.
        """

        #1. Atlantic basin, restricted to the depth range
        amoc = ds["msftyz"].sel(
            basin=basin,
            depthw=slice(*depth_range),
        )

        #2. Mask latitudes outside the range (nav_lat is the 2D model latitude)
        amoc = amoc.where(
            (amoc.nav_lat > lat_range[0])
            & (amoc.nav_lat < lat_range[1])
        )

        #3. Remove the initial (spin-up) time steps
        # The input data are annual, so skipping time steps skips whole years.
        if self.skip_years > 0:
            amoc = amoc.isel(
                time_counter=slice(self.skip_years, None)
            )

        #4. Annual means (yearly bins starting on 1 January)
        amoc = amoc.resample(
            time_counter="YS"
        ).mean()

        #5. Maximum over all remaining spatial dimensions, ignoring masked (NaN) points
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
        Calculate the global mean surface air temperature time series.
        The same initial years removed from the AMOC time series are
        removed here so that TAS and AMOC cover the same simulation years.
        The result, in degrees Celsius (°C), is stored in self.tas.

        The calculation is performed in the following order:
        1. Remove the first skip_years time steps.
        2. Convert TAS from Kelvin to degrees Celsius and calculate the
           area-weighted global mean for each time step.
        """

        #1. Remove the initial (spin-up) time steps
        # TAS is stored as annual values, so no annual resampling is needed and
        # the series lines up year by year with the annual AMOC time series.
        atm = {
            exp: ds.isel(
                time_counter=slice(self.skip_years, None)
            )
            for exp, ds in self.atm.items()
        }

        #2. Global mean TAS in degrees Celsius
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
        Generate figures for the AMOC diagnostics of all experiments.
        The mean AMOC structure is plotted separately for each experiment and saved in
        the "amoc" subdirectory of that experiment's plot directory. The AMOC time series
        and the AMOC-TAS plot combine all experiments and are saved in the "amoc"
        subdirectory of the comparison plot directory.

        Parameters:
        results : dict[str, dict]
            Output of run(): for each experiment, a dictionary with keys "mean"
            and "timeseries".

        f_years : int, optional
            Number of initial years shown in the AMOC-TAS plot.
            If None, all years are shown.

        moving_mean : int, optional
            Window length (years) of the moving mean added to the AMOC time series plot.
            If None, no moving mean is plotted.

        save : bool, default=True
            If True, save the figures to disk.
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
        # sets are analysed.

        experiment_dirs = [
            self.plot_dirs[exp].parent.name
            for exp in self.moc.keys()
        ]

        experiment_string = "_".join(experiment_dirs)

        # Tag filenames with the analysed period (after spin-up removal or all years)
        if self.skip_years > 0:
            years_string = f"after_skip{self.skip_years}"
        else:
            years_string = "all_years"
        
        # Mean AMOC streamfunction plot for every experiment
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

        # Annual maximum AMOC strength time series for all experiments in one figure
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

        # AMOC-TAS "Gregory-style" plot: annual maximum AMOC strength against
        # global mean TAS (°C) for all experiments
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
