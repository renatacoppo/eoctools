from utils import plot_gregory, global_toa_ts, global_mean, global_surface_net_radiation, plot_radiation_balance

class TOADiagnostics:
    """
    Calculate and plot top-of-atmosphere and surface energy diagnostics.
    The class operates on atmospheric datasets from multiple climate experiments and calculates global
    mean annual time series suitable for energy-balance analysis and Gregory regression plots.

    Parameters
    atm : dict[str, xarray.Dataset]
        Dictionary containing atmospheric datasets for each experiment.

    reference : str
        Name of the reference experiment.

    plot_dirs : dict[str, pathlib.Path]
        Dictionary mapping experiment names to their plot directories.
        The parent directory names are used to build comparison figure filenames.

    comparison_plot_dir : pathlib.Path
        Base directory used to store plots comparing multiple experiments.

    experiment_labels : dict[str, str]
        Dictionary mapping experiment names to the labels shown in figures and legends.

    colors : dict[str, str]
        Dictionary mapping experiment names to their plot colors.

    deepmip_tas_ranges : dict, optional
        DeepMIP global mean surface air temperature ranges (°C) shown in the Gregory plot.
        If None, an empty dictionary is used.

    skip_years : int, default=0
        Number of initial simulation years to discard before calculating the diagnostics.

    warm_fix_activations : dict, optional
        Warm-fix activation information passed to the TOA - SFC radiation balance plot.
        If None, an empty dictionary is used.
    """
    def __init__(self, 
                 atm, 
                 reference, 
                 plot_dirs, 
                 comparison_plot_dir, 
                 experiment_labels, 
                 colors, 
                 deepmip_tas_ranges=None,
                 skip_years=0,
                 warm_fix_activations=None):
        """
        Initialize the TOA diagnostics container.
        The input datasets and plotting configuration are stored, while
        empty dictionaries are initialized to hold the calculated temperature
        and radiation diagnostics. Optional dictionaries default to empty
        dictionaries when None is passed.
        """

        # Input data and configuration
        self.atm = atm
        self.reference = reference
        self.plot_dirs = plot_dirs
        self.comparison_plot_dir = comparison_plot_dir
        self.colors = colors
        self.skip_years = skip_years
        self.warm_fix_activations = (
            warm_fix_activations
            if warm_fix_activations is not None
            else {}
        )
        self.deepmip_tas_ranges = (
            deepmip_tas_ranges
            if deepmip_tas_ranges is not None
            else {}
        )

        # Original experiment names used in figures and legends
        self.experiment_labels = experiment_labels

        # Diagnostic data containers
        # Annual global mean surface air temperature time series (°C)
        self.tas = {}

        # Annual global mean net TOA radiation time series (W m-2)
        self.toa = {}

        # Annual global mean net surface energy flux time series (W m-2)
        self.sfc = {}

        # Difference between net TOA and net surface radiation (W m-2)
        self.toa_sfc_difference = {}

    def run(self, 
            f_years=None, 
            skip_years=None, 
            moving_mean=None,
            plot=True, 
            save=True, 
            plot_experiment=None):
        """
        Run all TOA and surface energy diagnostics.
        The method first calculates global mean temperature and radiation time series
        for all experiments. Diagnostic figures are then generated if requested.

        Parameters:
        f_years : int, optional
            Number of simulation years (after the skipped years) shown in the
            figures. It is used only by the plotting functions and in the figure
            titles and filenames. If None, all available years are used.

        skip_years : int, optional
            Number of initial simulation years to discard before calculating the
            diagnostics. If None, the value given at initialization is used.

        moving_mean : int, optional
            Moving-mean window passed to the TOA - SFC radiation balance plot.
            If None, no moving mean is requested.

        plot : bool, default=True
            If True, generate diagnostic figures.

        save : bool, default=True
            If True, save generated figures to disk. If False, figures are
            displayed but not saved.

        plot_experiment : str, optional
            Experiment checked for existence before plotting. If None, the
            reference experiment is used. Figures are always stored in the
            common comparison directory.

        Returns:
        TOADiagnostics
            The diagnostics object containing all calculated results.
        """

        # Fall back on the number of skipped years given at initialization
        if skip_years is None:
            skip_years = self.skip_years

        # Calculate all temperature and radiation diagnostics
        self._calculate(skip_years=skip_years)
        
        # Generate figures if requested
        if plot:
            self._plot(f_years=f_years,
                       skip_years=skip_years,
                       moving_mean=moving_mean,
                       save=save,
                       plot_experiment=plot_experiment)

        return self
    
    def _calculate(self, f_years=None, skip_years=0):
        """
        Calculate global mean temperature and energy-budget diagnostics.
        The calculations are performed in the following order:
        1. Discard the first skip_years years of each simulation.
        2. Calculate the annual global mean surface air temperature (°C).
        3. Calculate the annual global mean net TOA radiation (W m-2).
        4. Calculate the annual global mean net surface energy flux (W m-2).
        5. Calculate the difference between net TOA and net surface fluxes (W m-2).

        Parameters:
        f_years : int, optional
            Currently unused: all years after the skipped ones are included
            in the calculations.

        skip_years : int, default=0
            Number of initial simulation years to discard.
        """

        #1. Discard the first skip_years years from each simulation

        atm = {
            exp: ds.isel(
                time_counter=slice(skip_years, None)
            )
            for exp, ds in self.atm.items()
        }

        # Store the selected datasets if useful for inspection
        self.atm_selected = atm
        
        #2. Global mean surface air temperature (°C)
        # TAS is converted from Kelvin to degrees Celsius before calculating the area-weighted global mean.
        self.tas = {
            k: global_mean(ds["tas"] - 273.15)
            for k, ds in atm.items()
        }
        
        #3. Net top-of-the-atmosphere radiation (W m-2)
        # Positive values indicate a net gain of energy by the climate system
        self.toa = {
            k: global_toa_ts(ds)
            for k, ds in atm.items()
        }

        #4. Net surface energy flux (W m-2)
        # Positive values indicate a net downward energy flux into the surface
        self.sfc = {
            k: global_surface_net_radiation(ds)
            for k, ds in atm.items()
        }

        #5. TOA - SFC energy flux difference (W m-2)
        # This quantity represents the difference between the radiative energy entering the climate system at the
        # top of the atmosphere and the net energy flux at the surface, i.e. the net energy gained by the atmosphere.
        self.toa_sfc_difference = {
            k: self.toa[k] - self.sfc[k]
            for k in atm
        }

    def _plot(self, 
              f_years=None, 
              skip_years=0, 
              moving_mean=None,
              save=True, 
              plot_experiment=None):
        """
        Generate TOA and surface energy-budget diagnostic plots.
        Two comparison figures are produced for all experiments: a Gregory plot
        (net TOA radiation vs global mean TAS) and the time series of the
        TOA - SFC energy flux difference. Figures are stored in the "toa"
        subdirectory of the comparison plot directory.

        Parameters:
        f_years : int, optional
            Number of simulation years (after the skipped years) shown in the
            figures. If None, all available years are shown.

        skip_years : int, default=0
            Number of initial years discarded in the calculations, used in the
            output filenames.

        moving_mean : int, optional
            Moving-mean window passed to the TOA - SFC radiation balance plot.

        save : bool, default=True
            If True, save figures to disk.

        plot_experiment : str, optional
            Experiment checked for existence before plotting. If None, the
            reference experiment is used.
        """

        # Select the reference experiment by default
        if plot_experiment is None:
            plot_experiment = self.reference

        exp = plot_experiment

        # Check that the requested experiment exists
        if exp not in self.atm:
            raise ValueError(
                f"Unknown experiment '{exp}'. "
                f"Available experiments: {list(self.atm.keys())}"
            )

        # Create output directory
        # These diagnostics compare multiple experiments, so figures are stored in the common comparison directory
        # rather than inside an individual experiment directory.
        plot_dir = self.comparison_plot_dir / "toa"

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
            for exp in self.atm.keys()
        ]

        experiment_string = "_".join(experiment_dirs)

        # Describe the analysed period for filenames and titles
        if f_years is None:
            years_string = f"after_skip_{skip_years}"
            period_title = (
                f"after first {skip_years} years"
                if skip_years > 0
                else "all years"
            )
        else:
            years_string = (
                f"after_skip{skip_years}_first_{f_years}_years"
            )
            period_title = (
                f"first {f_years} years after skipping the first {skip_years} years"
            )

        # Gregory plot for all experiments (the years plotted depend on the years loaded in atm)
        save_gregory = (
            plot_dir
            / f"gregory_{years_string}_{experiment_string}.png"
        )

        # Plot the relationship between global mean TAS and net TOA radiation
        # Each point represents one year. The fitted linear relationship can be used to investigate the radiative
        # response of the climate system to surface warming.
        # The number of years included depends on the atmospheric datasets loaded into self.atm.
        # DeepMIP TAS ranges (°C) are passed for comparison with the model results.
        plot_gregory(
            tas_series=self.tas,
            toa_series=self.toa,
            experiment_labels=self.experiment_labels,
            colors=self.colors,
            f_years=f_years,
            deepmip_tas_ranges=self.deepmip_tas_ranges,
            title=f"Gregory plots – first {f_years} years"
                if f_years is not None
                else "Gregory plots - all years",
            save=save_gregory if save else None
        )

        # TOA - SFC radiation balance
        save_balance = (
            plot_dir
            / f"toa_sfc_energy_flux_{years_string}_{experiment_string}.png"
        )

        # Plot the time series of the difference between net TOA radiation and net surface energy flux (W m-2),
        # optionally smoothed with a moving mean and marking the warm-fix activations.
        plot_radiation_balance(
            difference=self.toa_sfc_difference,
            experiment_labels=self.experiment_labels,
            f_years=f_years,
            moving_mean=moving_mean,
            colors=self.colors,
            warm_fix_activations=self.warm_fix_activations,
            title="TOA - SFC radiation balance",
            save=save_balance if save else None
        )