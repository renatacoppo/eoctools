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

    comparison_plot_dir : pathlib.Path
        Base directory used to store plots comparing multiple experiments.
    """
    def __init__(self, atm, reference, plot_dirs, comparison_plot_dir, experiment_labels):
        """
        Initialize the TOA diagnostics container.
        The input datasets and plotting configuration are stored, while
        empty dictionaries are initialized to hold the calculated temperature
        and radiation diagnostics.
        """

        # Input data and configuration
        self.atm = atm
        self.reference = reference
        self.plot_dirs = plot_dirs
        self.comparison_plot_dir = comparison_plot_dir

        # Original experiment names used in figures and legends
        self.experiment_labels = experiment_labels

        # Diagnostic data containers
        # Annual global mean surface air temperature time series (°C)
        self.toa = {}

        # Annual global mean net TOA radiation time series (W m-2)
        self.tas = {}

        # Annual global mean net surface energy flux time series (W m-2)
        self.sfc = {}

        # Difference between net TOA and net surface radiation (W m-2)
        self.toa_sfc_difference = {}

    def run(self, plot=True, plot_experiment=None):
        """
        Run all TOA and surface energy diagnostics.
        The method first calculates global mean temperature and radiation time series
        for all experiments. Diagnostic figures are then generated if requested.

        Parameters:
        plot: bool, optional
            If True, generate diagnostic figures.

        plot_experiment: str, optional
            Experiment used for selecting an experiment-specific plot directory. If Noone, the reference experiment is used.

        Returns:
        TOADiagnostics
            The diagnostics object containing all calculated results.
        """

        # Calculate all temperature and radiation diagnostics
        self._calculate()
        
        # Generate figures if requested
        if plot:
            self._plot(plot_experiment=plot_experiment)

        return self
    
    def _calculate(self):
        """
        Calculate global mean temperature and energy-budget diagnostics.
        """
        # Global mean surface air temperature
        # TAS is converted from Kelvin to degrees Celsius before calculating the latitude-area-weighted global mean.
        self.tas = {
            k: global_mean(ds["tas"] - 273.15)
            for k, ds in self.atm.items()
        }
        
        # Net top-of-the-atmosphere radiation
        # Positive values indicate a net gain of energy by the climate system
        self.toa = {
            k: global_toa_ts(ds)
            for k, ds in self.atm.items()
        }

        # Net surface energy flux
        # Positive values indicate a net downward energy flux into the surface
        self.sfc = {
            k: global_surface_net_radiation(ds)
            for k, ds in self.atm.items()
        }

        # TOA - SFC radiation imbalance
        # This quantity represents the difference between the radiative energy entering the climate system at the 
        # top of the atmosphere and the net energy flux at the surface.
        self.toa_sfc_difference = {
            k: self.toa[k] - self.sfc[k]
            for k in self.atm
        }

    def _plot(self, plot_experiment=None):
        """
        Generate TOA and surface energy-budget diagnostic plots.
        Parameters:
        plot_experiment : str, optional
            Experiment used to select the default output contect. If None, the reference experiment is used.
        """

        # Select thhe reference experiment by default
        # Default: plot the reference experiment
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

        plot_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Create filename from experiment directories
        # The experiment directory names are included so that comparison figures remain identifiable when multiple simulation 
        # sets are analysed

        experiment_dirs = [
            self.plot_dirs[exp].parent.name
            for exp in self.atm.keys()
        ]

        experiment_string = "_".join(experiment_dirs)

        # Gregory plot for all experiments (the years plotted depend on the years loaded on atm)
        save_gregory = (
            plot_dir
            / f"gregory_all_experiments_{experiment_string}.png"
        )

        # Plot the relationship between global mean TAS and net TOA radiation
        # Each point represents one year. The fitted linear relationship can be used to investigate the radiative
        # response of the climate system to surface warming.
        # The number of years included depends on the atmospheric datasets loaded into self.atm
        plot_gregory(
            tas_series=self.tas,
            toa_series=self.toa,
            experiment_labels=self.experiment_labels,
            title="Gregory plots – All experiments",
            save=save_gregory
        )

        # TOA - SFC radiation balance
        save_balance = (
            plot_dir
            / f"toa_sfc_radiations_{experiment_string}.png"
        )

        # Compare net TOA radiation and net surface energy flux through time toghether with their difference.
        plot_radiation_balance(
            toa=self.toa,
            sfc=self.sfc,
            difference=self.toa_sfc_difference,
            experiment_labels=self.experiment_labels,
            title="TOA - SFC radiation balance",
            save=save_balance
        )