import xarray as xr
from pathlib import Path


from utils import global_mean_tas, plot_global_mean_vs_co2_dp, DEEP_MIP_MODEL_STYLES
from utils import global_mean_sst, regional_mean_sst, plot_sst_gradient_vs_global_sst_dp

class DeepMIPDiagnostics:
    """
    Compare model temperature diagnostics with DeepMIP multi-model data.
    DeepMIP (Deep-Time Model Intercomparison Project) provides Eocene simulations
    from several models at different CO2 concentrations. This class reads the
    DeepMIP surface air temperature (TAS) and sea surface temperature (SST) files,
    calculates global and regional means, and optionally adds the results of the
    user's own model (EC-Earth4) for comparison.

    Diagnostics include global mean TAS vs CO2 concentration, the DeepMIP TAS range
    across models for each CO2 level, and the meridional SST gradient vs global mean SST.

    Parameters
    base_path : str or pathlib.Path
        Base directory containing the DeepMIP data files.

    deepmip_models : dict[str, dict[str, dict[str, str]]]
        Dictionary mapping each DeepMIP model name to its data files, in the form
        {model: {"tas": {experiment: filename}, "sst": {experiment: filename}}}.
        Filenames are relative to base_path and are opened with xarray.open_mfdataset.

    plot_dir : str or pathlib.Path
        Base plot directory. Figures are saved in its "deepmip" subdirectory.

    reference : str
        Name of the DeepMIP reference experiment (e.g. "pi"), flagged in the plots.

    experiment_dirs : list[str]
        Names of the experiments being compared, joined to build the figure filenames.
    """

    def __init__(
        self,
        base_path,
        deepmip_models,
        plot_dir,
        reference,
        experiment_dirs
    ):
        """
        Initialize the DeepMIP diagnostics container.
        The constructor stores the input configuration and initializes empty containers
        that will later contain the calculated diagnostics.
        """

        # Input data and experiment configuration
        self.base_path = Path(base_path)
        self.deepmip_models = deepmip_models
        self.plot_dir = Path(plot_dir)
        self.reference = reference
        self.experiment_dirs = experiment_dirs

        # TAS diagnostics
        # Global mean TAS (°C) for each DeepMIP model and experiment
        self.results = {}

        # Plotting records for global mean TAS vs CO2
        self.data = []

        # (min, max) global mean TAS (°C) across DeepMIP models for each CO2 level
        self.tas_ranges = {}

        # SST diagnostics
        # Global, tropical and high-latitude mean SST and meridional SST gradient
        # for each DeepMIP model and experiment
        self.sst_results = {}

        # Plotting records for SST gradient vs global mean SST
        self.sst_data = []

        # DeepMIP experiment names mapped to CO2 concentration relative to the pre-industrial value
        self.co2_map = {
            "pi": 1,
            "1x": 1,
            "1.5x": 1.5,
            "2x": 2,
            "3x": 3,
            "4x": 4,
            "6x": 6,
            "9x": 9,
        }

    def run(self,
            plot=True,
            own_results=None,
            own_co2_levels=None,
            own_reference=None,
            own_sst_gradient=None,
            own_sst_results=None,
            save=True
    ):
        """
        Run all DeepMIP diagnostics.
        The method first calculates the DeepMIP diagnostics (optionally adding the user's
        own model results) and then, optionally, generates the corresponding figures.

        Parameters:
        plot : bool, default=True
            If True, generate diagnostic figures.

        own_results : dict[str, float], optional
            Global mean TAS (°C) of the user's own model for each experiment.
            If provided, own_co2_levels must also be provided.

        own_co2_levels : dict[str, float], optional
            Dictionary mapping the user's experiment names to their CO2 concentration
            relative to the pre-industrial value.

        own_reference : str, optional
            Reference experiment of the user's own model. If None, self.reference is used.

        own_sst_gradient : dict[str, float], optional
            Meridional SST gradient (tropical minus high-latitude mean SST) of the
            user's own model for each experiment. Required if own_sst_results is provided.

        own_sst_results : dict[str, float], optional
            Global mean SST of the user's own model for each experiment.
            If provided, own_sst_gradient and own_co2_levels must also be provided.

        save : bool, default=True
            If True, save the generated figures to disk. If False, figures are
            displayed but not saved.

        Returns:
        DeepMIPDiagnostics
            The diagnostics object containing all calculated results.
        """
        # Calculate all DeepMIP diagnostics
        self._calculate(own_results=own_results,
                        own_co2_levels=own_co2_levels,
                        own_reference=own_reference,
                        own_sst_gradient=own_sst_gradient,
                        own_sst_results=own_sst_results
        )

        # Generate figures if requested
        if plot:
            self._plot(save=save)

        return self

    def _calculate(self,
                   own_results=None,
                   own_co2_levels=None,
                   own_reference=None,
                   own_sst_gradient=None,
                   own_sst_results=None
    ):
        """
        Calculate all DeepMIP diagnostics.
        The calculations are performed in the following order:
        1. Calculate the global mean TAS for each DeepMIP model and experiment.
        2. Calculate the DeepMIP TAS range across models for each CO2 level.
        3. Calculate global, tropical and high-latitude mean SST and the meridional
           SST gradient for each DeepMIP model and experiment.
        4. Convert the DeepMIP results to plotting records.
        5. Optionally add the user's own model (EC-Earth4) TAS results.
        6. Optionally add the user's own model (EC-Earth4) SST results.

        Parameters:
        own_results : dict[str, float], optional
            Global mean TAS (°C) of the user's own model for each experiment.

        own_co2_levels : dict[str, float], optional
            Dictionary mapping the user's experiment names to their CO2 concentration
            relative to the pre-industrial value. Required if own_results or
            own_sst_results is provided.

        own_reference : str, optional
            Reference experiment of the user's own model. If None, self.reference is used.

        own_sst_gradient : dict[str, float], optional
            Meridional SST gradient of the user's own model for each experiment.
            Required if own_sst_results is provided.

        own_sst_results : dict[str, float], optional
            Global mean SST of the user's own model for each experiment.
        """

        # Default: use the DeepMIP reference experiment name for the user's own model
        if own_reference is None:
            own_reference=self.reference

        #1. DeepMIP global mean TAS (°C) for each model and experiment.
        self.results = {}

        for model, model_data in self.deepmip_models.items():

            self.results[model] = {}

            for exp, filename in model_data.get("tas", {}).items():

                path = self.base_path / filename

                ds = xr.open_mfdataset(
                    path,
                    combine="by_coords",
                    decode_times=False,
                )

                # Time-mean, area-weighted global mean TAS, converted to °C
                self.results[model][exp] = global_mean_tas(ds)

                ds.close()

        #2. DeepMIP TAS range (min, max) across models for each CO2 level.
        # Only models that provide the given experiment are included.
        self.tas_ranges= {}

        for exp in ["pi", "1x", "3x", "6x", "9x"]:

            values = [
                float(experiments[exp])
                for experiments in self.results.values()
                if exp in experiments
            ]

            if values:
                self.tas_ranges[exp] = (
                    min(values),
                    max(values)
                )

        #3. DeepMIP SST diagnostics for each model and experiment.
        self.sst_results = {}

        for model, model_data in self.deepmip_models.items():

            self.sst_results[model] = {}

            for exp, filename in model_data.get("sst", {}).items():

                path = self.base_path / filename

                ds = xr.open_mfdataset(
                    path,
                    combine="by_coords",
                    decode_times=False
                )

                # Global mean SST
                gm_sst = (
                    global_mean_sst(ds).compute().item()
                )

                # Tropical SST: 31°S–31°N
                trop_sst = (
                    regional_mean_sst(ds, -31, 31).compute().item()
                )

                # Northern high-latitude SST: 60°N–90°N
                north_sst = (
                    regional_mean_sst(ds, 60.01, 90.01).compute().item()
                )

                # Southern high-latitude SST: 90°S–60°S
                south_sst = (
                    regional_mean_sst(ds, -90.01, -60.01).compute().item()
                )

                # Mean high-latitude SST: average of the two hemispheres
                high_sst = 0.5 * (
                    north_sst + south_sst
                )

                # Meridional SST gradient: tropical minus high-latitude mean SST
                gradient = trop_sst - high_sst

                self.sst_results[model][exp]= {
                    "global": gm_sst,
                    "tropical": trop_sst,
                    "high": high_sst,
                    "gradient": gradient
                }

                ds.close()


        #4. Convert DeepMIP results to plotting records.
        # Experiments without a known CO2 level are skipped.
        # TAS records: global mean TAS vs CO2 concentration
        self.data = []

        for model, experiments in self.results.items():

            for exp, temp in experiments.items():

                if exp not in self.co2_map:
                    continue

                self.data.append({
                    "model": model,
                    "CO2": self.co2_map[exp],
                    "T": float(temp),
                    "pi": exp == self.reference
                })

        # SST records: meridional SST gradient (y) vs global mean SST (x)
        self.sst_data = []

        for model, experiments in self.sst_results.items():

            for exp, values in experiments.items():

                if exp not in self.co2_map:
                    continue

                self.sst_data.append({
                    "model": model,
                    "x": values["global"],
                    "y": values["gradient"],
                    "exp": exp,
                    "pi": exp == self.reference,
                })

        #5. Add the user's own model (EC-Earth4) TAS results, if provided.
        if own_results is not None:

            if own_co2_levels is None:
                raise ValueError(
                    "own_co2_levels must be provided when own_results is provided"
                )

            # Experiments without a CO2 level are skipped
            for exp, temp in own_results.items():

                if exp not in own_co2_levels:
                    continue

                self.data.append({
                    "model": "EC-EARTH4",
                    "CO2": own_co2_levels[exp],
                    "T": float(temp),
                    "pi": exp == own_reference,
                })

        #6. Add the user's own model (EC-Earth4) SST results, if provided.
        if own_sst_results is not None:

            if own_sst_gradient is None:
                raise ValueError(
                    "own_sst_gradient must be provided when own_sst_results is provided"
                )

            if own_co2_levels is None:
                raise ValueError(
                    "own_co2_levels must be provided when own_sst_results is provided"
                )

            # Experiments without a CO2 level or an SST gradient are skipped
            for exp, sst in own_sst_results.items():
                if exp not in own_co2_levels:
                    continue

                if exp not in own_sst_gradient:
                    continue

                self.sst_data.append({
                    "model": "EC-EARTH4",
                    "x": float(sst),
                    "y": float(own_sst_gradient[exp]),
                    "exp": exp,
                    "pi": exp == own_reference,

                })


    def _plot(self, save=True):
        """
        Generate figures for the DeepMIP diagnostics.
        Two figures are produced, each showing all DeepMIP models (and the user's own
        model, if added):
        - Global mean TAS (°C) vs CO2 concentration.
        - Meridional SST gradient vs global mean SST.
        If save is True, all figures are saved in the "deepmip" subdirectory of the plot directory.

        Parameters:
        save : bool, default=True
            If True, save the figures to disk. If False, figures are only displayed.
        """

        # Create output directory for the DeepMIP plots
        plot_dir = self.plot_dir / "deepmip"

        # Only create the directory if we are actually saving files
        if save:
            plot_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        # Filename suffix built from the compared experiment names
        experiment_string = "_".join(
            self.experiment_dirs
        )

        save_tas = (
            plot_dir
            / f"deepmip_global_mean_tas_{experiment_string}.png"
        )

        # Global mean TAS vs CO2 concentration for all models
        plot_global_mean_vs_co2_dp(
            data=self.data,
            title=(
                "Global mean surface temperature vs CO₂ concentration"
            ),
            save=save_tas if save else None,
            model_styles=DEEP_MIP_MODEL_STYLES
        )


        save_sst= (
            plot_dir
            / f"deepmip_sst_gradient_{experiment_string}.png"
        )

        # Meridional SST gradient vs global mean SST for all models
        plot_sst_gradient_vs_global_sst_dp(
            data=self.sst_data,
            title=(
                "Meridional SST gradient vs global mean SST"
            ),
            save=save_sst if save else None,
            model_styles=DEEP_MIP_MODEL_STYLES
        )



