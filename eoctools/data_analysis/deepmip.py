import xarray as xr
from pathlib import Path


from utils import global_mean_tas, plot_global_mean_vs_co2_dp, DEEP_MIP_MODEL_STYLES
from utils import global_mean_sst, regional_mean_sst, plot_sst_gradient_vs_global_sst_dpn

class DeepMIPDiagnostics:
    """
    Compare model TAS diagnostics with DeepMIP model data.
    """

    def __init__(
        self,
        base_path,
        deepmip_models,
        plot_dir, 
        reference,
        experiment_dirs
    ):

        self.base_path = Path(base_path)
        self.deepmip_models = deepmip_models
        self.plot_dir = Path(plot_dir)
        self.reference = reference
        self.experiment_dirs = experiment_dirs

        self.results = {}
        self.data = []

        self.sst_results = {}
        self.sst_data = []

        #DeepMIP models
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
            own_sst_results=None
    ):

        self._calculate(own_results=own_results, 
                        own_co2_levels=own_co2_levels,
                        own_reference=own_reference,
                        own_sst_gradient=own_sst_gradient,
                        own_sst_results=own_sst_results
        )

        if plot:
            self._plot()

        return self

    def _calculate(self,
                   own_results=None,
                   own_co2_levels=None,
                   own_reference=None,
                   own_sst_gradient=None,
                   own_sst_results=None
    ):

        if own_reference is None:
            own_reference=self.reference
        
        # DeepMIP TAS
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

                self.results[model][exp] = global_mean_tas(ds)

                ds.close()

        # DeepMIP SST
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

                #Global mean SST
                gm_sst = (
                    global_mean_sst(ds).compute().item()
                )

                # Tropical SST: 31°S–31°N
                trop_sst = (
                    regional_mean_sst(ds, -31, 31).compute().item()
                )

                #Northern high latitude SST
                north_sst = (
                    regional_mean_sst(ds, 60.01, 90.01).compute().item()
                )

                #Southern high latitude SST
                south_sst = (
                    regional_mean_sst(ds, -90.01, -60.01).compute().item()
                )

                #Mean high-latitude SST
                high_sst = 0.5 * (
                    north_sst + south_sst
                )

                #Meridional SST gradient
                gradient = trop_sst - high_sst

                self.sst_results[model][exp]= {
                    "global": gm_sst,
                    "tropical": trop_sst,
                    "high": high_sst,
                    "gradient": gradient
                }

                ds.close()


        # Convert DeepMIP results to plotting data
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

        # EC-Earth4 model
        if own_results is not None:

            if own_co2_levels is None:
                raise ValueError(
                    "own_co2_levels must be provided whe"
                    "own_results is provided"
                )

            for exp, temp in own_results.items():

                if exp not in own_co2_levels:
                    continue

                self.data.append({
                    "model": "EC-EARTH4",
                    "CO2": own_co2_levels[exp],
                    "T": float(temp),
                    "pi": exp == own_reference,
                })

        if own_sst_results is not None:

            if own_sst_gradient is None:
                raise ValueError(
                    "own_sst_gradient must be provided when own_sst_results is provided"
                )
            
            if own_co2_levels is None:
                raise ValueError(
                    "own_co2_levels must be provided when own_sst_results is provided"
                )
            
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


    def _plot(self):

        # Output directory
        plot_dir = self.plot_dir / "deepmip"

        plot_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Filename
        experiment_string = "_".join(
            self.experiment_dirs
        )

        save_tas = (
            plot_dir
            / f"deepmip_global_mean_tas_{experiment_string}.png"
        )

        # Plot
        plot_global_mean_vs_co2_dp(
            data=self.data,
            title=(
                "Global mean surface temperature vs CO₂ concentration"
            ),
            save=save_tas,
            model_styles=DEEP_MIP_MODEL_STYLES
        )
    

        save_sst= (
            plot_dir
            / f"deepmip_sst_gradient_{experiment_string}.png"
        )

        plot_sst_gradient_vs_global_sst_dpn(
            data=self.sst_data,
            title=(
                "Meridional SST gradient vs global mean SST"
            ),
            save=save_sst,
            model_styles=DEEP_MIP_MODEL_STYLES
        )

    
    