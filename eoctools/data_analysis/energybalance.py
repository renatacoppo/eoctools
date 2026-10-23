#==================================================#
# ------- ENERGY BALANCE ANALYSIS (EBA) ---------- #
#==================================================#

# Zonal mean energy balance analysis following Heinemann et al. (2009), Lunt et al. (2012)
# and Lunt et al. (2021, Clim. Past, 17, 203-227; Eqs. 1-3 and Fig. 4).
#
# At each latitude, the zonal mean surface temperature (tau), planetary albedo (alpha_p),
# emissivity (eps), incoming TOA solar radiation (S) and meridional heat flux convergence (H)
# are related by:
#
#     S (1 - alpha_p) + H = eps * sigma * tau^4                                         (1)
#
# so that tau = [(S (1 - alpha_p) + H) / (eps * sigma)]^(1/4). The warming of an experiment (E)
# relative to a reference (P) is then decomposed into the contributions of each term:
#
#     dtau_eps     = tau(alpha_p^P, eps^E, H^P) - tau(alpha_p^P, eps^P, H^P)
#     dtau_H       = tau(alpha_p^P, eps^P, H^E) - tau(alpha_p^P, eps^P, H^P)            (2)
#     dtau_alpha_p = tau(alpha_p^E, eps^P, H^P) - tau(alpha_p^P, eps^P, H^P)
#
# and the planetary albedo term is split into a surface (alpha_s) and non-surface term:
#
#     dtau_alpha_s  = tau(alpha_p^P + (alpha_s^E - alpha_s^P), eps^P, H^P) - tau(alpha_p^P, eps^P, H^P)
#     dtau_alpha_ns = dtau_alpha_p - dtau_alpha_s                                        (3)

#==================================================#
# ------------------ IMPORTS --------------------- #
#==================================================#

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from utils import get_lat_lon_names

# Stefan-Boltzmann constant (W m-2 K-4)
SIGMA = 5.670374419e-8

# Terms of the decomposition, in the order and style of Lunt et al. (2021), Fig. 4
EBA_TERMS = {
    "dT_gcm":      {"label": r"$\Delta T$ GCM",          "color": "black", "linestyle": "--"},
    "dT_eba":      {"label": r"$\Delta\tau$",             "color": "black", "linestyle": "-"},
    "dT_alpha_p":  {"label": r"$\Delta\tau_{\alpha_p}$",  "color": "blue",  "linestyle": "-"},
    "dT_alpha_s":  {"label": r"$\Delta\tau_{\alpha_s}$",  "color": "blue",  "linestyle": ":"},
    "dT_alpha_ns": {"label": r"$\Delta\tau_{\alpha_{ns}}$", "color": "blue", "linestyle": "--"},
    "dT_eps":      {"label": r"$\Delta\tau_{\epsilon}$",  "color": "cyan",  "linestyle": "-"},
    "dT_H":        {"label": r"$\Delta\tau_{H}$",         "color": "green", "linestyle": "-"},
    "dT_sum":      {"label": "sum",                       "color": "red",   "linestyle": "-"},
}


class EnergyBalanceDiagnostics:
    """
    Calculate and plot the zonal mean energy balance analysis of Lunt et al. (2021).
    The warming of each experiment relative to a reference experiment is decomposed into the
    contributions from changes in planetary albedo (split into surface and non-surface albedo),
    emissivity and meridional heat flux.

    The datasets must contain 'rsdt', 'rsut' and 'rlut' (W m-2) and a surface temperature
    variable ('ts', or 'tas' as a fallback). The surface albedo split requires either 'alb'
    (surface albedo, 0-1) or 'rsds' and 'rsus' (W m-2); if none is available, only the planetary
    albedo, emissivity and heat flux terms are calculated.

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

    temperature_var : str, optional
        Name of the surface temperature variable. If None, 'ts' is used when available,
        otherwise 'tas'.
    """
    def __init__(self,
                 atm,
                 reference,
                 plot_dirs,
                 comparison_plot_dir,
                 experiment_labels,
                 temperature_var=None):
        """
        Initialize the energy balance diagnostics container.
        The input datasets and plotting configuration are stored, while empty dictionaries
        are initialized to hold the calculated zonal mean quantities and decompositions.
        """

        # Input data and configuration
        self.atm = atm
        self.reference = reference
        self.plot_dirs = plot_dirs
        self.comparison_plot_dir = comparison_plot_dir
        self.experiment_labels = experiment_labels
        self.temperature_var = temperature_var

        # Diagnostic data containers
        # Time-mean zonal mean energy balance quantities (S, alpha_p, alpha_s, eps, H, T) for each experiment
        self.zonal = {}

        # Zonal mean temperature change decomposition (°C) for each (experiment, reference) pair
        self.decomposition = {}

        # Global mean (GMT) and polar amplification (AMP) of each decomposition term (°C)
        self.summary = {}

    def run(self, pairs=None, plot=True, save=True):
        """
        Run the energy balance analysis.

        Parameters:
        pairs : list[tuple[str, str]], optional
            List of (experiment, reference) pairs to decompose, e.g. [("x4", "pi"), ("x4", "x1")].
            If None, every experiment is compared with the reference experiment.

        plot : bool, default=True
            If True, generate the diagnostic figure.

        save : bool, default=True
            If True, save the generated figure to disk. If False, the figure is
            displayed but not saved.

        Returns:
        EnergyBalanceDiagnostics
            The diagnostics object containing all calculated results.
        """

        # Compare every experiment with the reference by default
        if pairs is None:
            pairs = [
                (exp, self.reference)
                for exp in self.atm
                if exp != self.reference
            ]

        # Check that all requested experiments exist
        for pair in pairs:
            for exp in pair:
                if exp not in self.atm:
                    raise ValueError(
                        f"Unknown experiment '{exp}'. "
                        f"Available experiments: {list(self.atm.keys())}"
                    )

        self._calculate(pairs=pairs)

        # Generate figures if requested
        if plot:
            self._plot(pairs=pairs, save=save)

        return self

    def _calculate(self, pairs):
        """
        Calculate the energy balance decomposition.
        The calculations are performed in the following order:
        1. Calculate the time-mean zonal mean energy balance quantities of each experiment.
        2. Decompose the zonal mean temperature change of each pair (Eqs. 2 and 3).
        3. Calculate the global mean and polar amplification of each term.

        Parameters:
        pairs : list[tuple[str, str]]
            List of (experiment, reference) pairs to decompose.
        """

        #1. Zonal mean energy balance quantities
        experiments = {exp for pair in pairs for exp in pair}

        self.zonal = {
            exp: self._zonal_quantities(self.atm[exp])
            for exp in experiments
        }

        for exp, ref in pairs:
            name = f"{exp}-{ref}"

            #2. Temperature change decomposition (°C)
            self.decomposition[name] = decompose_warming(
                self.zonal[exp],
                self.zonal[ref]
            )

            #3. Global mean warming and polar amplification of each term (°C)
            self.summary[name] = {
                term: {
                    "GMT": float(zonal_profile_mean(da)),
                    "AMP": float(polar_amplification(da)),
                }
                for term, da in self.decomposition[name].data_vars.items()
            }

            self._print_summary(name)

    def _zonal_quantities(self, ds):
        """
        Calculate the time-mean zonal mean quantities entering Eq. (1).

        The fluxes are first averaged in time. Planetary albedo, emissivity and heat flux
        convergence are then calculated at each grid point and averaged zonally. Note that,
        as discussed in Lunt et al. (2021), the results are sensitive to the order of the
        averaging and the albedo/emissivity operators, especially at high latitudes.

        Parameters:
        ds : xarray.Dataset
            Dataset containing the TOA radiative fluxes and the surface temperature.

        Returns:
        xarray.Dataset
            Zonal mean S (W m-2), alpha_p (1), alpha_s (1, if available), eps (1), H (W m-2)
            and T (K).
        """

        # Time mean over the analysis period
        time_dims = [dim for dim in ds["rsdt"].dims if "time" in dim]
        ds = ds.mean(time_dims)

        lat_name, lon_name = get_lat_lon_names(ds["rsdt"])

        T = ds[self._get_temperature_var(ds)]

        # Incoming solar radiation, planetary albedo and outgoing longwave radiation
        S = ds["rsdt"]
        alpha_p = ds["rsut"] / ds["rsdt"]
        olr = ds["rlut"]

        # Emissivity: ratio of outgoing longwave radiation to surface blackbody emission
        eps = olr / (SIGMA * T**4)

        # Meridional heat flux convergence: in equilibrium, the energy leaving at the TOA
        # in excess of the absorbed solar radiation is supplied by the heat transport
        H = olr - S * (1 - alpha_p)

        zonal = xr.Dataset({
            "S": S,
            "alpha_p": alpha_p,
            "eps": eps,
            "H": H,
            "T": T,
        })

        # Surface albedo, needed only for the surface / non-surface albedo split
        alpha_s = self._get_surface_albedo(ds)
        if alpha_s is not None:
            zonal["alpha_s"] = alpha_s

        return zonal.mean(lon_name)

    def _get_temperature_var(self, ds):
        """
        Return the name of the surface temperature variable.
        The variable given at initialization is used if provided; otherwise 'ts' is
        preferred (as in Lunt et al., 2021) and 'tas' is used as a fallback.
        """
        if self.temperature_var is not None:
            return self.temperature_var

        if "ts" in ds:
            return "ts"

        if "tas" in ds:
            print("Energy balance: 'ts' not found, using 'tas' as surface temperature.")
            return "tas"

        raise ValueError("No surface temperature variable ('ts' or 'tas') found.")

    @staticmethod
    def _get_surface_albedo(ds):
        """
        Return the surface albedo field, or None if it cannot be calculated.
        'alb' (OIFS forecast albedo) is used when available; otherwise the albedo is
        calculated from the surface shortwave fluxes as rsus / rsds.
        """
        if "alb" in ds:
            return ds["alb"]

        if "rsds" in ds and "rsus" in ds:
            return ds["rsus"] / ds["rsds"]

        print(
            "Energy balance: no surface albedo ('alb' or 'rsds'/'rsus') found, "
            "the surface / non-surface albedo split is skipped."
        )
        return None

    def _print_summary(self, name):
        """
        Print the global mean warming (GMT) and polar amplification (AMP) of each term.
        """
        print(f"\nEnergy balance analysis: {name}")
        print(f"{'term':<14}{'GMT (°C)':>10}{'AMP (°C)':>10}")

        for term, values in self.summary[name].items():
            print(f"{term:<14}{values['GMT']:>10.2f}{values['AMP']:>10.2f}")

    def _plot(self, pairs, save=True):
        """
        Plot the energy balance decomposition, one panel per (experiment, reference) pair,
        as in Lunt et al. (2021), Fig. 4. The legend of each panel gives the global mean
        warming (GMT) and polar amplification (AMP) of each term.
        Figures are stored in the "energy_balance" subdirectory of the comparison plot directory.

        Parameters:
        pairs : list[tuple[str, str]]
            List of (experiment, reference) pairs to plot.

        save : bool, default=True
            If True, save the figure to disk.
        """

        # Create output directory
        plot_dir = self.comparison_plot_dir / "energy_balance"

        # Only create the directory if we are actually saving files
        if save:
            plot_dir.mkdir(
                parents=True,
                exist_ok=True
            )

        # Create filename from experiment directories
        experiment_dirs = [
            self.plot_dirs[exp].parent.name
            for exp in dict.fromkeys(exp for pair in pairs for exp in pair)
        ]

        experiment_string = "_".join(experiment_dirs)

        save_path = (
            plot_dir
            / f"energy_balance_{experiment_string}.png"
        )

        plot_energy_balance(
            decomposition={
                f"{exp}-{ref}": self.decomposition[f"{exp}-{ref}"]
                for exp, ref in pairs
            },
            summary=self.summary,
            titles={
                f"{exp}-{ref}": f"{self.experiment_labels[exp]} - {self.experiment_labels[ref]}"
                for exp, ref in pairs
            },
            save=save_path if save else None
        )


#==================================================#
# ------------- ANALYSIS FUNCTIONS --------------- #
#==================================================#

def eba_temperature(S, alpha_p, eps, H):
    """
    Surface temperature (K) implied by the energy balance of Eq. (1):
    tau = [(S (1 - alpha_p) + H) / (eps * sigma)]^(1/4)
    """
    return ((S * (1 - alpha_p) + H) / (eps * SIGMA))**0.25

def decompose_warming(exp, ref):
    """
    Decompose the zonal mean temperature change between two experiments into the
    contributions from planetary albedo, emissivity and meridional heat flux changes
    (Lunt et al., 2021, Eqs. 2 and 3).

    The incoming solar radiation of the reference experiment is used in all terms,
    since the decomposition assumes the same insolation in both experiments.

    Parameters:
    exp : xarray.Dataset
        Zonal mean energy balance quantities of the perturbed experiment.

    ref : xarray.Dataset
        Zonal mean energy balance quantities of the reference experiment.

    Returns:
    xarray.Dataset
        Zonal mean temperature changes (°C):
        - 'dT_gcm': surface temperature change simulated by the model
        - 'dT_eba': temperature change derived from the radiative fluxes
        - 'dT_alpha_p', 'dT_eps', 'dT_H': contributions from planetary albedo, emissivity and heat flux
        - 'dT_alpha_s', 'dT_alpha_ns': surface and non-surface albedo contributions (if available)
        - 'dT_sum': sum of the planetary albedo, emissivity and heat flux contributions
    """
    S = ref["S"]

    # Reference temperature derived from the radiative fluxes
    tau_ref = eba_temperature(S, ref["alpha_p"], ref["eps"], ref["H"])

    out = xr.Dataset()

    out["dT_gcm"] = exp["T"] - ref["T"]
    out["dT_eba"] = eba_temperature(exp["S"], exp["alpha_p"], exp["eps"], exp["H"]) - tau_ref

    # Contributions of the individual terms (Eq. 2)
    out["dT_alpha_p"] = eba_temperature(S, exp["alpha_p"], ref["eps"], ref["H"]) - tau_ref
    out["dT_eps"] = eba_temperature(S, ref["alpha_p"], exp["eps"], ref["H"]) - tau_ref
    out["dT_H"] = eba_temperature(S, ref["alpha_p"], ref["eps"], exp["H"]) - tau_ref

    # Surface and non-surface albedo contributions (Eq. 3)
    if "alpha_s" in exp and "alpha_s" in ref:
        alpha_s_change = exp["alpha_s"] - ref["alpha_s"]
        out["dT_alpha_s"] = eba_temperature(S, ref["alpha_p"] + alpha_s_change, ref["eps"], ref["H"]) - tau_ref
        out["dT_alpha_ns"] = out["dT_alpha_p"] - out["dT_alpha_s"]

    out["dT_sum"] = out["dT_alpha_p"] + out["dT_eps"] + out["dT_H"]

    return out

def zonal_profile_mean(da, lat_min=-90, lat_max=90):
    """
    Area-weighted mean of a zonal mean profile over a latitude band.
    The band is defined on absolute latitude, so that both hemispheres are included.
    """
    lat = da[[dim for dim in da.dims if "lat" in dim][0]]

    band = da.where(
        (np.abs(lat) >= lat_min) & (np.abs(lat) <= lat_max),
        drop=True
    )

    weights = np.cos(np.deg2rad(band[lat.name]))

    return band.weighted(weights).mean(lat.name)

def polar_amplification(da, polar_lat=60, tropical_lat=30):
    """
    Polar amplification of a zonal mean temperature change, defined as in Lunt et al. (2021)
    as the difference in warming between the high latitudes (polewards of ±60°) and the
    tropics (±30°).
    """
    return (
        zonal_profile_mean(da, lat_min=polar_lat)
        - zonal_profile_mean(da, lat_max=tropical_lat)
    )


#==================================================#
# ---------------- PLOTTING ---------------------- #
#==================================================#

def plot_energy_balance(
    decomposition,
    summary,
    titles,
    save=None,
    ncols=3,
    panel_size=(6, 4.5),
    dpi=300,
):
    """
    Plot the energy balance decomposition, one panel per experiment pair (Lunt et al., 2021, Fig. 4).

    Parameters:
    decomposition : dict[str, xarray.Dataset]
        Zonal mean temperature change decomposition for each pair (see decompose_warming).
    summary : dict[str, dict]
        Global mean (GMT) and polar amplification (AMP) of each term for each pair.
    titles : dict[str, str]
        Panel title for each pair.
    save : str or pathlib.Path, optional
        Output filename. If None, the figure is only shown.
    ncols : int, default=3
        Maximum number of panels per row.
    panel_size : tuple, default=(6, 4.5)
        Size of each panel in inches.
    dpi : int, default=300
        Figure resolution.
    """

    n = len(decomposition)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        dpi=dpi,
        squeeze=False,
        sharey=True
    )

    for ax, (name, ds) in zip(axes.flat, decomposition.items()):

        lat = ds[[dim for dim in ds.dims if "lat" in dim][0]]

        for term, style in EBA_TERMS.items():
            if term not in ds:
                continue

            values = summary[name][term]

            ax.plot(
                lat,
                ds[term],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                label=f"{style['label']}: GMT={values['GMT']:.1f}, AMP={values['AMP']:.1f}"
            )

        ax.axhline(0, color="grey", linewidth=0.8)

        ax.set_xlim(-90, 90)
        ax.set_xticks(np.arange(-90, 91, 30))
        ax.set_xlabel("Latitude")
        ax.set_title(titles[name])

        ax.grid(
            True,
            linestyle="--",
            alpha=0.4
        )

        ax.legend(fontsize=7, loc="best")

    for ax in axes[:, 0]:
        ax.set_ylabel("Temperature change (°C)")

    # Hide unused panels
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    plt.tight_layout()

    if save is not None:
        plt.savefig(
            save,
            bbox_inches="tight"
        )

    plt.show()
