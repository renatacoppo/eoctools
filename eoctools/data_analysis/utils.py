import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

# open datasets
def load_last_years(
    file_path,
    nyears=100,
    time_dim="time_counter",
    end_year=None,
):
    ds = xr.open_dataset(file_path)

    # Determine final year
    if end_year is None:
        end_year = ds[time_dim].dt.year.max().item()

    first_year = end_year - nyears + 1

    ds = ds.sel(
        {
            time_dim: slice(
                f"{first_year}-01-01",
                f"{end_year}-12-31",
            )
        }
    )

    return ds

def load_simulation(
    file_path,
    skip_years=10,
    time_dim="time_counter",
    end_year=None,
):
    """
    Load all years except the first `skip_years`,
    optionally stopping at `end_year`.
    """

    ds = xr.open_dataset(file_path)

    years = ds[time_dim].dt.year

    first_available_year = years.min().item()
    last_available_year = years.max().item()

    if end_year is None:
        end_year = last_available_year

    first_year = first_available_year + skip_years

    ds = ds.sel(
        {
            time_dim: slice(
                f"{first_year}-01-01",
                f"{end_year}-12-31",
            )
        }
    )

    return ds

def open_exp(base_path, path):
    """Open dataset lazily (no slicing here)."""
    return xr.open_mfdataset(str(base_path / path), combine="by_coords", chunks={})

# annual weighted mean
def annual_mean(da):
    """Compute time-weighted annual mean."""
    w = da.time_counter.dt.days_in_month
    annual = (da * w).resample(time_counter="1YE").sum() / w.resample(time_counter="1YE").sum()
    return annual.mean("time_counter")

# global mean
def global_mean(da):
    """Area-weighted global mean for (lat, lon) DataArray."""
    weights = np.cos(np.deg2rad(da.lat))
    return da.weighted(weights).mean(("lat", "lon"))

def zonal_mean(da):
    """
    Compute latitude-weighted zonal mean.
    """
    weights = np.cos(np.deg2rad(da.lat))
    weights = weights / weights.mean()
    return da.weighted(weights).mean("lon")

def regional_mean_sst(da, lat_min, lat_max):
    """
    Area-weighted mean SST over a latitude band.

    Works with:
    (lat, lon)
    (time, lat, lon)
    or weird CMIP time names.
    """

    # ---- if dataset, grab tos automatically ----
    if isinstance(da, xr.Dataset):
        if "tos" in da:
            da = da["tos"]
        else:
            raise ValueError("Dataset does not contain 'tos'")

    # ---- detect coordinate names ----
    lat_name = "lat" if "lat" in da.dims else "latitude"
    lon_name = "lon" if "lon" in da.dims else "longitude"

    # ---- detect time dimension FLEXIBLY ----
    possible_time_names = ["time", "month", "time_counter", "t"]
    time_name = next((d for d in possible_time_names if d in da.dims), None)

    lat = da[lat_name]

    # ---- make slice work regardless of latitude order ----
    if lat[0] > lat[-1]:
        da_sel = da.sel({lat_name: slice(lat_max, lat_min)})
    else:
        da_sel = da.sel({lat_name: slice(lat_min, lat_max)})

    # ---- weights ----
    weights = np.cos(np.deg2rad(da_sel[lat_name]))

    # ---- dimensions to average over ----
    dims = [lat_name, lon_name]
    if time_name is not None:
        dims.append(time_name)

    return da_sel.weighted(weights).mean(dim=dims)

def high_lat_mean_sst(da):
    north = regional_mean_sst(da, 60, 90)
    south = regional_mean_sst(da, -90, -60)
    return 0.5 * (north + south)

def global_mean_tas(ds):
    tas = ds["tas"]

    # detect time dimension name
    for dim in ["month", "time", "t", "time_counter"]:
        if dim in tas.dims:
            tas = tas.mean(dim)
            break

    # Kelvin → Celsius
    tas = tas - 273.15

    # latitude weights
    weights = np.cos(np.deg2rad(tas.latitude))

    gm = tas.weighted(weights).mean(("latitude", "longitude"))

    return gm.values

# Annual global mean TAS timeseries
def annual_global_mean_tas(ds):
    """
    Compute global mean TAS per year.
    Returns a DataArray with 'time_counter' = year end.
    """
    tas = ds["tas"] - 273.15  # Kelvin → Celsius

    # Time weights for monthly data
    w = tas.time_counter.dt.days_in_month

    # Compute weighted annual mean
    annual = (tas * w).resample(time_counter="1YE").sum() / w.resample(time_counter="1YE").sum()

    # Latitude weights for global mean
    weights = np.cos(np.deg2rad(annual.lat))

    lat_name = "lat" if "lat" in ds.dims else "latitude"
    lon_name = "lon" if "lon" in ds.dims else "longitude"
    gm_annual = annual.weighted(weights).mean((lat_name, lon_name))

    return gm_annual

def annual_ts(da):
    w = da.time_counter.dt.days_in_month
    return (da * w).resample(time_counter="1YE").sum() / w.resample(time_counter="1YE").sum()

def global_mean_sst(da):
    """
    Area-weighted global mean SST.

    Works with Dataset or DataArray.
    Automatically:
      - extracts 'tos'
      - detects lat/lon names
      - detects time dimension
      - computes annual mean
    """

    # ---- if dataset, grab tos ----
    if isinstance(da, xr.Dataset):
        if "tos" in da:
            da = da["tos"]
        else:
            raise ValueError("Dataset does not contain 'tos'")

    # ---- detect coordinate names ----
    lat_name = "lat" if "lat" in da.dims else "latitude"
    lon_name = "lon" if "lon" in da.dims else "longitude"

    # ---- detect time dimension FLEXIBLY ----
    possible_time_names = ["time", "month", "time_counter", "t"]
    time_name = next((d for d in possible_time_names if d in da.dims), None)

    # ---- compute annual mean if time exists ----
    if time_name is not None:
        da = da.mean(dim=time_name)

    # ---- weights ----
    weights = np.cos(np.deg2rad(da[lat_name]))

    return da.weighted(weights).mean(dim=(lat_name, lon_name))

def simple_regional_mean_sst(da, lat_min, lat_max):
    """
    Area-weighted mean SST over a latitude band.
    da must have dimensions (lat, lon)
    """
    # select latitude band
    da_sel = da.sel(lat=slice(lat_min, lat_max))
    
    # latitude weights
    weights = np.cos(np.deg2rad(da_sel.lat))
    
    return da_sel.weighted(weights).mean(dim=("lat", "lon"))

def trop_mean_sst(da):
    trop = regional_mean_sst(da, -31, 31)
    return trop

def global_toa_ts(ds):
    """
    Annual global mean TOA net radiation (W m-2)
    Returns one value per year.
    """

    # --- net radiation ---
    net = ds["rsdt"] - ds["rsut"] - ds["rlut"]

    # --- annual mean (time-weighted like TAS) ---
    net_ann = annual_ts(net)

    # --- area weights ---
    weights = np.cos(np.deg2rad(net_ann.lat))

    # --- global mean per year ---
    net_gm = net_ann.weighted(weights).mean(("lat", "lon"))

    return net_gm

def global_surface_net_radiation(ds):

    net_sfc = (
        ds["rsns"]+ ds["rlns"]
    )

    return global_mean(net_sfc)

def calculate_ecs(global_means, co2_levels):
    """
    Calculate ECS from a linear fit of global mean temperature
    against log2(CO2).

    Parameters
    ----------
    global_means : dict
        Experiment names and global mean temperatures.

    co2_levels : dict
        Experiment names and CO2 multipliers.

    Returns
    -------
    ecs : float
        Temperature response per CO2 doubling.
    """

    co2 = np.array(
        [
            co2_levels[k]
            for k in global_means
        ]
    )

    temps = np.array(
        [
            float(global_means[k])
            for k in global_means
        ]
    )

    x = np.log2(co2)

    slope, intercept = np.polyfit(
        x,
        temps,
        1
    )

    return slope

def polar_mean(da, lat_threshold=60):
    """
    Area-weighted mean over both polar regions.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with lat/lon dimensions.

    lat_threshold : float
        Minimum absolute latitude defining the polar region.

    Returns
    -------
    xarray.DataArray
        Area-weighted polar mean.
    """

    polar = da.where(
        np.abs(da.lat) >= lat_threshold,
        drop=True
    )

    weights = np.cos(np.deg2rad(polar.lat))

    return polar.weighted(weights).mean(("lat", "lon"))


def prepare_pr(da):
    """Convert kg m-2 s-1 → mm/day and compute annual mean."""
    da = da * 86400.0
    da.attrs["units"] = "mm/day"
    return (da)

### -------------------------- ###
###     Plotting functions     ###
### -------------------------- ###

def plot_reference_anomalies(
    reference, 
    anomalies,
    reference_name="pi",
    absolute_title="Mean",
    anomaly_title="Anomaly",
    absolute_levels=None,
    anomaly_levels=None,
    absolute_cmap="viridis",
    anomaly_cmap="RdBu_r",
    absolute_label="",
    anomaly_label="",
    figsize_height=5,
    dpi=150,
    save=None,
):
    """
    Plot one reference map and anomalies relative to it.

    Parameters
    ----------
    reference : xarray.DataArray
        2D (lat, lon) reference field.

    anomalies : dict[str, xarray.DataArray]
        Dictionary of anomaly fields.

    reference_name : str
        Name displayed in first panel.

    absolute_title : str
        Title of reference panel.

    anomaly_title : str
        Suffix for anomaly titles.

    absolute_levels, anomaly_levels : array-like
        Contour levels.

    absolute_cmap, anomaly_cmap
        Colormaps.

    absolute_label, anomaly_label : str
        Colorbar labels.

    save : str or Path, optional
        Output filename.
    """

    experiments = list(anomalies.keys())
    nexp = len(experiments)

    fig, axes = plt.subplots(
        1,
        nexp + 1,
        figsize=(6 * (nexp + 1), figsize_height),
        dpi=dpi,
        constrained_layout=True,
        subplot_kw={"projection": ccrs.Robinson()},
    )

    if nexp == 1:
        axes = np.asarray(axes)

    # ---------------- reference ----------------

    ref_cyc, lon_cyc = add_cyclic_point(
        reference,
        coord=reference.lon,
    )

    cf_ref = axes[0].contourf(
        lon_cyc,
        reference.lat,
        ref_cyc,
        levels=absolute_levels,
        cmap=absolute_cmap,
        transform=ccrs.PlateCarree(),
        extend="both",
    )

    axes[0].set_title(f"{reference_name}: {absolute_title}")
    axes[0].set_global()

    # ---------------- anomalies ----------------

    for i, exp in enumerate(experiments, start=1):

        da = anomalies[exp]

        da_cyc, lon_cyc = add_cyclic_point(
            da,
            coord=da.lon,
        )

        cf_anom = axes[i].contourf(
            lon_cyc,
            da.lat,
            da_cyc,
            levels=anomaly_levels,
            cmap=anomaly_cmap,
            transform=ccrs.PlateCarree(),
            extend="both",
        )

        axes[i].set_title(f"{exp}: {anomaly_title}")
        axes[i].set_global()

    # ---------------- colorbars ----------------

    cbar1 = fig.colorbar(
        cf_ref,
        ax=axes[0],
        orientation="horizontal",
        shrink=0.8,
        pad=0.08,
    )
    cbar1.set_label(absolute_label)

    cbar2 = fig.colorbar(
        cf_anom,
        ax=axes[1:],
        orientation="horizontal",
        shrink=0.8,
        pad=0.08,
    )
    cbar2.set_label(anomaly_label)

    if save is not None:
        plt.savefig(save, bbox_inches="tight")

    plt.show()

def plot_mean_map(
    field,
    experiment,
    title,
    levels,
    cmap,
    colorbar_label,
    figsize=(10, 5),
    dpi=150,
    save=None,
):
    """
    Plot a single global mean field.

    Parameters
    ----------
    field : xarray.DataArray
        Two-dimensional (lat, lon) field.

    experiment : str
        Experiment name (e.g. "pi", "x3").

    title : str
        Variable title.

    levels : array-like
        Contour levels.

    cmap : matplotlib colormap

    colorbar_label : str

    save : str or Path, optional
    """

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi,
        constrained_layout=True,
        subplot_kw={"projection": ccrs.Robinson()},
    )

    field_cyc, lon_cyc = add_cyclic_point(
        field,
        coord=field.lon,
    )

    cf = ax.contourf(
        lon_cyc,
        field.lat,
        field_cyc,
        levels=levels,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        extend="both",
    )

    ax.set_title(f"{experiment}: {title}")
    ax.set_global()

    cbar = fig.colorbar(
        cf,
        ax=ax,
        orientation="horizontal",
        shrink=0.8,
        pad=0.08,
    )

    cbar.set_label(colorbar_label)

    if save is not None:
        plt.savefig(save, bbox_inches="tight")

    plt.show()

def plot_zonal_mean(
    field,
    experiment,
    title,
    ylabel,
    save=None,
    figsize=(9, 5),
    dpi=300,
):
    """
    Plot zonal mean as a function of latitude.

    Parameters
    ----------
    field : xarray.DataArray
        Zonal mean field with dimension (lat).

    experiment : str
        Experiment name (e.g. "pi", "x3").

    title : str
        Plot title.

    ylabel : str
        Y-axis label.

    save : str or Path, optional
        Output filename.
    """

    plt.figure(
        figsize=figsize,
        dpi=dpi
    )

    plt.plot(
        field.lat,
        field,
        linewidth=2.2,
        label=experiment
    )

    plt.xlabel("Latitude")
    plt.ylabel(ylabel)

    plt.title(title)

    plt.legend()

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    if save is not None:
        plt.savefig(
            save,
            bbox_inches="tight"
        )

    plt.show()

def plot_zonal_anomalies(
    anomalies,
    title,
    ylabel,
    save=None,
    figsize=(9, 5),
    dpi=300,
):
    """
    Plot zonal mean anomalies as a function of latitude.

    Parameters
    ----------
    anomalies : dict[str, xarray.DataArray]
        Dictionary containing anomaly fields.

    title : str
        Plot title.

    ylabel : str
        Y-axis label.

    save : str or Path, optional
        Output filename.
    """

    plt.figure(
        figsize=figsize,
        dpi=dpi
    )

    for exp, da in anomalies.items():

        plt.plot(
            da.lat,
            da,
            linewidth=2.2,
            label=exp
        )

    plt.axhline(
        0,
        color="k",
        linestyle="--",
        linewidth=1
    )

    plt.xlabel("Latitude")
    plt.ylabel(ylabel)

    plt.title(title)

    plt.legend()

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    if save is not None:
        plt.savefig(
            save,
            bbox_inches="tight"
        )

    plt.show()

def plot_zonal_time(
    field,
    experiment,
    title,
    cmap,
    colorbar_label,
    levels,
    save,
    figsize=(8, 5),
    dpi=150
):

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi
    )

    # Convert cftime dates to integer years
    years = field.time_counter.dt.year.values
    lat = field.lat.values
    temperature = field.T.values

    cf = ax.contourf(
        years,
        lat,
        temperature,
        levels=levels,
        cmap=cmap,
        extend="both"
    )

    cbar = fig.colorbar(
        cf,
        ax=ax,
        orientation="vertical"
    )

    cbar.set_label(colorbar_label)

    ax.set_xlabel("Year")
    ax.set_ylabel("Latitude")
    ax.set_title(title)

    fig.tight_layout()

    if save is not None:
        fig.savefig(
            save,
            bbox_inches="tight"
        )

    plt.show()
    plt.close(fig)

def plot_zonal_time_old(
    field,
    experiment,
    title,
    cmap,
    colorbar_label,
    levels=20,
    save=None,
    figsize=(10, 5),
    dpi=150,
):
    """
    Plot zonal mean time evolution (latitude vs time).

    Parameters
    ----------
    field : xarray.DataArray
        Zonal mean field with dimensions (time_counter, lat).

    experiment : str
        Experiment name.

    title : str
        Plot title.

    cmap : matplotlib colormap
        Colormap.

    colorbar_label : str
        Colorbar label.

    levels : int or array-like
        Number of contour levels or explicit levels.

    save : str or Path, optional
        Output filename.
    """

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi
    )

    cf = ax.contourf(
        field.time_counter,
        field.lat,
        field.T,
        levels=levels,
        cmap=cmap,
        extend="both"
    )

    cbar = fig.colorbar(
        cf,
        ax=ax,
        orientation="vertical"
    )

    cbar.set_label(colorbar_label)

    ax.set_xlabel("Year")
    ax.set_ylabel("Latitude")

    ax.set_title(
        f"{experiment}: {title}"
    )

    plt.tight_layout()

    if save is not None:
        plt.savefig(
            save,
            bbox_inches="tight"
        )

    plt.show()


def plot_global_mean_vs_co2(
    values,
    co2_levels,
    ecs=None,
    polar_amp=None,
    ylabel="Global mean temperature (°C)",
    xlabel="CO₂ (× pre-industrial)",
    title=None,
    ylim=None,
    xlim=None,
    figsize=(6, 4),
    dpi=300,
    save=None,
):
    """
    Plot global mean value as a function of CO2 concentration.

    Parameters
    ----------
    values : dict
        Dictionary with experiment names as keys and scalar xarray DataArrays
        as values.

        Example:
        {
            "pi": 14.5,
            "x3": 22.3,
            "x6": 31.1
        }

    co2_levels : dict
        Dictionary mapping experiments to CO2 multipliers.

        Example:
        {
            "pi": 1,
            "x3": 3,
            "x6": 6
        }
    
    ecs : float, optional
        Equilibrium climate sensitivity.

    polar_amp : dict, optional
        Dictionary containing the polar amplification factor for each
        non-reference experiment.

    ylabel : str
        Y-axis label.

    xlabel : str
        X-axis label.

    title : str, optional
        Plot title.

    ylim : tuple, optional
        Y-axis limits.

    xlim : tuple, optional
        X-axis limits.

    save : str or Path, optional
        Output filename.
    """

    plt.figure(
        figsize=figsize,
        dpi=dpi
    )

    for exp, value in values.items():

        co2 = co2_levels[exp]

        value = float(value)

        plt.scatter(
            co2,
            value,
            s=80,
            label=f"{exp}: {value:.2f}"
        )

        plt.text(
            co2,
            value + 0.15,
            exp,
            ha="center"
        )

    if xlim is not None:
        plt.xlim(*xlim)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    plt.xticks(
        sorted(set(co2_levels.values()))
    )

    # -----------------------------
    # Legend
    # -----------------------------

    handles, labels = plt.gca().get_legend_handles_labels()

    # Add ECS as a text-only legend entry
    if ecs is not None:
        handles.append(
            plt.Line2D(
                [],
                [], 
                linestyle="none",
                marker="",
                label=f"ECS = {ecs:.2f} °C"
            )
        )

    # Add polar amplification values
    if polar_amp is not None:

        handles.append(
            plt.Line2D(
                [],
                [],
                linestyle="none",
                marker="",
                label="Polar amplification"
            )
        )

        for exp, value in polar_amp.items():
            handles.append(
                plt.Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="",
                    label=f"{exp} = {value:.2f}"
                )
            )

    plt.legend(
        handles=handles,
        title="Global mean",
    )

    plt.tight_layout()

    if save is not None:
        plt.savefig(
            save,
            dpi=dpi,
            bbox_inches="tight"
        )

    plt.show()
    plt.close()

def plot_global_mean_vs_co2_dp(
    data,
    title,
    save,
    model_styles,
    figsize=(8, 5),
    dpi=150,
):
    """
    Plot global mean surface temperature against CO2 concentration.

    Parameters
    ----------
    data : list of dict
        Plotting data. Each dictionary must contain:
        "model", "CO2", "T", and "pi".

    title : str
        Figure title.

    save : Path
        Output filename.

    model_styles : dict
        Dictionary defining color and marker for each model.

    figsize : tuple
        Figure size.

    dpi : int
        Figure resolution.
    """

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi,
    )

    models = sorted(
        set(d["model"] for d in data)
    )

    for model in models:

        style = model_styles[model]

        model_data = [
            d for d in data
            if d["model"] == model
        ]

        # Sort by CO2
        model_data = sorted(
            model_data,
            key=lambda d: d["CO2"],
        )

        xs = np.array([
            d["CO2"]
            for d in model_data
        ])

        ys = np.array([
            d["T"]
            for d in model_data
        ])

        # Connecting line
        ax.plot(
            xs,
            ys,
            color=style["color"],
            linewidth=1.5,
            zorder=2,
        )

        # Points
        for d in model_data:

            size = 40 if d["pi"] else 120

            ax.scatter(
                d["CO2"],
                d["T"],
                color=style["color"],
                marker=style["marker"],
                s=size,
                edgecolor="k",
                zorder=3,
            )

    # Labels
    ax.set_xlabel(
        "CO₂ concentration (× pre-industrial)"
    )

    ax.set_ylabel(
        "Global mean surface temperature (°C)"
    )

    ax.set_title(title)

    ax.grid(
        True,
        alpha=0.4,
        linestyle="--",
    )

    # Legend
    for model in models:

        style = model_styles[model]

        ax.scatter(
            [],
            [],
            color=style["color"],
            marker=style["marker"],
            label=model,
        )

    ax.legend(
        title="Model",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    fig.tight_layout()

    if save is not None:
        fig.savefig(
            save,
            bbox_inches="tight",
        )

    plt.show()
    plt.close(fig)

def plot_global_timeseries(
    series,
    ylabel,
    title,
    use_model_years=False,
    xlabel="Simulation year",
    figsize=(10, 5),
    dpi=150,
    save=None,
):
    """
    Plot global mean time series for one or more experiments.

    Parameters
    ----------
    series : dict
        Dictionary of xarray.DataArray objects with dimension
        (time_counter,).

    ylabel : str
        Label for the y-axis.

    title : str
        Figure title.

    xlabel : str, optional
        Label for the x-axis.

    save : str or Path, optional
        Output filename.
    """

    plt.figure(figsize=figsize, dpi=dpi)

    for exp, da in series.items():

        if use_model_years:
            years = da.time_counter.dt.year
        else:
            years = np.arange(1, len(da) + 1)

        plt.plot(
            years,
            da.values,
            linewidth=2,
            label=exp
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4
    )
    
    plt.legend()

    plt.tight_layout()

    if save is not None:
        plt.savefig(
            save,
            bbox_inches="tight"
        )

    plt.show()

def plot_gregory(
    tas_series,
    toa_series,
    colors=None,
    xlabel="Global mean TAS (°C)",
    ylabel="TOA net radiation (W m⁻²)",
    title="Gregory plots",
    figsize=(7, 6),
    dpi=150,
    save=None,
):

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi
    )

    # Automatic colors
    if colors is None:
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        colors = {
            exp: color_cycle[i % len(color_cycle)]
            for i, exp in enumerate(tas_series)
        }

    for k in tas_series:

        tas = tas_series[k]
        toa = toa_series[k]

        # Convert directly to numpy arrays
        x = tas.values
        y = toa.values

        # Check lengths
        if len(x) != len(y):
            raise ValueError(
                f"{k}: TAS has {len(x)} points, "
                f"TOA has {len(y)} points"
            )

        # Remove NaNs
        valid = np.isfinite(x) & np.isfinite(y)

        x = x[valid]
        y = y[valid]

        print(
            f"{k}: {len(x)} valid points"
        )

        if len(x) < 2:
            raise ValueError(
                f"{k}: not enough valid data points "
                f"for regression."
            )

        # Regression
        slope, intercept = np.polyfit(x, y, 1)

        x_fit = np.linspace(
            x.min(),
            x.max(),
            100
        )

        y_fit = slope * x_fit + intercept

        color = colors[k]

        # Evolution
        ax.plot(
            x,
            y,
            color=color,
            alpha=0.6,
            linewidth=1
        )

        # Scatter
        ax.scatter(
            x,
            y,
            color=color,
            s=25,
            label=k
        )

        # First year
        ax.scatter(
            x[0],
            y[0],
            color=color,
            s=100,
            marker="o",
            edgecolor="k",
            zorder=3
        )

        # Last year
        ax.scatter(
            x[-1],
            y[-1],
            color=color,
            s=100,
            marker="s",
            edgecolor="k",
            zorder=3
        )

        # Regression
        ax.plot(
            x_fit,
            y_fit,
            color=color,
            linestyle="--"
        )

    ax.axhline(
        0,
        color="k",
        linestyle=":",
        linewidth=1
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    ax.legend(title="Experiment")

    fig.tight_layout()

    if save is not None:
        fig.savefig(
            save,
            bbox_inches="tight"
        )

    plt.show()

def plot_global_mean_vs_logco2(
    values,
    logco2_levels,
    ylabel="Global mean precipitation (mm/day)",
    xlabel="log₂(CO₂)",
    title=None,
    ylim=None,
    xlim=None,
    figsize=(6, 4),
    dpi=300,
    save=None,
):
    """
    Plot global mean values as a function of log2(CO2),
    including a linear regression.

    Parameters
    ----------
    values : dict
        Dictionary with experiment names as keys and scalar values
        (or scalar xarray.DataArrays) as values.

    co2_levels : dict
        Dictionary mapping experiment names to CO2 multipliers.

    ylabel : str
        Y-axis label.

    xlabel : str
        X-axis label.

    title : str, optional
        Figure title.

    ylim : tuple, optional
        Y-axis limits.

    xlim : tuple, optional
        X-axis limits.

    save : str or Path, optional
        Output filename.
    """

    # --------------------------------------------------
    # Prepare data
    # --------------------------------------------------

    
    plt.figure(figsize=figsize, dpi=dpi)

    experiments = []
    x = []
    y = []

    for exp, value in values.items():

        if exp not in logco2_levels:
            continue

        xi = float(logco2_levels[exp])
        yi = float(value)

        if not np.isfinite(xi) or not np.isfinite(yi):
            continue

        experiments.append(exp)
        x.append(xi)
        y.append(yi)

    x = np.asarray(x)
    y = np.asarray(y)

    # Remove invalid values
    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    experiments = [
        exp for exp, valid_value in zip(
            experiments,
            valid
        )
        if valid_value
    ]

    if len(x) < 2:
        raise ValueError(
            "At least two valid experiments are required "
            "for the regression."
        )

    # --------------------------------------------------
    # Linear regression
    # --------------------------------------------------

    slope, intercept = np.polyfit(x, y, 1)

    x_fit = np.linspace(
        x.min(),
        x.max(),
        100
    )

    y_fit = slope * x_fit + intercept

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    plt.figure(
        figsize=figsize,
        dpi=dpi
    )

    # Experiment points
    for exp, xi, yi in zip(experiments, x, y):

        plt.scatter(
            xi,
            yi,
            s=80,
            label=f"{exp}: {yi:.2f}"
        )

        plt.text(
            xi,
            yi + 0.02 * (y.max() - y.min()),
            exp,
            ha="center"
        )

    # Regression line
    plt.plot(
        x_fit,
        y_fit,
        linestyle="--",
        linewidth=2,
        label=(
            f"Regression\n"
            f"slope = {slope:.3f}\n"
            f"intercept = {intercept:.3f}"
        )
    )

    # --------------------------------------------------
    # Axis limits
    # --------------------------------------------------

    if xlim is not None:
        plt.xlim(*xlim)

    if ylim is not None:
        plt.ylim(*ylim)

    # --------------------------------------------------
    # X ticks at regular log2(CO2) intervals
    # --------------------------------------------------

    max_logco2 = int(np.ceil(max(logco2_levels.values())))

    xticks = np.arange(
        0,
        max_logco2 + 1
    )

    plt.xticks(
        xticks,
        labels=[
            f"{int(x)}" for x in xticks
        ]
    )

    # --------------------------------------------------
    # Labels and title
    # --------------------------------------------------

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    # --------------------------------------------------
    # Legend
    # --------------------------------------------------

    plt.legend(
        title="Global mean"
    )

    # --------------------------------------------------
    # Grid
    # --------------------------------------------------

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    if save is not None:
        plt.savefig(
            save,
            bbox_inches="tight"
        )

    plt.show()

    return slope, intercept

def plot_sst_gradient_vs_global_mean(
    global_mean_sst,
    meridional_gradient,
    reference=None,
    xlabel="Global mean SST (°C)",
    ylabel="Tropics – High latitude SST (°C)",
    title="Meridional SST gradient vs global mean SST",
    figsize=(6, 4),
    dpi=150,
    ylim=None,
    xlim=None,
    annotate=True,
    save=None,
):
    """
    Plot meridional SST gradient against global mean SST
    for multiple experiments.

    The reference experiment is included in the plot and is
    identified by `reference`. All other experiments available
    in the input dictionaries are plotted automatically.

    Parameters
    ----------
    global_mean_sst : dict
        Dictionary containing one scalar global mean SST value
        per experiment.

    meridional_gradient : dict
        Dictionary containing one scalar meridional SST gradient
        per experiment.

    reference : str, optional
        Name of the reference experiment. Default is "pi".

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    title : str, optional
        Figure title.

    figsize : tuple, optional
        Figure size.

    dpi : int, optional
        Figure resolution.

    ylim : tuple, optional
        Y-axis limits.

    xlim : tuple, optional
        X-axis limits.

    annotate : bool, optional
        Whether to write experiment names next to points.

    save : str or Path, optional
        Output filename.
    """

    # Find experiments available in both dictionaries
    experiments = [
        exp
        for exp in global_mean_sst
        if exp in meridional_gradient
    ]

    if not experiments:
        raise ValueError(
            "No common experiments found between "
            "global_mean_sst and meridional_gradient."
        )

    # Create figure
    plt.figure(
        figsize=figsize,
        dpi=dpi
    )

    # Plot each experiment
    for exp in experiments:

        x = float(global_mean_sst[exp])
        y = float(meridional_gradient[exp])

        # Reference experiment gets a different marker
        if exp == reference:
            plt.scatter(
                x,
                y,
                s=90,
                marker="o",
                color="black",
                edgecolor="black",
                zorder=3,
                label=exp
            )
        else:
            plt.scatter(
                x,
                y,
                s=80,
                zorder=3,
                label=exp
            )

        # Experiment name
        if annotate:
            plt.text(
                x,
                y + 0.15,
                exp,
                ha="center",
                fontsize=9
            )

    # Axis limits
    if xlim is not None:
        plt.xlim(*xlim)

    if ylim is not None:
        plt.ylim(*ylim)

    # Labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Legend
    plt.legend(
        title="Experiment"
    )

    # Grid
    plt.grid(
        True,
        linestyle="--",
        alpha=0.4
    )
    plt.tight_layout()

    # Save
    if save is not None:
        plt.savefig(
            save,
            bbox_inches="tight"
        )

    plt.show()


def plot_sst_gradient_vs_global_sst_dpn(
    data,
    title,
    save,
    model_styles,
    figsize=(8, 5),
    dpi=150,
):
    """
    Plot meridional SST gradient against global mean SST. For comparison with DeepMIP results.
    """

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi,
    )

    models = sorted(
        set(d["model"] for d in data)
    )

    # --------------------------------------------------
    # Scatter points
    # --------------------------------------------------

    for d in data:

        style = model_styles[d["model"]]

        size = 50 if d["pi"] else 130

        ax.scatter(
            float(d["x"]),
            float(d["y"]),
            color=style["color"],
            marker=style["marker"],
            s=size,
            edgecolor="k",
            zorder=3,
        )

        ax.text(
            float(d["x"]) + 0.1,
            float(d["y"]) + 0.1,
            d["exp"],
            fontsize=8,
            zorder=4,
        )

    # --------------------------------------------------
    # Connecting lines
    # --------------------------------------------------

    for model in models:

        style = model_styles[model]

        model_data = [
            d for d in data
            if d["model"] == model
        ]

        model_data = sorted(
            model_data,
            key=lambda d: d["x"],
        )

        xs = np.array([
            float(d["x"])
            for d in model_data
        ])

        ys = np.array([
            float(d["y"])
            for d in model_data
        ])

        ax.plot(
            xs,
            ys,
            color=style["color"],
            linewidth=1.5,
            zorder=2,
        )

    # --------------------------------------------------
    # Labels
    # --------------------------------------------------

    ax.set_xlabel(
        "Global mean SST (°C)"
    )

    ax.set_ylabel(
        "Tropics – High latitude SST (°C)"
    )

    ax.set_title(title)

    ax.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    # --------------------------------------------------
    # Legend
    # --------------------------------------------------

    handles = []

    for model in models:

        style = model_styles[model]

        handles.append(
            plt.Line2D(
                [],
                [],
                color=style["color"],
                marker=style["marker"],
                linestyle="None",
                label=model,
            )
        )

    ax.legend(
        handles=handles,
        title="Model",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    fig.tight_layout()

    if save is not None:
        fig.savefig(
            save,
            bbox_inches="tight",
        )

    plt.show()
    plt.close(fig)



DEEP_MIP_MODEL_STYLES = {
    "IPSL":      {"color": "lightblue", "marker": "D"},
    "GFDL":      {"color": "orange",    "marker": "o"},
    "CESM":      {"color": "blue",      "marker": "s"},
    "INM":       {"color": "purple",    "marker": "*"},
    "COSMOS":    {"color": "brown",     "marker": "^"},
    "HadCM":     {"color": "yellow",    "marker": "v"},
    "MIROC":     {"color": "red",       "marker": "^"},
    "NorESM":    {"color": "pink",      "marker": "v"},
    "EC-EARTH4": {"color": "green",     "marker": "X"},
}

def plot_radiation_balance(
    toa,
    sfc,
    difference,
    title,
    save,
):

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
    )

    # --------------------------------------------------
    # Net TOA and net SFC
    # --------------------------------------------------

    for exp in toa:

        axes[0].plot(
            toa[exp]["time_counter"],
            toa[exp],
            label=f"{exp} TOA",
        )

        axes[0].plot(
            sfc[exp]["time_counter"],
            sfc[exp],
            linestyle="--",
            label=f"{exp} SFC",
        )

    axes[0].axhline(
        0,
        linestyle=":",
        linewidth=1,
    )

    axes[0].set_ylabel("Net radiation (W m$^{-2}$)")
    axes[0].set_title("Net TOA and net SFC radiation")
    axes[0].legend()

    # --------------------------------------------------
    # Difference
    # --------------------------------------------------

    for exp in difference:

        axes[1].plot(
            difference[exp]["time_counter"],
            difference[exp],
            label=exp,
        )

    axes[1].axhline(
        0,
        linestyle=":",
        linewidth=1,
    )

    axes[1].set_ylabel(
        "TOA − SFC (W m$^{-2}$)"
    )

    axes[1].set_xlabel("Year")
    axes[1].set_title(
        "TOA − SFC radiation imbalance"
    )

    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save, dpi=300)
    plt.close()