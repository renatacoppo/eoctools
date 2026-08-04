import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

# open datasets
def load_last_years(file_path, nyears=100, time_dim="time_counter"):

    ds = xr.open_dataset(file_path)

    last_year = ds[time_dim].dt.year.max().item()
    first_year = last_year - nyears + 1

    ds = ds.sel(
        {time_dim: slice(
            f"{first_year}-01-01",
            f"{last_year}-12-31"
        )}
    )

    return ds

def load_last_years_old(base_path, pattern, nyears=None, which="last", time_dim="time_counter"):

    full_pattern = base_path / pattern
    files = sorted(full_pattern.parent.glob(full_pattern.name))

    # Extract start year from filenames
    years = [int(f.stem.split("_")[-1].split("-")[0]) for f in files]

    first_year = min(years)
    last_year = max(years)

    # --- check if last year is complete ---
    last_file = files[years.index(last_year)]
    ds_last = xr.open_dataset(last_file)

    if ds_last[time_dim].size < 12:
        print(f"Dropping incomplete year {last_year}")
        last_year -= 1

    if nyears is None:
        # load all years
    #    selected_files = [f for f, y in zip(files, years) if y <= last_year]
    #    start_year = min(years)
    #else:
    #    start_year = last_year - nyears + 1
    #    selected_files = [
    #        f for f, y in zip(files, years)
    #        if first_year <= y <= last_year
    #    ]
        start_year = first_year
        end_year = last_year

    else:
        if which == "last":
            start_year = last_year - nyears + 1               
            end_year = last_year

        elif which == "first":
            start_year = first_year
            end_year = first_year + nyears - 1

        else:
            raise ValueError("which must be 'first' or 'last'")

        selected_files = [
            f for f, y in zip(files, years)
            if start_year <= y <= end_year
        ]

    print(f"{pattern}: {start_year}-{end_year} ({len(selected_files)} files)")


    ds = xr.open_mfdataset(
        selected_files,
        combine="by_coords",
        parallel=True,
        chunks={time_dim: 120}
    )

    ds = ds.sortby(time_dim)

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

    if ecs is not None:
        plt.legend(
            title=f"Global mean\nECS = {ecs:.2f} °C"
        )
    else:
        plt.legend(
            title="Global mean"
        )

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