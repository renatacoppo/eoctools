#==================================================#
# - UTILITY FUNCTIONS FOR CLIMATE MODEL ANALYSIS - #
#==================================================#

# This module contains helper functions used throughout the climate-model analysis workflow.
# The input data are assumed to already contain annual mean values.

#==================================================#
# ------------------ IMPORTS --------------------- #
#==================================================#

import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

#==================================================#
# ---------------- DATA LOADING -------------------#
#==================================================#

def load_last_years(
    file_path,
    nyears=100,
    time_dim="time_counter",
    end_year=None,
):
    """
    Open a dataset and retain only the last nyears simulation years.
    This is useful for equilibrium or quasi-equilibrium analysis where the climate
    state is represented by the final part of a long simulation.

    Parameters:
    file_path : str or pathlib.Path
        Path to the NetCDF dataset.

    nyears : int, default=100
        Number of years to retain.

    time_dim : str, default="time_counter"
        Name of the time coordinate.

    end_year : int, optional
        Final year to include. If None, the last available year is used.

    Returns:
    xarray.Dataset
        Dataset sliced from 1 January of the first selected year
        through 31 December of the final selected year.
    """
    ds = xr.open_dataset(file_path)

    # Determine final year
    if end_year is None:
        end_year = ds[time_dim].dt.year.max().item()

    # First year of the nyears-long window ending at end_year (inclusive)
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
    time_dim="time_counter",
    end_year=None,
):
    """
    Open a full simulation, optionally truncated at a given final year.
    The dataset is kept from the first available year up to end_year.
    No spin-up period is removed: every year from the start of the file is retained.

    Parameters:
    file_path : str or pathlib.Path
        Path to the NetCDF dataset.

    time_dim : str, default="time_counter"
        Name of the time coordinate.

    end_year : int, optional
        Final year to include. If None, the last available year is used.

    Returns:
    xarray.Dataset
        Dataset from 1 January of the first available year
        through 31 December of end_year.
    """

    ds = xr.open_dataset(file_path)

    # Range of years available in the file
    years = ds[time_dim].dt.year

    first_available_year = years.min().item()
    last_available_year = years.max().item()

    if end_year is None:
        end_year = last_available_year

    first_year = first_available_year

    ds = ds.sel(
        {
            time_dim: slice(
                f"{first_year}-01-01",
                f"{end_year}-12-31",
            )
        }
    )

    return ds


#======================================#
# ----- GENERAL SPATIAL AVERAGING ---- #
#======================================#

def get_lat_lon_names(da):
    """
    Detect the names of the latitude and longitude dimensions.
    Both the short ('lat', 'lon') and long ('latitude', 'longitude')
    naming conventions are supported, so the averaging functions work
    with either type of model output.

    Parameters:
    da : xarray.DataArray
        Field with latitude and longitude dimensions.

    Returns:
    lat_name : str
        Name of the latitude dimension.

    lon_name : str
        Name of the longitude dimension.
    """
    for lat_name, lon_name in [("lat", "lon"), ("latitude", "longitude")]:
        if lat_name in da.dims and lon_name in da.dims:
            return lat_name, lon_name

    raise ValueError(
        "Could not find latitude/longitude dimensions "
        "('lat'/'lon' or 'latitude'/'longitude'). "
        f"Available dimensions: {list(da.dims)}"
    )

# Global mean
def global_mean(da):
    """
    Compute an area-weighted global mean.
    Latitude weighting using cos(latitude) approximates the decreasing area
    of lat-lon grid cells toward the poles.

    Parameters:
    da : xarray.DataArray
        Field with dimensions 'lat'/'lon' or 'latitude'/'longitude'.

    Returns:
    xarray.DataArray
        Area-weighted global mean (any remaining dimensions, e.g. time, are kept).
    """
    lat_name, lon_name = get_lat_lon_names(da)

    # Approximate grid-cell area with cos(latitude) weights
    weights = np.cos(np.deg2rad(da[lat_name]))
    return da.weighted(weights).mean((lat_name, lon_name))

def zonal_mean(da):
    """
    Compute the zonal mean of a field.
    The longitude dimension is averaged, leaving a latitude-dependent profile.

    Parameters:
    da : xarray.DataArray
        Field with dimensions 'lat'/'lon' or 'latitude'/'longitude'.

    Returns:
    xarray.DataArray
        Zonal mean as a function of latitude (and any remaining dimensions, e.g. time).
    """
    lat_name, lon_name = get_lat_lon_names(da)

    # Latitude weights are included for consistency with other spatial averaging functions,
    # although they do not affect a simple longitude mean.

    weights = np.cos(np.deg2rad(da[lat_name]))
    weights = weights / weights.mean()

    return da.weighted(weights).mean(lon_name)

#===========================================#
# SEA SURFACE TEMPERATURE (SST) DIAGNOSTICS #
#===========================================#

def regional_mean_sst(da, lat_min, lat_max):
    """
    Compute the area-weighted mean SST over a latitude band.
    The function accepts either an xarray Dataset containing 'tos'
    or a DataArray. It also supports several common names for latitude,
    longitude and time dimensions.
    
    If a time dimension is present, it is averaged as well.

    Parameters:
    da : xarray.Dataset or xarray.DataArray
        SST dataset or field.

    lat_min : float
        Southern boundary of the latitude band (degrees north).

    lat_max : float
        Northern boundary of the latitude band (degrees north).

    Returns:
    xarray.DataArray
        Area-weighted mean SST over the selected latitude band (same units as the input).
    """

    # If a Dataset is provided, automatically extract sea surface temperature.
    if isinstance(da, xr.Dataset):
        if "tos" in da:
            da = da["tos"]
        else:
            raise ValueError("Dataset does not contain 'tos'")

    # Support both common coordinate naming conventions
    lat_name, lon_name = get_lat_lon_names(da)

    # Identify the time dimension if one is present
    possible_time_names = ["time", "month", "year", "time_counter", "t"]
    time_name = next((d for d in possible_time_names if d in da.dims), None)

    lat = da[lat_name]

    # Latitude coordinates may run north-to-south or south-to-north.
    # Select the correct slice in either case.
    if lat[0] > lat[-1]:
        da_sel = da.sel({lat_name: slice(lat_max, lat_min)})
    else:
        da_sel = da.sel({lat_name: slice(lat_min, lat_max)})

    # Approximate grid-cell area using cosine latitude weighting
    weights = np.cos(np.deg2rad(da_sel[lat_name]))

    # Average horizontally and, if present, over time
    dims = [lat_name, lon_name]
    if time_name is not None:
        dims.append(time_name)

    return da_sel.weighted(weights).mean(dim=dims)

def high_lat_mean_sst(da):
    """
    Compute the mean SST of the two high-latitude regions.
    The Arctic and Antarctic latitude bands are calculated separately and
    then averaged so that both polar regions contribute equally.

    High-latitude regions are defined as:
        60°N to 90°N
        90°S to 60°S

    Parameters:
    da : xarray.Dataset or xarray.DataArray
        SST dataset (containing 'tos') or SST field.

    Returns:
    xarray.DataArray
        Mean of the northern and southern high-latitude SST.
    """
    north = regional_mean_sst(da, 60, 90)
    south = regional_mean_sst(da, -90, -60)
    return 0.5 * (north + south)

def trop_mean_sst(da):
    """
    Compute the mean SST over the tropical latitude band.
    The tropics are defined here as 31°S to 31°N.

    Parameters:
    da : xarray.Dataset or xarray.DataArray
        SST dataset (containing 'tos') or SST field.

    Returns:
    xarray.DataArray
        Area-weighted tropical mean SST.
    """
    trop = regional_mean_sst(da, -31, 31)
    return trop

def global_mean_sst(da):
    """
    Compute the area-weighted global mean SST.

    The function accepts either a Dataset containing 'tos' or an SST
    DataArray. Coordinate names are detected automatically.

    If a time dimension is present, the field is first averaged over time.

    Parameters:
    da : xarray.Dataset or xarray.DataArray
        SST dataset (containing 'tos') or SST field.

    Returns:
    xarray.DataArray
        Area-weighted global mean SST (same units as the input).
    """

    # Automatically extract SST from a Dataset
    if isinstance(da, xr.Dataset):
        if "tos" in da:
            da = da["tos"]
        else:
            raise ValueError("Dataset does not contain 'tos'")

    # Detect coordinate naming convention
    lat_name, lon_name = get_lat_lon_names(da)

    # Detect a possible time dimension
    possible_time_names = ["time", "month", "year", "time_counter", "t"]
    time_name = next((d for d in possible_time_names if d in da.dims), None)

    # Convert a time-dependent field into a climatological mean.
    if time_name is not None:
        da = da.mean(dim=time_name)

    # Apply latitude-based area weighting.
    weights = np.cos(np.deg2rad(da[lat_name]))

    return da.weighted(weights).mean(dim=(lat_name, lon_name))


#===================================================#
# --- SURFACE AIR TEMPERATURE (TAS) DIAGNOSTICS --- #
#===================================================#

def global_mean_tas(ds):
    """
    Compute the area-weighted global mean surface air temperature.
    The function:
    1. Extracts 'tas'
    2. Averages over the available time dimension
    3. Converts Kelvin to Celsius
    4. Computes a cosine-latitude-weighted global mean

    Both 'lat'/'lon' and 'latitude'/'longitude' dimension names are supported.

    Parameters:
    ds : xarray.Dataset
        Dataset containing the 'tas' variable (K).

    Returns:
    numpy.ndarray
        0-d array with the global mean surface air temperature in °C.
    """
    tas = ds["tas"]

    # Average over whichever supported time dimension is present.
    for dim in ["month", "time", "t", "time_counter"]:
        if dim in tas.dims:
            tas = tas.mean(dim)
            break

    # Convert Kelvin to Celsius
    tas = tas - 273.15

    # Area weighting for a regular latitude-longitude grid.
    lat_name, lon_name = get_lat_lon_names(tas)
    weights = np.cos(np.deg2rad(tas[lat_name]))

    gm = tas.weighted(weights).mean((lat_name, lon_name))

    return gm.values


def polar_mean(da, lat_threshold=60):
    """
    Compute an area-weighted mean over both polar regions.
    Unlike high_lat_mean_sst, the two hemispheres are pooled together, so each
    contributes in proportion to its (unmasked) area.

    Parameters:
    da : xarray.DataArray
        DataArray with 'lat'/'lon' or 'latitude'/'longitude' dimensions.

    lat_threshold : float, default=60
        Minimum absolute latitude (degrees) defining the polar regions.

    Returns:
    xarray.DataArray
        Area-weighted mean over both polar regions combined.
    """

    lat_name, lon_name = get_lat_lon_names(da)

    # Retain only grid cells poleward of the selected latitude threshold
    polar = da.where(
        np.abs(da[lat_name]) >= lat_threshold,
        drop=True
    )

    weights = np.cos(np.deg2rad(polar[lat_name]))

    return polar.weighted(weights).mean((lat_name, lon_name))

def calculate_ecs(global_means, co2_levels):
    """
    Estimate equilibrium climate sensitivity (ECS) from a linear CO2 response.

    Global mean temperature is fitted as a linear function of log2(CO2).
    The fitted slope therefore represents the temperature response associated with
    a doubling of atmospheric CO2.

    Parameters:
    global_means : dict[str, float]
        Mapping between experiment names and global mean temperatures (°C).

    co2_levels : dict[str, float]
        Mapping between experiment names and CO2 multipliers relative to the
        reference concentration. Must contain every key of global_means.

    Returns:
    float
        Estimated temperature response per CO2 doubling in °C.
    """

    # Extract CO2 multipliers in the same experiment order as temperatures.
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

    # A CO2 doubling corresponds to an increase of one in log2(CO2).
    x = np.log2(co2)

    # Least-squares linear fit: T = slope * log2(CO2) + intercept
    slope, intercept = np.polyfit(
        x,
        temps,
        1
    )

    # The slope is the temperature change per CO2 doubling.
    return slope

#==========================================#
# ---- PRECIPITATION (PR) DIAGNOSTICS ---- #
#==========================================#

def prepare_pr(da):
    """
    Convert precipitation from kg m-2 s-1 to mm/day.
    The conversion assumes liquid-water density such that:
    1 kg m-2 = 1 mm of water

    Parameters:
    da : xarray.DataArray
        Precipitation rate in kg m-2 s-1.

    Returns:
    xarray.DataArray
        Precipitation rate in mm day-1, with the 'units' attribute updated.
    """

    # Convert per second to per day (86400 s per day)
    da = da * 86400.0

    da.attrs["units"] = "mm/day"
    return (da)

#==========================================#
# RADIATION AND ENERGY BALANCE DIAGNOSTICS #
#==========================================#

def global_toa_ts(ds):
    """
    Compute the annual global mean net TOA radiation time series (W m-2).
    Net top-of-atmosphere radiation is calculated as:
    incoming shortwave - reflected shortwave - outgoing longwave

    Positive values indicate a net gain of energy by the climate system.

    Parameters:
    ds : xarray.Dataset
        Dataset containing 'rsdt', 'rsut', and 'rlut' (W m-2).

    Returns:
    xarray.DataArray
        Annual global mean TOA net radiation in W m-2. One value per year
    """

    # Net radiative flux at the top of the atmosphere
    net_toa = ds["rsdt"] - ds["rsut"] - ds["rlut"]

    # Calculate latitude-based area weights
    # Grid-cell area decreases towards the poles approximately as cos(latitude)
    lat_name, lon_name = get_lat_lon_names(net_toa)
    weights = np.cos(np.deg2rad(net_toa[lat_name]))

    # Global mean per year
    net_gm = net_toa.weighted(weights).mean((lat_name, lon_name))

    return net_gm

def global_surface_net_radiation(ds):
    """
    Compute the annual global mean net surface energy flux (W m-2).

    The surface energy balance is calculated as:
    
    net surface flux =
        surface net solar radiation
        + surface net thermal radiation
        - upward sensible heat flux
        - upward latent heat flux
        - snowfall * latent heat of fusion (334000 J kg-1)

    The last term is the energy needed to melt the falling snow.
    Positive values indicate a net downward energy flux into the surface.

    Parameters:
    ds : xarray.Dataset
        Dataset containing:
        - 'rsns': surface net shortwave radiation (W m-2)
        - 'rlns': surface net longwave radiation (W m-2)
        - 'hfss': upward sensible heat flux (W m-2)
        - 'hfls': upward latent heat flux (W m-2)
        - 'prsn': snowfall flux (kg m-2 s-1)

    Returns:
    xarray.DataArray
        Annual global mean net surface energy flux in W m-2.
    """

    # Net surface energy flux; snowfall (kg m-2 s-1) times the latent heat
    # of fusion (334000 J kg-1) gives the snow-melt energy term in W m-2
    net_sfc = (
        ds["rsns"]
        + ds["rlns"]
        - ds["hfss"]
        - ds["hfls"]
        - ds["prsn"]*334000
    )

    # Latitude-based area weighting
    lat_name, lon_name = get_lat_lon_names(net_sfc)
    weights = np.cos(np.deg2rad(net_sfc[lat_name]))

    # Global mean per year
    net_sfc_gm = (
        net_sfc
        .weighted(weights)
        .mean((lat_name, lon_name))
    )

    return net_sfc_gm

def count_warm_fix_activations(log_file):
        """
        Count the number of times the ECE paleo warm fix was activated.
        Activations are identified by lines containing the string
        "ECE_PALEO_WARM_FIX: ACTIVATED! RESETTING" in the simulation log.

        Parameters:
        log_file : str or pathlib.Path, optional
            Path to the simulation log file. If None, 0 is returned.

        Returns:
        int
            Number of warm-fix activations.
        """
        if log_file is None:
            return 0

        log_file= Path(log_file)

        if not log_file.exists():
            raise FileNotFoundError(
                f"Warm-fix log file not found: {log_file}"
            )
        
        pattern = "ECE_PALEO_WARM_FIX: ACTIVATED! RESETTING"

        # Count matching lines; undecodable characters are ignored
        with log_file.open("r", errors="ignore") as f:
            count = sum(pattern in line for line in f)

        return count

### -------------------------- ###
###     Plotting functions     ###
### -------------------------- ###

#===============================#
# ------- GLOBAL MAPS ----------#
#===============================#

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
    Plot a reference climatology map alongside anomalies from multiple experiments.

    The first panel shows the absolute reference field. All remaining panels
    show anomalies relative to that reference, sharing a single colorbar.
    All panels use a Robinson projection.

    Parameters:
    reference : xarray.DataArray
        2D (lat, lon) reference field.
    anomalies : dict[str, xarray.DataArray]
        Mapping between experiment names and 2D (lat, lon) anomaly fields; at least one is required.
    reference_name : str, default="pi"
        Label for the reference experiment, shown in the first panel title.
    absolute_title : str, default="Mean"
        Title suffix for the reference panel.
    anomaly_title : str, default="Anomaly"
        Title suffix for the anomaly panels.
    absolute_levels, anomaly_levels : int or array-like, optional
        Contour levels for the reference and anomaly fields.
    absolute_cmap, anomaly_cmap : str or matplotlib colormap, default="viridis" / "RdBu_r"
        Colormaps for the reference and anomaly fields.
    absolute_label, anomaly_label : str, default=""
        Colorbar labels.
    figsize_height : float, default=5
        Figure height in inches; the width is 6 inches per panel.
    dpi : int, default=150
        Figure resolution.
    save : str or pathlib.Path, optional
        Output filename. If None, the figure is only shown.
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

    # Make sure axes can be indexed as an array
    if nexp == 1:
        axes = np.asarray(axes)

    # Reference field
    # Add a cyclic longitude point to avoid a visual gap at the map boundary
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

    # Anomaly fields 
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

    # Colorbars: one for the reference panel, one shared by all anomaly panels
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
    Plot a single global climatological field using a Robinson projection.
    A cyclic longitude point is added to prevent a gap at the edge of the global map.

    Parameters:
    field : xarray.DataArray
        Two-dimensional (lat, lon) field.
    experiment : str
        Experiment name, used as a title prefix.
    title : str
        Variable or diagnostic title.
    levels : int or array-like
        Contour levels.
    cmap : str or matplotlib colormap
        Colormap.
    colorbar_label : str
        Label for the colorbar.
    figsize : tuple, default=(10, 5)
        Figure size in inches.
    dpi : int, default=150
        Figure resolution.
    save : str or pathlib.Path, optional
        Output filename. If None, the figure is only shown.
    """

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi,
        constrained_layout=True,
        subplot_kw={"projection": ccrs.Robinson()},
    )

    # Close the longitude seam for global plotting
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

#==============================#
# ------- ZONAL MEANS -------- #
#==============================#

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
    Plot a zonal-mean field as a function of latitude.

    Parameters:
    field : xarray.DataArray
        Latitude-dependent zonal mean (dimension 'lat').
    experiment : str
        Experiment name, used as the legend label.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    save : str or pathlib.Path, optional
        Output filename. If None, the figure is only shown.
    figsize : tuple, default=(9, 5)
        Figure size in inches.
    dpi : int, default=300
        Figure resolution.
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
    Plot zonal-mean anomalies for multiple experiments on the same axes.

    Parameters:
    anomalies : dict[str, xarray.DataArray]
        Mapping between experiment names and zonal anomaly profiles (dimension 'lat').
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    save : str or pathlib.Path, optional
        Output filename. If None, the figure is only shown.
    figsize : tuple, default=(9, 5)
        Figure size in inches.
    dpi : int, default=300
        Figure resolution.
    """

    plt.figure(
        figsize=figsize,
        dpi=dpi
    )

    # Plot each experiment using the same latitude axis
    for exp, da in anomalies.items():

        plt.plot(
            da.lat,
            da,
            linewidth=2.2,
            label=exp
        )

    # Highlight the zero-anomaly reference
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
    """
    Plot the time evolution of a zonal-mean field.

    The resulting contour plot shows latitude versus simulation year.

    Parameters:
    field : xarray.DataArray
        Field with dimensions (time_counter, lat), in that order.
    experiment : str
        Experiment name (currently not used in the plot).
    title : str
        Plot title.
    cmap : str or matplotlib colormap
        Colormap.
    colorbar_label : str
        Colorbar label.
    levels : int or array-like
        Contour levels.
    save : str or pathlib.Path
        Output filename. Required positional argument; pass None to only show the figure.
    figsize : tuple, default=(8, 5)
        Figure size in inches.
    dpi : int, default=150
        Figure resolution.
    """

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi
    )

    # Convert potentially cftime-based coordinates into integer calendar years
    years = field.time_counter.dt.year.values
    lat = field.lat.values

    # Transpose so contourf receives dimensions as (lat, time)
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



#==================================#
# -------- CO2 RESPONSE ---------- #
#==================================#

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
    Plot global mean climate values as a function of CO2 concentration.

    Each experiment is plotted at its corresponding CO2 multiplier relative to the pre-industrial reference.

    Optional ECS and polar amplification diagnostics can be included in the legend.

    Parameters:
    values : dict[str, float or xarray.DataArray]
        Mapping between experiment names and scalar global mean values
        (e.g. global mean temperature in °C).

        Example:
        {
            "pi": 14.5,
            "x3": 22.3,
            "x6": 31.1
        }

    co2_levels : dict[str, float]
        Mapping between experiment names and CO2 multipliers.

        Example:
        {
            "pi": 1,
            "x3": 3,
            "x6": 6
        }

    ecs : float, optional
        Equilibrium climate sensitivity (°C) to display in the legend.
    polar_amp : dict[str, float], optional
        Polar amplification factor for each non-reference experiment, displayed in the legend.
    ylabel : str, default="Global mean temperature (°C)"
        Y-axis label.
    xlabel : str, default="CO₂ (× pre-industrial)"
        X-axis label.
    title : str, optional
        Plot title.
    ylim : tuple, optional
        Y-axis limits.
    xlim : tuple, optional
        X-axis limits.
    figsize : tuple, default=(6, 4)
        Figure size in inches.
    dpi : int, default=300
        Figure resolution.
    save : str or pathlib.Path, optional
        Output filename. If None, the figure is only shown.
    """

    plt.figure(
        figsize=figsize,
        dpi=dpi
    )

    # One point per experiment at its CO2 multiplier
    for exp, value in values.items():

        co2 = co2_levels[exp]

        value = float(value)

        plt.scatter(
            co2,
            value,
            s=80,
            label=f"{exp}: {value:.2f}"
        )

        # Label the experiment directly above its point.
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

    # Use the available CO2 multipliers as x-axis ticks.
    plt.xticks(
        sorted(set(co2_levels.values()))
    )

    # Legend
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

    # Add polar amplification values as text-only legend entries
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

    Because CO2 forcing scales approximately with the logarithm of CO2
    concentration, plotting climate variables against log2(CO2) provides a
    convenient way to examine approximately linear responses.
    Experiments missing from logco2_levels or with non-finite values are skipped.

    Parameters:
    values : dict[str, float or xarray.DataArray]
        Mapping between experiment names and scalar global mean values.
    logco2_levels : dict[str, float]
        Mapping between experiment names and log2 of their CO2 multiplier.
    ylabel : str, default="Global mean precipitation (mm/day)"
        Y-axis label.
    xlabel : str, default="log₂(CO₂)"
        X-axis label.
    title : str, optional
        Figure title.
    ylim : tuple, optional
        Y-axis limits.
    xlim : tuple, optional
        X-axis limits.
    figsize : tuple, default=(6, 4)
        Figure size in inches.
    dpi : int, default=300
        Figure resolution.
    save : str or pathlib.Path, optional
        Output filename. If None, the figure is only shown.

    Returns:
    slope : float
        Regression slope (units of the values per CO2 doubling).

    intercept : float
        Regression intercept (value at log2(CO2) = 0, i.e. the reference CO2).
    """

    experiments = []
    x = []
    y = []

    # Collect valid experiment/value pairs.
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

    # Fit a first-order response to log2(CO2).
    slope, intercept = np.polyfit(x, y, 1)

    # Regression line evaluated over the range of the data
    x_fit = np.linspace(
        x.min(),
        x.max(),
        100
    )

    y_fit = slope * x_fit + intercept

    # Plot
    plt.figure(
        figsize=figsize,
        dpi=dpi
    )

    # Plot experiment values
    for exp, xi, yi in zip(experiments, x, y):

        plt.scatter(
            xi,
            yi,
            s=80,
            label=f"{exp}: {yi:.2f}"
        )

        # Label each point slightly above it (2% of the data range)
        plt.text(
            xi,
            yi + 0.02 * (y.max() - y.min()),
            exp,
            ha="center"
        )

    # Plot fitted linear response
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

    # Axis limits
    if xlim is not None:
        plt.xlim(*xlim)

    if ylim is not None:
        plt.ylim(*ylim)

    # X ticks at every integer log2(CO2), i.e. every CO2 doubling
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

    # Labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    # Legend
    plt.legend(
        title="Global mean"
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

    return slope, intercept

#======================================#
# -------- GLOBAL TIME SERIES -------- #
#======================================#

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

    Parameters:
    series : dict[str, xarray.DataArray]
        Mapping between experiment names and 1D annual time series (dimension 'time_counter').
    ylabel : str
        Label for the y-axis.
    title : str
        Figure title.
    use_model_years : bool, default=False
        If True, use calendar/model years from 'time_counter'; if False, number years from 1.
    xlabel : str, default="Simulation year"
        Label for the x-axis.
    figsize : tuple, default=(10, 5)
        Figure size in inches.
    dpi : int, default=150
        Figure resolution.
    save : str or pathlib.Path, optional
        Output filename. If None, the figure is only shown.
    """

    plt.figure(figsize=figsize, dpi=dpi)

    for exp, da in series.items():

        # Choose the x-axis: model calendar years or sequential year index
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

#=======================================#
# ----- TOA AND RADIATION BALANCE ----- #
#=======================================#

def plot_gregory(
    tas_series,
    toa_series,
    experiment_labels=None,
    colors=None,
    f_years=None,
    deepmip_tas_ranges=None,
    xlabel="Global mean TAS (°C)",
    ylabel="TOA net radiation (W m⁻²)",
    title="Gregory plots",
    figsize=(7, 6),
    dpi=150,
    save=None,
):
    """
    Create Gregory plots for one or more climate experiments.

    A Gregory plot shows the relationship between global mean surface temperature
    and net TOA radiation. A linear regression is fitted to each experiment to characterize
    its radiative response.

    The first (circle) and last (square) simulation years are highlighted separately.
    Years with missing values are removed before plotting and fitting.

    Parameters:
    tas_series : dict[str, xarray.DataArray]
        Annual global mean surface temperature time series (°C), dimension 'time_counter'.
    toa_series : dict[str, xarray.DataArray]
        Corresponding annual global mean net TOA radiation time series (W m-2); same keys as tas_series.
    experiment_labels : dict[str, str], optional
        Display labels for the legend. If None, experiment names are used.
    colors : dict[str, str], optional
        Mapping between experiment names and plot colors. If None, the Matplotlib color cycle is used.
    f_years : int, optional
        If given, only the first f_years years of each experiment are used.
    deepmip_tas_ranges : dict[str, tuple[float, float]], optional
        DeepMIP (min, max) global mean TAS range per CO2 level, drawn as shaded vertical bands.
    xlabel : str, default="Global mean TAS (°C)"
        X-axis label.
    ylabel : str, default="TOA net radiation (W m⁻²)"
        Y-axis label.
    title : str, default="Gregory plots"
        Figure title.
    figsize : tuple, default=(7, 6)
        Figure size in inches.
    dpi : int, default=150
        Figure resolution.
    save : str or pathlib.Path, optional
        Output filename. If None, the figure is only shown.
    """

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi
    )

    # Use internal experiment names if no display labels are provided
    if experiment_labels is None:
        experiment_labels = {
            exp: exp
            for exp in tas_series
        }

    # Assign default Matplotlib colors if none are supplied
    if colors is None:
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        colors = {
            exp: color_cycle[i % len(color_cycle)]
            for i, exp in enumerate(tas_series)
        }

    # Display labels for the DeepMIP TAS range bands
    deepmip_labels = {
        "pi": "PI",
        "1x": "1×CO₂",
        "3x": "3×CO₂",
        "6x": "6×CO₂",
        "9x": "9×CO₂",
    }

    # Shade the DeepMIP TAS range for each CO2 level, labelled at the top of the axes
    if deepmip_tas_ranges is not None:
        for exp, (tas_min, tas_max) in deepmip_tas_ranges.items():

            ax.axvspan(
                tas_min,
                tas_max,
                alpha=0.2,
                zorder=0
            )

            ax.text(
                (tas_min + tas_max) / 2,
                0.98,
                deepmip_labels.get(exp, exp),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top"
            )

    # Loop over the experiments
    for k in tas_series:

        tas = tas_series[k]
        toa = toa_series[k]

        # Restrict the analysis to the first N years if requested
        if f_years is not None:

            tas = tas.isel(
                time_counter=slice(0, f_years)
            )

            toa = toa.isel(
                time_counter=slice(0, f_years)
            )

        # Convert directly to numpy arrays
        x = tas.values
        y = toa.values

        # TAS and TOA must represent the same years
        if len(x) != len(y):
            raise ValueError(
                f"{k}: TAS has {len(x)} points, "
                f"TOA has {len(y)} points"
            )

        # Remove years containing missing values (NaNs)
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

        # Linear Gregory regression: N = slope * T + intercept
        # (slope is the radiative feedback parameter, in W m-2 °C-1)
        slope, intercept = np.polyfit(x, y, 1)

        x_fit = np.linspace(
            x.min(),
            x.max(),
            100
        )

        y_fit = slope * x_fit + intercept

        color = colors[k]

        # Connect points to show the temporal evolution of the simulation.
        ax.plot(
            x,
            y,
            color=color,
            alpha=0.6,
            linewidth=1
        )

        # Annual values
        ax.scatter(
            x,
            y,
            color=color,
            s=25,
            label=experiment_labels.get(k, k)
        )

        # Mark the first simulation year
        ax.scatter(
            x[0],
            y[0],
            color=color,
            s=100,
            marker="o",
            edgecolor="k",
            zorder=3
        )

        # Mark the last simulation year
        ax.scatter(
            x[-1],
            y[-1],
            color=color,
            s=100,
            marker="s",
            edgecolor="k",
            zorder=3
        )

        # Regression line
        ax.plot(
            x_fit,
            y_fit,
            color=color,
            linestyle="--"
        )

    # Reference line representing radiative equilibrium.
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


def plot_radiation_balance(
    difference,
    experiment_labels=None,
    colors=None,
    f_years=None,
    warm_fix_activations=None,
    moving_mean=None,
    title=None,
    save=None
):
    """
    Plot the distribution and time evolution of the TOA-SFC net energy flux difference.

    The figure contains two panels:
        Top:
            Boxplots showing the distribution of annual global mean
            TOA-SFC net energy flux for each experiment.
        Bottom:
            Annual time series of the TOA-SFC net energy flux difference,
            optionally with a centred moving mean.

    Parameters:
    difference : dict[str, xarray.DataArray]
        Annual global mean TOA minus surface net energy flux (W m-2), dimension 'time_counter'.
    experiment_labels : dict[str, str], optional
        Display labels. If None, experiment names are used.
    colors : dict[str, str], optional
        Mapping between experiment names and colors. If None, the Matplotlib color cycle is used.
    f_years : int, optional
        If given, the boxplots use only the first f_years years (the time series panel shows all years).
    warm_fix_activations : dict[str, int], optional
        Number of warm-fix activations per experiment, appended to the boxplot labels.
    moving_mean : int, optional
        Window length (years) of a centred moving mean added to the time series panel.
    title : str, optional
        Overall figure title.
    save : str or pathlib.Path, optional
        Output filename. If None, the figure is only shown.
    """

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=False,
    )

    # Experiment labels and colors
    experiments = list(difference.keys())
    
    # Use internal experiment names if no display labels are provided
    if experiment_labels is None:
        experiment_labels = {
            exp: exp
            for exp in difference
        }

    # Use Matplotlib's default color cycle if no colors are provided
    if colors is None:
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        colors = {
            exp: color_cycle[i % len(color_cycle)]
            for i, exp in enumerate(experiments)
        }

    # ==========================================================
    # Top panel: distributions of the TOA-SFC net energy flux
    # ==========================================================

    difference_data = []
    labels = []

    for exp in experiments:

        values = difference[exp]

        # Boxplots use only the requested f_years
        if f_years is not None:
            values = values.isel(
                time_counter=slice(0,f_years)
            )

        # Convert to numpy arrays and remove NaNs
        values = values.values
        values = values[np.isfinite(values)]

        difference_data.append(values)

        # Build the tick label, adding the warm-fix count when available
        label = experiment_labels.get(exp, exp)

        if warm_fix_activations is None:
            warm_fix_activations = {}

        warm_fix_count = warm_fix_activations.get(exp)

        if warm_fix_count is not None:
            label = f"{label}\nWarm fix: {warm_fix_count}"

        labels.append(label)

    # Create one boxplot for each experiment so that each experiment can have its own color
    for i, (exp, values) in enumerate(
        zip(experiments, difference_data),
        start=1
    ):
    
        axes[0].boxplot(
            [values],
            positions=[i],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=colors[exp], alpha=0.7),
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color=colors[exp]),
            capprops=dict(color=colors[exp]),
            flierprops=dict(
                marker="o",
                markerfacecolor=colors[exp],
                markeredgecolor=colors[exp],
                markersize=4,
                alpha=0.6
            )
        )

    # Zero represents no TOA-SFC difference (energy consistency)
    axes[0].axhline(
        0,
        linestyle=":",
        linewidth=1,
    )

    axes[0].set_ylabel(
        "Net radiation (W m$^{-2}$)"
    )

    axes[0].set_title(
        "Distribution of TOA-SFC net energy flux"
    )

    # One tick per experiment, centred on its box
    axes[0].set_xticks(
        np.arange(1, len(experiments)+1)
    )

    axes[0].set_xticklabels(labels)

    axes[0].grid(
        True,
        linestyle="--",
        alpha=0.4,
        axis="y"
    )

    # ==========================================================
    # Bottom panel: time series of TOA minus surface net flux
    # ==========================================================
    for exp in experiments:

        years = difference[exp]["time_counter"].dt.year
        label_b = experiment_labels.get(exp, exp)

        # Original annual time series
        axes[1].plot(
            years,
            difference[exp],
            color=colors[exp],
            alpha=0.4,
            linewidth=1,
            label=label_b,
        )

        # Centred moving mean over moving_mean years
        if moving_mean is not None:
            moving_mean_data = (
                difference[exp]
                .rolling(
                    time_counter=moving_mean,
                    center=True
                )
                .mean()
            )

            axes[1].plot(
                years,
                moving_mean_data,
                color=colors[exp],
                linewidth=2.5,
                label=(
                    f"{experiment_labels.get(exp,exp)} "
                    f"({moving_mean}-yr moving mean)"
                )
            )

    # Zero represents no TOA-SFC difference
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
        "TOA − SFC net energy flux"
    )

    axes[1].legend()

    axes[1].grid(
        True,
        linestyle="--",
        alpha=0.4,
        axis="y"
    )

    # Figure title and layout
    if title is not None:
        fig.suptitle(title)
    
    plt.tight_layout()

    if save is not None:
        plt.savefig(
            save, 
            dpi=300,
            bbox_inches="tight"),
    plt.show()
    plt.close()

#====================================#
# ---- SST GRADIENT DIAGNOSTICS ---- #
#====================================#

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

    The meridional gradient is typically defined as the difference between
    tropical and high-latitude SST.
    
    The reference experiment, identified by `reference`, is drawn with a
    black marker. All experiments present in both input dictionaries are
    plotted automatically.

    Parameters:
    global_mean_sst : dict[str, float]
        One scalar global mean SST value (°C) per experiment.
    meridional_gradient : dict[str, float]
        One scalar meridional SST gradient (°C) per experiment.
    reference : str, optional
        Name of the reference experiment. If None, no experiment is highlighted.
    xlabel : str, default="Global mean SST (°C)"
        Label for the x-axis.
    ylabel : str, default="Tropics – High latitude SST (°C)"
        Label for the y-axis.
    title : str, default="Meridional SST gradient vs global mean SST"
        Figure title.
    figsize : tuple, default=(6, 4)
        Figure size in inches.
    dpi : int, default=150
        Figure resolution.
    ylim : tuple, optional
        Y-axis limits.
    xlim : tuple, optional
        X-axis limits.
    annotate : bool, default=True
        Whether experiment names should be displayed next to points.
    save : str or pathlib.Path, optional
        Output filename. If None, the figure is only shown.
    """

    # Retain only experiments available in both dictionaries
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


#==================================#
# --- DEEPMIP COMPARISON PLOTS --- #
#==================================#

def plot_global_mean_vs_co2_dp(
    data,
    title,
    save,
    model_styles,
    figsize=(8, 5),
    dpi=150,
):
    """
    Plot global mean surface temperature against CO2 concentration for multiple models.

    This function is designed for comparisons with DeepMIP-style multi-model datasets.
    Each model receives a consistent color and marker, and its experiments are
    connected in order of increasing CO2. Pre-industrial points are drawn smaller.

    Parameters:
    data : list of dict
        Plotting data, one dictionary per model experiment, with keys
        "model" (str), "CO2" (CO2 multiplier), "T" (global mean temperature, °C)
        and "pi" (bool, True for the pre-industrial run).
    title : str
        Figure title.
    save : str or pathlib.Path
        Output filename. Required positional argument; pass None to only show the figure.
    model_styles : dict[str, dict]
        Mapping between model names and {"color": ..., "marker": ...} (see DEEP_MIP_MODEL_STYLES).
    figsize : tuple, default=(8, 5)
        Figure size in inches.
    dpi : int, default=150
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

        # Sort experiments by CO2 concentration before connecting them.
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

        # Connecting line. Connect experiments belonging to the same model.
        ax.plot(
            xs,
            ys,
            color=style["color"],
            linewidth=1.5,
            zorder=2,
        )

        for d in model_data:

            # Make the pre-industrial reference smaller than perturbation runs.
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

    # Create one legend entry per model using empty proxy scatter points
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


def plot_sst_gradient_vs_global_sst_dp(
    data,
    title,
    save,
    model_styles,
    figsize=(8, 5),
    dpi=150,
):
    """
    Plot meridional SST gradient against global mean SST for multiple models.

    This function is intended for comparison with DeepMIP-style multi-model results.

    Each model is represented using a consistent color and marker. Experiments belonging to the
    same model are connected (in order of increasing global mean SST) to highlight their
    response across climate states. Pre-industrial points are drawn smaller.

    Parameters:
    data : list of dict
        One dictionary per model experiment, with keys "model" (str), "x" (global mean SST, °C),
        "y" (tropics minus high-latitude SST, °C), "exp" (experiment label) and "pi" (bool).
    title : str
        Figure title.
    save : str or pathlib.Path
        Output filename. Required positional argument; pass None to only show the figure.
    model_styles : dict[str, dict]
        Mapping between model names and {"color": ..., "marker": ...} (see DEEP_MIP_MODEL_STYLES).
    figsize : tuple, default=(8, 5)
        Figure size in inches.
    dpi : int, default=150
        Figure resolution.
    """

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi,
    )

    models = sorted(
        set(d["model"] for d in data)
    )

    # Plot experiment points
    for d in data:

        style = model_styles[d["model"]]

        # Use larger symbols for non-pre-industrial experiments
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

        # Label individual experiments
        ax.text(
            float(d["x"]) + 0.1,
            float(d["y"]) + 0.1,
            d["exp"],
            fontsize=8,
            zorder=4,
        )

    # Connecting lines. Connect experiments from the same model
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

    # Labels
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

    # Model legend built from proxy marker-only handles
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


#============================#
# ------ MODEL STYLES ------ #
#============================#

# Consistent visual identity for DeepMIP comparison figures.

# Using a single dictionary ensures that each model has the
# same appearance (color and marker) across all plots in the analysis.

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

#==============================#
# -------- MOC plots ----------#
#==============================#

def plot_moc_mean(
    amoc_mean,
    experiment=None,
    experiment_label=None,
    levels=None,
    title=None,
    save_path=None,
):
    """
    Plot the time-mean Atlantic meridional overturning circulation.
    The streamfunction (Sv) is shown as filled contours of latitude versus depth,
    over the upper 3000 m.

    Parameters:
    amoc_mean : xarray.DataArray
        Time-mean MOC streamfunction (Sv) with coordinates 'nav_lat' and 'depthw' (m).
    experiment : str, optional
        Internal experiment name; used as fallback for experiment_label.
    experiment_label : str, optional
        User-facing experiment label (currently not used in the plot).
    levels : array-like, optional
        Contour levels. If None, defaults to np.arange(-17, 17, 2).
    title : str, optional
        Plot title.
    save_path : str or pathlib.Path, optional
        Output filename. If None, the figure is only shown.
    """

    if levels is None:
        levels = np.arange(-17, 17, 2)

    if experiment_label is None:
        experiment_label = experiment

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    amoc_mean.plot.contourf(
        ax=ax,
        x="nav_lat",
        y="depthw",
        levels=levels,
        cmap="RdBu_r",
        add_colorbar=True,
        cbar_kwargs={
            "label": "MOC (Sv)"
        },
    )

    # Depth axis inverted (surface at the top), limited to the upper 3000 m
    ax.set_ylim(3000, 0)
    ax.set_xlabel("Latitude (°N)")
    ax.set_ylabel("Depth (m)")

    ax.set_title(title)

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()
    plt.close()

def plot_moc_timeseries(
    amoc_timeseries,
    colors=None,
    experiment_labels=None,
    plot_dir=None,
    moving_mean=None,
    ylim=None,
    title=None,
    save_path=None,
):
    """
    Plot the annual AMOC maximum time series for one or more experiments.

    Parameters:
    amoc_timeseries : dict[str, xarray.DataArray]
        Mapping between experiment names and annual AMOC maximum (Sv), dimension 'time_counter'.
    colors : dict[str, str], optional
        Mapping between experiment names and colors. If None, Matplotlib defaults are used.
    experiment_labels : dict[str, str], optional
        Display labels for the legend. If None, experiment names are used.
    plot_dir : str or pathlib.Path, optional
        Currently not used; use save_path to save the figure.
    moving_mean : int, optional
        Width of a centred moving mean in years. If None, no moving mean is plotted.
    ylim : tuple, optional
        Y-axis limits.
    title : str, optional
        Plot title.
    save_path : str or pathlib.Path, optional
        Output filename. If None, the figure is only shown.
    """

    if colors is None:
        colors = {}

    if experiment_labels is None:
        experiment_labels = {}

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    # Plot each experiment
    for exp, data in amoc_timeseries.items():

        color=colors.get(exp, None)
        label=experiment_labels.get(exp,exp)

        # ------------------------------------------------------
        # Annual AMOC
        # ------------------------------------------------------

        data.plot(
            ax=ax,
            color=color,
            linewidth=1.0,
            alpha=0.5,
            label=f"{label} annual",
        )

        # ------------------------------------------------------
        # Centred moving mean over moving_mean years
        # ------------------------------------------------------

        if moving_mean is not None:

            amoc_smooth = (
                data
                .rolling(
                    time_counter=moving_mean,
                    center=True,
                )
                .mean()
            )

            amoc_smooth.plot(
                ax=ax,
                color=color,
                linewidth=2.0,
                label=f"{label} ({moving_mean}-year mean)",
            )

    # ----------------------------------------------------------
    # Formatting
    # ----------------------------------------------------------


    ax.set_xlabel("Year")
    ax.set_ylabel("AMOC maximum (Sv)")

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_title(title)

    ax.legend(
        frameon=False,
        ncol=1
    )

    fig.tight_layout()

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()
    plt.close()

def plot_amoc_gregory(
    tas_series,
    amoc_series,
    experiment_labels=None,
    colors=None,
    f_years=None,
    xlabel="Global mean TAS (°C)",
    ylabel="Annual maximum AMOC strength (Sv)",
    title="AMOC–TAS relationship",
    figsize=(7, 6),
    dpi=150,
    save=None,
):
    """
    Create AMOC–TAS relationship plots for one or more climate experiments.

    The plot shows the relationship between global mean surface temperature
    and annual maximum AMOC strength. A linear regression is fitted to each
    experiment to characterize the temperature dependence of AMOC strength.

    The first (circle) and last (square) simulation years are highlighted separately.
    Years with missing values are removed before plotting and fitting.

    Parameters:
    tas_series : dict[str, xarray.DataArray]
        Annual global mean surface temperature time series (°C), dimension 'time_counter'.
    amoc_series : dict[str, xarray.DataArray]
        Corresponding annual maximum AMOC strength time series (Sv); same keys as tas_series.
    experiment_labels : dict[str, str], optional
        Mapping between internal experiment names and display labels.
    colors : dict[str, str], optional
        Mapping between experiment names and plot colors. If None, the Matplotlib color cycle is used.
    f_years : int, optional
        If given, only the first f_years years of each experiment are used.
    xlabel : str, default="Global mean TAS (°C)"
        X-axis label.
    ylabel : str, default="Annual maximum AMOC strength (Sv)"
        Y-axis label.
    title : str, default="AMOC–TAS relationship"
        Figure title.
    figsize : tuple, default=(7, 6)
        Figure size in inches.
    dpi : int, default=150
        Figure resolution.
    save : str or pathlib.Path, optional
        Output filename. If None, the figure is only shown.
    """

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi,
    )

    # Use internal experiment names if no display labels are provided
    if experiment_labels is None:
        experiment_labels = {
            exp: exp
            for exp in tas_series
        }

    # Assign default Matplotlib colors if none are supplied
    if colors is None:

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        colors = {
            exp: color_cycle[i % len(color_cycle)]
            for i, exp in enumerate(tas_series)
        }

    # Loop over experiments
    for exp in tas_series:

        tas = tas_series[exp]
        amoc = amoc_series[exp]

        # Restrict to the first N years if requested
        if f_years is not None:

            tas = tas.isel(
                time_counter=slice(0, f_years)
            )

            amoc = amoc.isel(
                time_counter=slice(0, f_years)
            )

        # Convert to numpy arrays
        x = tas.values
        y = amoc.values

        # TAS and AMOC must represent the same years
        if len(x) != len(y):
            raise ValueError(
                f"{exp}: TAS has {len(x)} points, "
                f"AMOC has {len(y)} points"
            )

        # Remove years containing missing values
        valid = np.isfinite(x) & np.isfinite(y)

        x = x[valid]
        y = y[valid]

        print(
            f"{exp}: {len(x)} valid points"
        )

        if len(x) < 2:
            raise ValueError(
                f"{exp}: not enough valid data points "
                f"for regression."
            )

        # Linear regression of AMOC strength on TAS (slope in Sv °C-1)
        slope, intercept = np.polyfit(
            x,
            y,
            1,
        )

        x_fit = np.linspace(
            x.min(),
            x.max(),
            100,
        )

        y_fit = slope * x_fit + intercept

        color = colors[exp]

        # Connect annual values to show temporal evolution
        ax.plot(
            x,
            y,
            color=color,
            alpha=0.6,
            linewidth=1,
        )

        # Annual values
        ax.scatter(
            x,
            y,
            color=color,
            s=25,
            label=experiment_labels.get(exp, exp),
        )

        # First simulation year
        ax.scatter(
            x[0],
            y[0],
            color=color,
            s=100,
            marker="o",
            edgecolor="k",
            zorder=3,
        )

        # Last simulation year
        ax.scatter(
            x[-1],
            y[-1],
            color=color,
            s=100,
            marker="s",
            edgecolor="k",
            zorder=3,
        )

        # Regression line
        ax.plot(
            x_fit,
            y_fit,
            color=color,
            linestyle="--",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    ax.legend(
        title="Experiment"
    )

    fig.tight_layout()

    if save is not None:
        fig.savefig(
            save,
            bbox_inches="tight",
        )

    plt.show()