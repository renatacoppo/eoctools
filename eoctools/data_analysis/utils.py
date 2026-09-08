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
    Open a dataset and retain only the last nyears complete simulation years.
    This is useful for equilibrium or quasi-equilibrium analysis whete the climate 
    state is represented by the final part of a long simulation.

    Parameters
    file_path : str ot Path
                Path to the NetCDF dataset
    nyears : int, default=100
            Number of years to retain
    time_dim : str, default="time_counter"
            Name of the time coordinate.
    end_year : int, optional
            Final year to include. If omitted, the last 
            available year is used

    Returs
    xarray.Dataset
        Dataset sliced from January 1 of the first selected year 
        through December of the final selected year
    """
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
    Open a simulation while excluding an initial spin-up period.
    The first "skip_years" are removed because climate simulations often need an adjustment
    period before their statistics are representative of the simulated climate state.

    Parameters
    file_path : str ot Path
            Path to the NetCDF dataset
    skip_years : int, default=10
            Number of initial simulations years to discard
    time_dim : str, default="time_counter"
            Name of the time coordinate.
    end_year : int, optional
            Final year to include. If omitted, the last
            available year is used

    Returs
    xarray.Dataset
        Dataset excluding the initial spin-up period
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


#======================================#
# ----- GENERAL SPATIAL AVERAGING ---- #
#======================================#

# global mean
def global_mean(da): 
    """
    Compute an area-weighted global mean.
    Latitude weighting using cos (latitude) approximates the changing area 
    of lat-lon grid cells toward the poles.

    Parameters
    da: xarray.DataArray
        Field with dimensions including 'lat' and 'lon'.
    
    Returns
    xarray.DataArray
        Area-weighted global mean.
    """

    weights = np.cos(np.deg2rad(da.lat))
    return da.weighted(weights).mean(("lat", "lon"))

def zonal_mean(da):
    """
    Compute latitude-weighted zonal mean.
    The longitude dimension is averaged, leaving a latitude-dependent profil

    Parameters
    da : xarray.DataArray
        Field containing longitude and latitude dimensions.

    Returns
    xarray.DataArray
        Zonal mean as a function of latitude
    """

    # Latitude weights are included for consistency with other spatial averaging functions,
    # although they do not affect a simple longitude mean.

    weights = np.cos(np.deg2rad(da.lat))
    weights = weights / weights.mean()

    return da.weighted(weights).mean("lon")

#===========================================#
# SEA SURFACE TEMPERATURE (SST) DIAGNOSTICS #
#===========================================#

def regional_mean_sst(da, lat_min, lat_max):
    """
    Compute the area-weighted mean SST over a latitude band.
    The function accepts either an xarray Dataset containing 'tos'
    or a DataArray. It also supports several common names for latitude,
    longitude and time dimensions.
    
    Parameters
    da : xarray.Dataset or xarray.DataArray
        SST dataset of field

    lat_min : float
        Southern boundary of the latitude band

    lat_max : float
        Northern boundary of the latitude band

    Returns:
    xarray.DataArray
        Area-weighted mean SST over the selected latitude band.
    
    """

    # If a Dataset is provided, automatically extract sea surface temperature.
    if isinstance(da, xr.Dataset):
        if "tos" in da:
            da = da["tos"]
        else:
            raise ValueError("Dataset does not contain 'tos'")

    # Support both common CMIP coordinate naming conventions
    lat_name = "lat" if "lat" in da.dims else "latitude"
    lon_name = "lon" if "lon" in da.dims else "longitude"

    # Identify the time dimension if one is persent
    possible_time_names = ["time", "month", "year", "time_counter", "t"]
    time_name = next((d for d in possible_time_names if d in da.dims), None)

    lat = da[lat_name]

    # Latitude coordinates may run north-to-south ot south-to-north
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
    Ther Arctic and Antarctic latitude bands are calculated separatedly and
    then averaged so that both polar regions contribute equally.

    High-latitude regions are defined as:
        60°N to 90°N
        90°S to 60°S
    """
    north = regional_mean_sst(da, 60, 90)
    south = regional_mean_sst(da, -90, -60)
    return 0.5 * (north + south)

def trop_mean_sst(da):
    """
    Compute the mean SST over the tropical latitude band.
    The tropics are defined here as 31°S to 31°N
    """
    trop = regional_mean_sst(da, -31, 31)
    return trop

def global_mean_sst(da):
    """
    Compute the area-weighted global mean SST.

    The function accepts either a Dataset containing 'tos' or an SST
    DataArray. Coordinate names are detected automatically.

    If a time dimension is present, the field is first averaged over time.

    Parameters
    da : xarray.DataArray
        SST dataset or field

    Returns
    xarray.DataArray
        Area-weighted global mean SST
    """

    # Automatically extract SST from a Dataset
    if isinstance(da, xr.Dataset):
        if "tos" in da:
            da = da["tos"]
        else:
            raise ValueError("Dataset does not contain 'tos'")

    # Detect coordinate naming convention
    lat_name = "lat" if "lat" in da.dims else "latitude"
    lon_name = "lon" if "lon" in da.dims else "longitude"

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
    2. Averaged over the available time dimension
    3. Converts Kelvin to Celsius
    4. Computes a cosine-latitude-weighted global mena

    Parameters
    ds : xarray.Dataset
        Dataset containing the 'tas' variable

    Returns
    numpy scalar
        Global mean surface air temperature in °C
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
    weights = np.cos(np.deg2rad(tas.latitude))

    gm = tas.weighted(weights).mean(("latitude", "longitude"))

    return gm.values


def polar_mean(da, lat_threshold=60):
    """
    Compute an area-weighted mean over both polar regions.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with lat/lon dimensions.

    lat_threshold : float, default=60
        Minimum absolute latitude defining the polar region.

    Returns
    -------
    xarray.DataArray
        Area-weighted mean over both polar regions combined.
    """

    # Retain only grid cells poleward of the selected latitude threshold
    polar = da.where(
        np.abs(da.lat) >= lat_threshold,
        drop=True
    )

    weights = np.cos(np.deg2rad(polar.lat))

    return polar.weighted(weights).mean(("lat", "lon"))

def calculate_ecs(global_means, co2_levels):
    """
    Estimate equilibrium climate sensitivity (ECS) from a linear CO2 response.

    Global mean temperature is fitted as a linear function of log2(CO2).
    The fitted slope therefore represents the temperature response associate with
    the doubling of atmospheric CO2.

    Parameters
    global_means : dict
        Mapping between experiment names and global mean temperatures.

    co2_levels : dict
        Mapping between experiment names and CO2 multipliers relative to the
        reference concentration.

    Returns
    -------
    ecs : float
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

    # A cO2 doubling corresponds to an increase of one in log2(CO2).
    x = np.log2(co2)

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
    1 kg m⁻² = 1 mm of water

    Parameters
    da : xarray.DataArray
        Precipitation rate in kg m-2 s-1

    Returns
    xarray.DataArray
        Precipitation rate in mm day-1.
    """

    # Convert seconds to days
    da = da * 86400.0

    da.attrs["units"] = "mm/day"
    return (da)

#==========================================#
# RADIATION AND ENERGY BALANCE DIAGNOSTICS #
#==========================================#

def global_toa_ts(ds):
    """
    Compute the annual global mean net TOA radiation time series (W m-2)
    Net top-of-atmosphere radiation is calculated as:
    incoming shortwave - reflected shortwave - outgoing longwave

    Positive values indicate a net gain of energy by the climate system.

    Parameters
    ds : xarray.Dataset
        Dataset containing 'rsdt', 'rsut', and 'rlut'.

    Returns
    xarray.DataArray
        Annual global mean TOA net radiation in W m-2. One value per year
    """

    # Net radiative flux at the top of the atmosphere
    net_toa = ds["rsdt"] - ds["rsut"] - ds["rlut"]

    # Calculate latitude-based area weights
    # Grid-cell area decreases towards the poles approximately as cos(latitude)
    weights = np.cos(np.deg2rad(net_toa.lat))

    # Global mean per year
    net_gm = net_toa.weighted(weights).mean(("lat", "lon"))

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
    
    Positive values indicate a net downward energy flux into the surface.

    Parameters
    ds : xarray.Dataset
        Dataset containing: 
        - 'rsns': surface net shortwave radiation 
        - 'rlns': surface net longwave radiation 
        - 'hfss': upward sensible heat flux 
        - 'hfls': upward latent heat flux

    Returns
    xarray.DataArray
        Annual global mean net surface energy flux in W m-2.
    """

    # Net surface energy flux
    net_sfc = (
        ds["rsns"]
        + ds["rlns"]
        - ds["hfss"]
        - ds["hfls"]
    )

    # Calculate latitude-based area weights
    # Latitude-based area weighting
    weights = np.cos(np.deg2rad(net_sfc.lat))

    # Global mean per year
    net_sfc_gm = (
        net_sfc
        .weighted(weights)
        .mean(("lat", "lon"))
    )

    return net_sfc_gm

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
    show anomalies relative to that reference.

    Parameters
    ----------
    reference : xarray.DataArray
        2D (lat, lon) reference field.

    anomalies : dict[str, xarray.DataArray]
        Dictionary of anomaly fields.

    reference_name : str
        Label for the reference experiment. Name displayed in first panel.

    absolute_title : str
        Title subfix for the reference field.

    anomaly_title : str
        Title suffix for anomaly panels.

    absolute_levels, anomaly_levels : array-like, optional
        Contour levels for absolute anomaly and anomaly fields.

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

    # Colorbars 
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
    
    Parameters
    ----------
    field : xarray.DataArray
        Two-dimensional (lat, lon) field.

    experiment : str
        Experiment name

    title : str
        Variable or diagnostic title.

    levels : array-like
        Contour levels.

    cmap : str or matplotlib colormap

    colorbar_label : str
        Label for the colorbar

    save : str or Path, optional
        Output filename
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

    Parameters
    ----------
    field : xarray.DataArray
        Latitude-dependent zonal mean.

    experiment : str
        Experiment name.

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
    Plot zonal-mean anomalies for multiple experiments

    Parameters
    ----------
    anomalies : dict[str, xarray.DataArray]
        Mapping between experiment names and zonal anomaly profiles.

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

    Parameters
    field: xarray.DataArray
        Field with dimensions (time_counter, lat).

    experiment : str
        Experiment name.
    
    title : str
        Plot title
    
    cmap : str or matplotlib colormap
        Colormap

    colorbar_label : str
        Colorbar label.
    
    levels : int or array-like
        Contour levels.

    save : str ot Path, optional.
        Output filename.
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

    Parameters
    ----------
    values : dict
        Dictionary with experiment names as keys and scalar xarray DataArrays
        as ECS values.

        Example:
        {
            "pi": 14.5,
            "x3": 22.3,
            "x6": 31.1
        }

    co2_levels : dict
        Dictionary mapping between experiment names and CO2 multipliers.

        Example:
        {
            "pi": 1,
            "x3": 3,
            "x6": 6
        }
    
    ecs : float, optional
        Equilibrium climate sensitivity to display in the legend

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

    # Add polar amplification values as text-only legend entry
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

    Returns
    slope : float
        Regression slope

    intercept : float
        Regression intercept
    """

    plt.figure(figsize=figsize, dpi=dpi)

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

    # X ticks at regular log2(CO2) intervals
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

    Parameters
    ----------
    series : dict (str, xarray.DataArray)
        Dictionary between experiment names and time series. 

    use_model_years : bool, default=False
        If True, use calendar/model years from 'time_counter'.
        If False, plot years sequentially starting from 1

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

#=======================================#
# ----- TOA AND RADIATION BALANCE ----- #
#=======================================#

def plot_gregory(
    tas_series,
    toa_series,
    experiment_labels=None,
    colors=None,
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

    The first (circle) and last (square) simulation years are highlighted separatedly.

    Parameters
    tas_series : dict[str, xarray.DataArray]
            Annual global mean surface temperature time series.
    
    toa_series : dict[str, xarray.DataArray]
            Corresponding annual global mean TOA radiation time series.

    colors : dict, optional
        Mapping between experiment names and plot colors.

    save : str or Path, optional
        Output filename.
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

    for k in tas_series:

        tas = tas_series[k]
        toa = toa_series[k]

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

        # Linear Gregory regression
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
    toa,
    sfc,
    difference,
    experiment_labels=None,
    title=None,
    save=None
):
    """
    Plot global TOA and surface radiation balance (SFC) time series.

    The figure contains two panels:
        Top:
            Net TOA radiation and net surface radiation
        Bottom:
            Difference between TOA and surface net radiation

    Parameters
    toa : dict[str, xarray.DataArray]
        Annual global mean TOA radiation time series
    
    sfc : dict[str, xarray.DataArray]
        Annual global mean surface radiation time series

    difference : dict[str, xarray.DataArray]
        TOA minus surface radiation imbalance

    title : str
        Overall figure title

    save : str or Path, optional
        Output filename
    """

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
    )

    # Use internal experiment names if no display labels are provided
    if experiment_labels is None:
        experiment_labels = {
            exp: exp
            for exp in toa
        }

    # Top panel: Net TOA and net SFC energy fluxes
    for exp in toa:

        years = toa[exp]["time_counter"].dt.year
        label_t = experiment_labels.get(exp, exp)

        axes[0].plot(
            years,
            toa[exp],
            label=f"{label_t} TOA",
        )

        axes[0].plot(
            years,
            sfc[exp],
            linestyle="--",
            label=f"{label_t} SFC",
        )

    # Zerp represents radiative balance
    axes[0].axhline(
        0,
        linestyle=":",
        linewidth=1,
    )

    axes[0].set_ylabel("Net radiation (W m$^{-2}$)")
    axes[0].set_title("Net TOA and net SFC radiation")
    axes[0].legend()

    # Bottom panel: TOA minus surface radiation
    for exp in difference:

        years = difference[exp]["time_counter"].dt.year
        label_b = experiment_labels.get(exp, exp)

        axes[1].plot(
            years,
            difference[exp],
            label=label_b,
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

    annotate : bool, defaulte=True
        Whether experiment names should be displayed next to points.

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    title : str, optional
        Figure title.

    ylim : tuple, optional
        Y-axis limits.

    xlim : tuple, optional
        X-axis limits.

    save : str or Path, optional
        Output filename.
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

    This function is designed for comparisons with DeepMIP.style multi-model datasets.
    Each model receives a consistent color and marker.

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

    # Create one legend entry per model
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
    Plot meridional SST gradient against global mean SST. For comparison with DeepMIP results.

    This function is intended for comparison with DeepMIP-style multi-model results.

    Each model is represented using a consistent color and marker. Experiments belonging to the 
    same model are connected to highlight their response across climate states.

    Parameters
    data : list of dict
        Each dictionary must contain:
        "model", "x", "y", "exp", "pi"

    title : str
        Figure title

    save : str of Path
        Output filename.

    model-styles : dict
        Model-specific colors and markers
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

        # Use larger symbols for non-reference experiments
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

    # Model legend
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

# Consisten visual identity for DeepMIP comparison figures

# Using a single dictionary ensures that each model has the 
# same appearance across all plots in the analysis

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

