import xarray as xr
import numpy as np
from pathlib import Path

# open datasets
def load_last_years(base_path, pattern, nyears=None, which="last", time_dim="time_counter"):

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