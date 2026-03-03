import numpy as np
import os
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

# This finds the absolute path to the 'tmdsi' root directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "pawc.dat"

def load_awc_data():
    """Loads AWC data using a relative path to the package root."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"AWC data not found at {DATA_PATH}")
        
    data = np.loadtxt(DATA_PATH, skiprows=6)
    data[data == -2] = np.nan
    
    # Grid definition based on Webb et al.
    lats = np.linspace(90, -90, data.shape[0])
    lons = np.linspace(-180, 180, data.shape[1])
    
    return data, lats, lons

def estimate_awc(lon, lat):
    
    """
    Interpolates AWC value. If result is NaN (coastlines), 
    searches nearby grid points.
    """
    data, lats, lons = load_awc_data()
    
    # Setup Interpolator (ensuring increasing coordinates for scipy)
    interp = RegularGridInterpolator(
        (np.flip(lats), lons), 
        np.flipud(data), 
        bounds_error=False, 
        fill_value=np.nan
    )
    
    val = interp([lat, lon])[0]
    
    # Fallback logic for NaN points (mimicking cszang.awc)
    if np.isnan(val):
        search_radius = 5
        lat_idx = np.where((lats <= lat + search_radius) & (lats >= lat - search_radius))[0]
        lon_idx = np.where((lons <= lon + search_radius) & (lons >= lon - search_radius))[0]
        
        subset = data[np.ix_(lat_idx, lon_idx)]
        sub_lons, sub_lats = np.meshgrid(lons[lon_idx], lats[lat_idx])
        
        mask = ~np.isnan(subset)
        if np.any(mask):
            dists = np.sqrt((sub_lons[mask] - lon)**2 + (sub_lats[mask] - lat)**2)
            val = subset[mask][np.argmin(dists)]
            
    return val