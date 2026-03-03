"""
Author: Kumar Puran Tripathy
Hargreaves-Samani (1985) Potential Evapotranspiration
------------------------------------------------------
Daily PET → aggregated to monthly totals.

References:
    Hargreaves, G.H. and Samani, Z.A. (1985) Reference crop
    evapotranspiration from temperature. Applied Engineering in
    Agriculture, 1(2), 96-99.

    Allen, R.G. et al. (1998) Crop evapotranspiration - FAO Irrigation
    and drainage paper 56. FAO, Rome. [eq. 52]
"""

from __future__ import annotations
import calendar
import math
import numpy as np

_SOLAR_CONSTANT     = 0.0820  # MJ m-2 min-1
_MONTH_DAYS_NONLEAP = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
_MONTH_DAYS_LEAP    = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


# --------------------------------------------------------------------------- #
# Solar geometry (same helpers as Thornthwaite — kept self-contained)
# --------------------------------------------------------------------------- #

def _solar_declination(doy: int) -> float:
    """Solar declination (radians). FAO eq. 24."""
    return 0.409 * math.sin((2.0 * math.pi / 365.0) * doy - 1.39)


def _sunset_hour_angle(lat_rad: float, decl_rad: float) -> float:
    """Sunset hour angle (radians). FAO eq. 25."""
    cos_ws = -math.tan(lat_rad) * math.tan(decl_rad)
    return math.acos(min(max(cos_ws, -1.0), 1.0))


def _extraterrestrial_radiation(lat_rad: float, doy: int) -> float:
    """
    Daily extraterrestrial radiation Ra (MJ m-2 day-1). FAO eq. 21.

    Ra = (24*60/π) * Gsc * dr * (Ws*sin(φ)*sin(δ) + cos(φ)*cos(δ)*sin(Ws))
    """
    decl = _solar_declination(doy)
    ws   = _sunset_hour_angle(lat_rad, decl)
    dr   = 1.0 + 0.033 * math.cos((2.0 * math.pi / 365.0) * doy)   # inv. rel. distance

    ra = (
        (24.0 * 60.0 / math.pi)
        * _SOLAR_CONSTANT
        * dr
        * (
            ws * math.sin(lat_rad) * math.sin(decl)
            + math.cos(lat_rad) * math.cos(decl) * math.sin(ws)
        )
    )
    return ra


def _monthly_ra(lat_rad: float, leap: bool = False) -> np.ndarray:
    """
    Mean daily Ra (MJ m-2 day-1) for each of the 12 calendar months.
    Returns shape (12,).
    """
    month_days = _MONTH_DAYS_LEAP if leap else _MONTH_DAYS_NONLEAP
    ra = np.zeros(12)
    doy = 1
    for m, ndays in enumerate(month_days):
        total = sum(_extraterrestrial_radiation(lat_rad, doy + d) for d in range(ndays))
        ra[m] = total / ndays
        doy  += ndays
    return ra


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def hargreaves_pet(
    tmax            : np.ndarray,
    tmin            : np.ndarray,
    latitude        : float,
    data_start_year : int = 1,
) -> np.ndarray:
    """
    Monthly PET via Hargreaves-Samani (1985).

    Equation (FAO-56, eq. 52):
        ETo = 0.0023 * (Tmean + 17.8) * (Tmax - Tmin)^0.5 * 0.408 * Ra
                                                                  [mm/day]
        PET_monthly = ETo_daily * N                               [mm/month]

    where:
        Ra    = mean daily extraterrestrial radiation (MJ m-2 day-1)
        N     = number of days in the month
        0.408 = conversion factor MJ m-2 day-1 → mm day-1

    Parameters
    ----------
    tmax            : 1-D array, monthly Tmax (°C)
    tmin            : 1-D array, monthly Tmin (°C)
    latitude        : decimal degrees (negative = Southern Hemisphere)
    data_start_year : calendar year of the first record (for leap-year Ra)

    Returns
    -------
    np.ndarray, shape (n_months,), PET in mm/month
    """
    tmean    = 0.5 * (tmax + tmin)
    td       = np.maximum(tmax - tmin, 0.0)   # temperature range, clipped to ≥ 0

    orig_len = tmean.size
    n_years  = orig_len // 12
    tmean_2d = tmean[: n_years * 12].reshape(n_years, 12)
    td_2d    = td   [: n_years * 12].reshape(n_years, 12)

    lat_rad = math.radians(float(latitude))
    ra_norm = _monthly_ra(lat_rad, leap=False)
    ra_leap = _monthly_ra(lat_rad, leap=True)

    pet = np.full((n_years, 12), np.nan)
    for yr in range(n_years):
        leap       = calendar.isleap(data_start_year + yr)
        month_days = _MONTH_DAYS_LEAP if leap else _MONTH_DAYS_NONLEAP
        ra         = ra_leap          if leap else ra_norm

        # daily ETo then multiply by days-in-month → mm/month
        eto_daily  = 0.0023 * (tmean_2d[yr] + 17.8) * td_2d[yr] ** 0.5 * 0.408 * ra
        pet[yr]    = eto_daily * month_days

    return pet.flatten()[:orig_len]