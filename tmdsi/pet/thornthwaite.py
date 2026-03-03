"""
Author: Kumar Puran Tripathy
Thornthwaite (1948) Potential Evapotranspiration
-------------------------------------------------
Reference:
    Thornthwaite, C.W. (1948) An approach toward a rational classification
    of climate. Geographical Review, 38, 55-94.
"""

from __future__ import annotations
import calendar
import math
import numpy as np

_MONTH_DAYS_NONLEAP = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
_MONTH_DAYS_LEAP    = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


# --------------------------------------------------------------------------- #
# Internal solar geometry helpers
# --------------------------------------------------------------------------- #

def _solar_declination(doy: int) -> float:
    """Solar declination angle (radians) for day-of-year. FAO eq. 24."""
    return 0.409 * math.sin((2.0 * math.pi / 365.0) * doy - 1.39)


def _sunset_hour_angle(lat_rad: float, decl_rad: float) -> float:
    """Sunset hour angle (radians). FAO eq. 25."""
    cos_ws = -math.tan(lat_rad) * math.tan(decl_rad)
    return math.acos(min(max(cos_ws, -1.0), 1.0))


def _mean_daylight_hours(lat_rad: float, leap: bool = False) -> np.ndarray:
    """Mean daily daylight hours for each of the 12 calendar months. Shape (12,)."""
    month_days = _MONTH_DAYS_LEAP if leap else _MONTH_DAYS_NONLEAP
    dlh = np.zeros(12)
    doy = 1
    for m, ndays in enumerate(month_days):
        total = 0.0
        for _ in range(ndays):
            ws     = _sunset_hour_angle(lat_rad, _solar_declination(doy))
            total += (24.0 / math.pi) * ws
            doy   += 1
        dlh[m] = total / ndays
    return dlh


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def thornthwaite_pet(
    tmax            : np.ndarray,
    tmin            : np.ndarray,
    latitude        : float,
    data_start_year : int = 1,
) -> np.ndarray:
    """
    Monthly PET via Thornthwaite (1948).

    Equation:
        PET = 16 * (L/12) * (N/30) * (10*Ta/I)^a   [mm/month]

    where:
        Ta  = mean monthly temperature (°C), clipped to ≥ 0
        N   = number of days in the month
        L   = mean daily daylight hours for that month
        I   = annual heat index = Σ_m (T_m / 5)^1.514
        a   = 6.75e-7*I³ - 7.71e-5*I² + 1.792e-2*I + 0.49239

    Parameters
    ----------
    tmax            : 1-D array, monthly Tmax (°C)
    tmin            : 1-D array, monthly Tmin (°C)
    latitude        : decimal degrees (negative = Southern Hemisphere)
    data_start_year : calendar year of the first record (used for leap-year
                      correction). Defaults to 1 (treated as non-leap).

    Returns
    -------
    np.ndarray, shape (n_months,), PET in mm/month
    """
    tmean    = np.clip(0.5 * (tmax + tmin), 0.0, None)
    orig_len = tmean.size
    n_years  = orig_len // 12
    tmean_2d = tmean[: n_years * 12].reshape(n_years, 12)

    # heat index I from long-term monthly means
    mean_monthly = np.nanmean(tmean_2d, axis=0)
    I = float(np.sum(np.maximum(mean_monthly / 5.0, 0.0) ** 1.514))

    # exponent a
    a = 6.75e-7 * I**3 - 7.71e-5 * I**2 + 1.792e-2 * I + 0.49239

    # daylight hours for leap and non-leap years
    lat_rad  = math.radians(float(latitude))
    dlh_norm = _mean_daylight_hours(lat_rad, leap=False)
    dlh_leap = _mean_daylight_hours(lat_rad, leap=True)

    pet = np.full((n_years, 12), np.nan)
    for yr in range(n_years):
        leap       = calendar.isleap(data_start_year + yr)
        month_days = _MONTH_DAYS_LEAP if leap else _MONTH_DAYS_NONLEAP
        dlh        = dlh_leap         if leap else dlh_norm
        if I > 0:
            pet[yr] = (
                16.0
                * (dlh / 12.0)
                * (month_days / 30.0)
                * (10.0 * tmean_2d[yr] / I) ** a
            )
        else:
            pet[yr] = 0.0

    return pet.flatten()[:orig_len]