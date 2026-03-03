"""
scPDSI — Self-Calibrating Palmer Drought Severity Index
=========================================================
Implements Wells et al. (2004) scPDSI.

Steps performed here (Steps 6–9 of the PDSI pipeline):
  6. K-factor calibration  (Palmer empirical K, scaled by 17.67)
  7. Z-index computation
  8. AR(1) recursion with phi=0.897, w=1/3
  9. Backtracking to resolve spell transitions

Steps 1–5 (PET, water balance, CAFEC) live in:
  tmdsi.pet.*
  tmdsi.water_balance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from tmdsi.pet.thornthwaite import (
    hargreaves_pet,
    penman_monteith_pet,
    thornthwaite_pet,
)
from tmdsi.water_balance import cafec_coefficients, run_water_balance

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
_PHI = 0.897          # Palmer AR(1) persistence (empirical)
_W   = 1.0 / 3.0     # Palmer AR(1) scaling (empirical)
_K8_INIT = 40        # initial backtrack buffer size


# --------------------------------------------------------------------------- #
# Public result container
# --------------------------------------------------------------------------- #
@dataclass
class PDSIResult:
    pdsi  : np.ndarray   # Palmer Drought Severity Index
    phdi  : np.ndarray   # Palmer Hydrological Drought Index
    pmdi  : np.ndarray   # Palmer Modified Drought Index (weighted)
    z     : np.ndarray   # Z-index (moisture anomaly)
    cafec : dict         # fitted CAFEC coefficients {alpha,beta,gamma,delta}
    phi   : float = _PHI # persistence parameter used
    w     : float = _W   # scaling parameter used


# --------------------------------------------------------------------------- #
# K-factor  (Step 6 in original scPDSI)
# --------------------------------------------------------------------------- #
def _compute_kfactors(
    precip   : np.ndarray,   # (n_years, 12)
    pet      : np.ndarray,
    wb       : dict,
    cafec    : dict,
    calib_s  : int,
    calib_e  : int,
    n_calib  : int,
) -> np.ndarray:
    """
    Palmer's empirical K with Wells et al. (2004) self-calibrating rescaling.
    Returns ak: shape (12,)
    """
    # mean demand/supply ratio per month
    trat = (wb["petsum"] + wb["rsum"] + wb["rosum"]) / (
           np.maximum(wb["psum"] + wb["tlsum"], 1e-9))

    # accumulate |d| over calibration period
    sabsd = np.zeros(12)
    for yr in range(calib_s, calib_e + 1):
        for mo in range(12):
            phat = (
                cafec["alpha"][mo] * pet[yr, mo]
                + cafec["beta"][mo]  * wb["pr"][yr, mo]
                + cafec["gamma"][mo] * wb["sp"][yr, mo]
                - cafec["delta"][mo] * wb["pl"][yr, mo]
            )
            sabsd[mo] += abs(precip[yr, mo] - phat)

    dbar  = sabsd / n_calib
    akhat = 1.5 * np.log10((trat + 2.8) / np.maximum(dbar, 1e-9)) + 0.5
    swtd  = np.sum(dbar * akhat)
    ak    = 17.67 * akhat / swtd if swtd != 0 else akhat
    return ak


# --------------------------------------------------------------------------- #
# Z-index  (Step 7)
# --------------------------------------------------------------------------- #
def _compute_zindex(
    precip : np.ndarray,   # (n_years, 12)
    pet    : np.ndarray,
    wb     : dict,
    cafec  : dict,
    ak     : np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns z (n_years,12) and cp (CAFEC precipitation, n_years,12).
    """
    n_years = precip.shape[0]
    z  = np.zeros((n_years, 12))
    cp = np.zeros((n_years, 12))

    for yr in range(n_years):
        for mo in range(12):
            phat = (
                cafec["alpha"][mo] * pet[yr, mo]
                + cafec["beta"][mo]  * wb["pr"][yr, mo]
                + cafec["gamma"][mo] * wb["sp"][yr, mo]
                - cafec["delta"][mo] * wb["pl"][yr, mo]
            )
            cp[yr, mo] = phat
            z[yr, mo]  = ak[mo] * (precip[yr, mo] - phat)

    return z, cp


# --------------------------------------------------------------------------- #
# AR(1) recursion + backtracking  (Steps 8–9)
# --------------------------------------------------------------------------- #
def _case(prob: float, x1: float, x2: float, x3: float) -> float:
    if x3 == 0:
        return x1 if abs(x1) > abs(x2) else x2
    if prob <= 0 or prob >= 100:
        return x3
    p = prob / 100.0
    return (1 - p) * x3 + (p * x1 if x3 <= 0 else p * x2)


class _BacktrackState:
    """Mutable state for the backtracking PDSI recursion."""

    def __init__(self, n_years: int, phi: float, w: float):
        self.phi = phi
        self.w   = w
        self.n_years = n_years

        # spell state
        self.v   = 0.0
        self.pro = 0.0
        self.x1 = self.x2 = self.x3 = 0.0
        self.iass = 0
        self.ze = self.ud = self.uw = 0.0

        # outputs
        self.pdsi = np.full((n_years, 12), np.nan)
        self.phdi = np.full((n_years, 12), np.nan)
        self.pmdi = np.full((n_years, 12), np.nan)
        self.ppr  = np.zeros((n_years, 12))
        self.px1  = np.zeros((n_years, 12))
        self.px2  = np.zeros((n_years, 12))
        self.px3  = np.zeros((n_years, 12))
        self.x    = np.zeros((n_years, 12))

        # backtrack buffer
        sz = _K8_INIT
        self.k8 = 0; self.k8max = 0
        self.sx  = np.zeros(sz); self.sx1 = np.zeros(sz)
        self.sx2 = np.zeros(sz); self.sx3 = np.zeros(sz)
        self.indexj = np.full(sz, np.nan)
        self.indexm = np.full(sz, np.nan)

    def _grow_buffers(self):
        extra = 10
        for attr in ("sx", "sx1", "sx2", "sx3"):
            setattr(self, attr, np.append(getattr(self, attr), np.zeros(extra)))
        self.indexj = np.append(self.indexj, np.full(extra, np.nan))
        self.indexm = np.append(self.indexm, np.full(extra, np.nan))

    def _ar1(self, z_val: float, x_prev: float) -> float:
        return self.phi * x_prev + self.w * z_val

    def _assign(self, yr: int, mo: int):
        self.sx[self.k8] = self.x[yr, mo]
        isave = self.iass

        if self.k8 == 0:
            self._write_outputs(yr, mo)
            return

        if self.iass == 3:
            for i in range(self.k8):
                self.sx[i] = self.sx3[i]
        else:
            for i in range(self.k8 - 1, -1, -1):
                if isave == 2:
                    if self.sx2[i] == 0:
                        isave = 1; self.sx[i] = self.sx1[i]
                    else:
                        self.sx[i] = self.sx2[i]
                else:
                    if self.sx1[i] == 0:
                        isave = 2; self.sx[i] = self.sx2[i]
                    else:
                        self.sx[i] = self.sx1[i]

        for idx in range(self.k8 + 1):
            j, m = int(self.indexj[idx]), int(self.indexm[idx])
            self.pdsi[j, m] = self.sx[idx]
            self.phdi[j, m] = self.px3[j, m] or self.sx[idx]
            self.pmdi[j, m] = _case(
                self.ppr[j, m], self.px1[j, m], self.px2[j, m], self.px3[j, m])
        self.k8 = 0

    def _write_outputs(self, yr: int, mo: int):
        self.pdsi[yr, mo] = self.x[yr, mo]
        self.phdi[yr, mo] = self.px3[yr, mo] or self.x[yr, mo]
        self.pmdi[yr, mo] = _case(
            self.ppr[yr, mo], self.px1[yr, mo], self.px2[yr, mo], self.px3[yr, mo])

    def _save_state(self, yr: int, mo: int):
        self.v   = self.pv
        self.pro = self.ppr[yr, mo]
        self.x1  = self.px1[yr, mo]
        self.x2  = self.px2[yr, mo]
        self.x3  = self.px3[yr, mo]

    # --- spell-transition logic -------------------------------------------- #

    def _stmt_210(self, yr: int, mo: int):
        """Abatement fizzled: accept all x3."""
        self.pv = 0.0
        self.px1[yr, mo] = 0.0
        self.px2[yr, mo] = 0.0
        self.ppr[yr, mo] = 0.0
        self.px3[yr, mo] = self._ar1(self.z[yr, mo], self.x3)
        self.x[yr, mo]   = self.px3[yr, mo]
        if self.k8 == 0:
            self._write_outputs(yr, mo)
        else:
            self.iass = 3
            self._assign(yr, mo)
        self._save_state(yr, mo)

    def _stmt_200(self, yr: int, mo: int):
        """Continue x1/x2; check if new spell established."""
        self.px1[yr, mo] = max(0.0, self._ar1(self.z[yr, mo], self.x1))
        if self.px1[yr, mo] >= 1 and self.px3[yr, mo] == 0:
            self.x[yr, mo] = self.px1[yr, mo]
            self.px3[yr, mo] = self.px1[yr, mo]
            self.px1[yr, mo] = 0.0
            self.iass = 1
            self._assign(yr, mo); self._save_state(yr, mo)
            return

        self.px2[yr, mo] = min(0.0, self._ar1(self.z[yr, mo], self.x2))
        if self.px2[yr, mo] <= -1 and self.px3[yr, mo] == 0:
            self.x[yr, mo] = self.px2[yr, mo]
            self.px3[yr, mo] = self.px2[yr, mo]
            self.px2[yr, mo] = 0.0
            self.iass = 2
            self._assign(yr, mo); self._save_state(yr, mo)
            return

        if self.px3[yr, mo] == 0:
            if self.px1[yr, mo] == 0:
                self.x[yr, mo] = self.px2[yr, mo]
                self.iass = 2; self._assign(yr, mo); self._save_state(yr, mo)
                return
            if self.px2[yr, mo] == 0:
                self.x[yr, mo] = self.px1[yr, mo]
                self.iass = 1; self._assign(yr, mo); self._save_state(yr, mo)
                return

        # no determination yet — buffer and backtrack later
        if self.k8 >= len(self.sx) - 1:
            self._grow_buffers()
        self.sx1[self.k8] = self.px1[yr, mo]
        self.sx2[self.k8] = self.px2[yr, mo]
        self.sx3[self.k8] = self.px3[yr, mo]
        self.x[yr, mo] = self.px3[yr, mo]
        self.k8 += 1; self.k8max = self.k8
        self._save_state(yr, mo)

    def _stmt_190(self, yr: int, mo: int):
        """Compute prob(end)."""
        q = self.ze if self.pro == 100 else self.ze + self.v
        self.ppr[yr, mo] = min((self.pv / q) * 100, 100.0)
        if self.ppr[yr, mo] >= 100:
            self.ppr[yr, mo] = 100.0
            self.px3[yr, mo] = 0.0
        else:
            self.px3[yr, mo] = self._ar1(self.z[yr, mo], self.x3)
        self._stmt_200(yr, mo)

    def _stmt_180(self, yr: int, mo: int):
        """Drought abatement check."""
        self.uw = self.z[yr, mo] + 0.15
        self.pv = self.uw + max(self.v, 0.0)
        if self.pv <= 0:
            self._stmt_210(yr, mo); return
        self.ze = -2.691 * self.x3 - 1.5
        self._stmt_190(yr, mo)

    def _stmt_170(self, yr: int, mo: int):
        """Wet-spell abatement check."""
        self.ud = self.z[yr, mo] - 0.15
        self.pv = self.ud + min(self.v, 0.0)
        if self.pv >= 0:
            self._stmt_210(yr, mo); return
        self.ze = -2.691 * self.x3 + 1.5
        self._stmt_190(yr, mo)

    # --- main loop --------------------------------------------------------- #

    def run(self, z: np.ndarray):
        self.z = z
        n_years = z.shape[0]

        for yr in range(n_years):
            for mo in range(12):
                k8 = self.k8
                if k8 >= len(self.indexj) - 1:
                    self._grow_buffers()
                self.indexj[k8] = yr
                self.indexm[k8] = mo
                self.ze = self.ud = self.uw = 0.0

                no_abatement = self.pro in (0.0, 100.0)
                z_val = z[yr, mo]
                x3    = self.x3

                if no_abatement:
                    if -0.5 <= x3 <= 0.5:
                        self.pv = 0.0
                        self.ppr[yr, mo] = 0.0
                        self.px3[yr, mo] = 0.0
                        self._stmt_200(yr, mo)
                    elif x3 > 0.5:
                        if z_val >= 0.15:
                            self._stmt_210(yr, mo)
                        else:
                            self._stmt_170(yr, mo)
                    else:
                        if z_val <= -0.15:
                            self._stmt_210(yr, mo)
                        else:
                            self._stmt_180(yr, mo)
                else:
                    if x3 > 0:
                        self._stmt_170(yr, mo)
                    else:
                        self._stmt_180(yr, mo)

    def finish(self):
        n_yr = self.n_years
        for k in range(self.k8max):
            i, j = int(self.indexj[k]), int(self.indexm[k])
            self.pdsi[i, j] = self.x[i, j]
            self.phdi[i, j] = self.px3[i, j] or self.x[i, j]
            self.pmdi[i, j] = _case(
                self.ppr[n_yr - 1, 11],
                self.px1[n_yr - 1, 11],
                self.px2[n_yr - 1, 11],
                self.px3[n_yr - 1, 11],
            )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
PETMethod = Literal["thornthwaite", "hargreaves", "penman_monteith"]


def compute(
    precip               : np.ndarray,
    tmax                 : np.ndarray,
    tmin                 : np.ndarray,
    awc                  : float,
    latitude             : float,
    data_start_year      : int,
    calibration_start_year: int,
    calibration_end_year : int,
    pet_method           : PETMethod = "thornthwaite",
    fitting_params       : dict | None = None,
    **pet_kwargs,
) -> PDSIResult:
    """
    Compute scPDSI (Wells et al. 2004).

    Parameters
    ----------
    precip                 : 1-D array, monthly precipitation (mm)
    tmax                   : 1-D array, monthly max temperature (°C)
    tmin                   : 1-D array, monthly min temperature (°C)
    awc                    : total available water capacity (mm)
    latitude               : decimal degrees
    data_start_year        : first year of the input record
    calibration_start_year : first year of the calibration period
    calibration_end_year   : last  year of the calibration period
    pet_method             : 'thornthwaite' | 'hargreaves' | 'penman_monteith'
    fitting_params         : optional dict with precomputed alpha/beta/gamma/delta
    **pet_kwargs           : extra keyword args forwarded to PET function
                             (e.g. wind_speed, rh, rs, elevation for PM)

    Returns
    -------
    PDSIResult
    """
    # --- guard: all-missing input ------------------------------------------
    if np.all(np.isnan(precip)):
        nan = np.full(precip.shape, np.nan)
        return PDSIResult(nan, nan, nan, nan, {})

    orig_len = precip.size
    precip = np.clip(precip, 0.0, None)

    # --- PET ---------------------------------------------------------------
    if pet_method == "thornthwaite":
        pet = thornthwaite_pet(tmax, tmin, latitude)
    elif pet_method == "hargreaves":
        pet = hargreaves_pet(tmax, tmin, latitude, **pet_kwargs)
    elif pet_method == "penman_monteith":
        pet = penman_monteith_pet(tmax, tmin, latitude=latitude, **pet_kwargs)
    else:
        raise ValueError(f"Unknown pet_method '{pet_method}'. "
                         "Choose: 'thornthwaite', 'hargreaves', 'penman_monteith'.")

    # --- reshape to (n_years, 12) ------------------------------------------
    n_years = len(precip) // 12
    precip_2d = precip[: n_years * 12].reshape(n_years, 12)
    pet_2d    = pet   [: n_years * 12].reshape(n_years, 12)

    cs = calibration_start_year - data_start_year
    ce = calibration_end_year   - data_start_year
    n_calib = ce - cs + 1

    # --- water balance (Steps 1–5) -----------------------------------------
    wb = run_water_balance(precip_2d, pet_2d, awc, cs, ce)

    # --- CAFEC coefficients -------------------------------------------------
    if fitting_params and all(k in fitting_params for k in ("alpha", "beta", "gamma", "delta")):
        cafec = {k: np.asarray(fitting_params[k]) for k in ("alpha", "beta", "gamma", "delta")}
    else:
        cafec = cafec_coefficients(wb, n_calib)

    # --- K-factors and Z-index (Steps 6–7) ---------------------------------
    ak = _compute_kfactors(precip_2d, pet_2d, wb, cafec, cs, ce, n_calib)
    z, _ = _compute_zindex(precip_2d, pet_2d, wb, cafec, ak)

    # --- AR(1) + backtracking (Steps 8–9) ----------------------------------
    state = _BacktrackState(n_years, phi=_PHI, w=_W)
    state.run(z)
    state.finish()

    # --- flatten and trim to original length --------------------------------
    def flat(arr): return arr.flatten()[:orig_len]

    return PDSIResult(
        pdsi  = flat(state.pdsi),
        phdi  = flat(state.phdi),
        pmdi  = flat(state.pmdi),
        z     = flat(z),
        cafec = cafec,
        phi   = _PHI,
        w     = _W,
    )