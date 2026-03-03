"""
Tripathy-Mishra Drought Severity Index (TMDSI)
===============================================
Reformulation of scPDSI with analytically derived parameters.

Reference:
    Tripathy, K.P. and Mishra, A.K. (in prep.)
    "Beyond Empirical Constants: A Physically Calibrated Drought
    Severity Index." Water Resources Research.

What changes vs. scPDSI (Steps 6–9 only; Steps 1–5 are unchanged):
--------------------------------------------------------------------
Step 6  | Palmer's empirical K            → Probability-based standardization
        |                                   Z_m = Φ⁻¹(F_{d,m}(d_m))
        |                                   using fitted monthly distributions

Step 7  | Fixed φ = 0.897 globally       → φ_m = 1 − (RO_m + L_m) / AWC
        |                                   derived from linear reservoir model

Step 8  | Fixed w = 1/3 globally         → w_m = √(1 − φ_m²)
        |                                   derived from AR(1) stationarity
        |                                   (enforces Var(X) = 1 everywhere)

Step 9  | AR(1) with fixed coefficients  → AR(1) with locally derived φ, w
        |   X_m = 0.897·X_{m-1} + Z/3     X_m = φ_m·X_{m-1} + w_m·Z_m
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

from tmdsi.pet.thornthwaite import thornthwaite_pet
from tmdsi.pet.hargreaves   import hargreaves_pet
from tmdsi.water_balance    import cafec_coefficients, run_water_balance

# --------------------------------------------------------------------------- #
# Supported distributions for Step 6
# --------------------------------------------------------------------------- #
_DISTRIBUTIONS = Literal["normal", "pearson3", "best"]
_K8_INIT = 40


# --------------------------------------------------------------------------- #
# Public result container
# --------------------------------------------------------------------------- #
@dataclass
class TMDSIResult:
    tmdsi : np.ndarray     # Tripathy-Mishra Drought Severity Index
    phdi  : np.ndarray     # Hydrological Drought Index (same logic as PDSI)
    pmdi  : np.ndarray     # Palmer Modified Drought Index
    z     : np.ndarray     # standardized Z-index (≈ N(0,1) by construction)
    cafec : dict           # CAFEC coefficients {alpha, beta, gamma, delta}
    phi   : np.ndarray     # derived monthly φ, shape (12,)
    w     : np.ndarray     # derived monthly w, shape (12,)
    dist  : list[str]      # distribution selected per month (length 12)
    d_raw : np.ndarray     # raw moisture departure d = P - P_hat, flattened


# --------------------------------------------------------------------------- #
# Step 6: Probability-based Z-index standardization
# --------------------------------------------------------------------------- #

def _fit_distribution(
    d_calib : np.ndarray,
    method  : str,
) -> tuple[str, object]:
    """
    Fit a distribution to calibration-period d values for one month.

    Parameters
    ----------
    d_calib : 1-D array of moisture departures for one calendar month
              over the calibration period
    method  : 'normal' | 'pearson3' | 'best'
              'best' selects based on KS-test p-value

    Returns
    -------
    (name, fitted_distribution_object)
    """
    d_calib = d_calib[~np.isnan(d_calib)]

    if method == "normal":
        mu, sig = np.mean(d_calib), np.std(d_calib, ddof=1)
        return "normal", stats.norm(loc=mu, scale=sig)

    if method == "pearson3":
        skew, loc, scale = stats.pearson3.fit(d_calib)
        return "pearson3", stats.pearson3(skew, loc=loc, scale=scale)

    # method == "best": test Normal vs Pearson III, pick higher KS p-value
    mu, sig = np.mean(d_calib), np.std(d_calib, ddof=1)
    dist_norm = stats.norm(loc=mu, scale=sig)
    ks_norm   = stats.kstest(d_calib, dist_norm.cdf).pvalue

    skew, loc, scale = stats.pearson3.fit(d_calib)
    dist_p3   = stats.pearson3(skew, loc=loc, scale=scale)
    ks_p3     = stats.kstest(d_calib, dist_p3.cdf).pvalue

    if ks_p3 > ks_norm:
        return "pearson3", dist_p3
    return "normal", dist_norm


def _compute_standardized_z(
    d         : np.ndarray,      # (n_years, 12)
    calib_s   : int,
    calib_e   : int,
    dist_method: str,
) -> tuple[np.ndarray, list[str]]:
    """
    Step 6 (TMDSI): Z_m = Φ⁻¹(F_{d,m}(d_m))

    For each calendar month m:
      1. Fit distribution F to {d_{y,m}} over the calibration period
      2. Transform each d value to a probability: p = F(d)
      3. Map probability to standard normal quantile: Z = Φ⁻¹(p)

    When d is Normal this reduces exactly to Z = (d − μ_d) / σ_d.

    Returns
    -------
    z       : np.ndarray, shape (n_years, 12), standard-normal Z values
    dist_names : list of str, length 12 — which distribution was fitted per month
    """
    n_years = d.shape[0]
    z          = np.zeros((n_years, 12))
    dist_names = []

    # small epsilon to keep probabilities strictly inside (0, 1)
    # to avoid Φ⁻¹(0) = -inf or Φ⁻¹(1) = +inf
    _EPS = 1e-6

    for mo in range(12):
        d_calib     = d[calib_s : calib_e + 1, mo]
        name, dist  = _fit_distribution(d_calib, dist_method)
        dist_names.append(name)

        for yr in range(n_years):
            p          = dist.cdf(d[yr, mo])
            p          = float(np.clip(p, _EPS, 1.0 - _EPS))
            z[yr, mo]  = stats.norm.ppf(p)

    return z, dist_names


# --------------------------------------------------------------------------- #
# Step 7: Derive φ from soil water balance (linear reservoir)
# --------------------------------------------------------------------------- #

def _derive_phi(wb: dict, awc: float, n_calib: int) -> np.ndarray:
    """
    Step 7 (TMDSI): φ_m = 1 − k_m,   k_m = (RO_m + L_m) / AWC

    From linear reservoir model of soil moisture anomaly:
        S'_m = (1 − k) · S'_{m-1} + d_m
    which gives the AR(1) persistence φ = 1 − k.

    k_m is the mean fractional monthly drainage/loss rate:
        k_m = mean(RO_m + Loss_m) / AWC

    Physical interpretation:
        - Arid region  : high k → low φ → fast drought memory decay
        - Humid region : low k  → high φ → long memory / persistent drought

    Returns
    -------
    phi : np.ndarray, shape (12,), monthly persistence parameter
          clipped to [0.1, 0.99] for numerical stability
    """
    mean_ro   = wb["rosum"] / n_calib   # mean monthly runoff
    mean_loss = wb["tlsum"] / n_calib   # mean monthly soil loss

    k   = (mean_ro + mean_loss) / max(awc, 1e-6)
    phi = 1.0 - k

    # clip: φ must be in (0, 1) for AR(1) to be stationary
    # lower bound 0.1: prevents instantaneous memory loss
    # upper bound 0.99: prevents unit root (non-stationary)
    return np.clip(phi, 0.10, 0.99)


# --------------------------------------------------------------------------- #
# Step 8: Derive w from AR(1) stationarity condition
# --------------------------------------------------------------------------- #

def _derive_w(phi: np.ndarray) -> np.ndarray:
    """
    Step 8 (TMDSI): w_m = √(1 − φ_m²)

    Derivation:
        For stationary AR(1):  X_m = φ·X_{m-1} + w·Z_m
        Variance at stationarity:
            Var(X) = w² · Var(Z) / (1 − φ²)

        Require Var(X) = 1 (standard, comparable across all climates):
            1 = w² · σ²_Z / (1 − φ²)

        Since Z ~ N(0,1) by construction in Step 6 → σ_Z = 1:
            w = √(1 − φ²)

    Note: w decreases as φ increases — high-memory systems are less
    sensitive to a single month's anomaly (physically correct).

    Compare: Palmer's w = 1/3 corresponds to φ = 0.897 with σ_Z ≈ 2.4,
    implying Z was NOT standardized, which is the inconsistency TMDSI resolves.

    Returns
    -------
    w : np.ndarray, shape (12,)
    """
    return np.sqrt(1.0 - phi ** 2)


# --------------------------------------------------------------------------- #
# AR(1) recursion + backtracking (Step 9) — same logic, now with local φ, w
# --------------------------------------------------------------------------- #

def _case(prob: float, x1: float, x2: float, x3: float) -> float:
    if x3 == 0:
        return x1 if abs(x1) > abs(x2) else x2
    if prob <= 0 or prob >= 100:
        return x3
    p = prob / 100.0
    return (1 - p) * x3 + (p * x1 if x3 <= 0 else p * x2)


class _TMDSIBacktrackState:
    """
    Same backtracking structure as scPDSI but φ and w are
    month-specific arrays rather than global scalars.
    """

    def __init__(self, n_years: int, phi: np.ndarray, w: np.ndarray):
        self.phi = phi   # shape (12,)
        self.w   = w     # shape (12,)
        self.n_years = n_years

        self.v   = 0.0
        self.pro = 0.0
        self.x1 = self.x2 = self.x3 = 0.0
        self.iass = 0
        self.ze = self.ud = self.uw = 0.0

        self.tmdsi = np.full((n_years, 12), np.nan)
        self.phdi  = np.full((n_years, 12), np.nan)
        self.pmdi  = np.full((n_years, 12), np.nan)
        self.ppr   = np.zeros((n_years, 12))
        self.px1   = np.zeros((n_years, 12))
        self.px2   = np.zeros((n_years, 12))
        self.px3   = np.zeros((n_years, 12))
        self.x     = np.zeros((n_years, 12))

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

    def _ar1(self, z_val: float, x_prev: float, mo: int) -> float:
        """AR(1) step with month-specific φ and w."""
        return self.phi[mo] * x_prev + self.w[mo] * z_val

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
            self.tmdsi[j, m] = self.sx[idx]
            self.phdi[j, m]  = self.px3[j, m] or self.sx[idx]
            self.pmdi[j, m]  = _case(
                self.ppr[j, m], self.px1[j, m], self.px2[j, m], self.px3[j, m])
        self.k8 = 0

    def _write_outputs(self, yr: int, mo: int):
        self.tmdsi[yr, mo] = self.x[yr, mo]
        self.phdi[yr, mo]  = self.px3[yr, mo] or self.x[yr, mo]
        self.pmdi[yr, mo]  = _case(
            self.ppr[yr, mo], self.px1[yr, mo], self.px2[yr, mo], self.px3[yr, mo])

    def _save_state(self, yr: int, mo: int):
        self.v   = self.pv
        self.pro = self.ppr[yr, mo]
        self.x1  = self.px1[yr, mo]
        self.x2  = self.px2[yr, mo]
        self.x3  = self.px3[yr, mo]

    def _stmt_210(self, yr: int, mo: int):
        self.pv = 0.0
        self.px1[yr, mo] = 0.0
        self.px2[yr, mo] = 0.0
        self.ppr[yr, mo] = 0.0
        self.px3[yr, mo] = self._ar1(self.z[yr, mo], self.x3, mo)
        self.x[yr, mo]   = self.px3[yr, mo]
        if self.k8 == 0:
            self._write_outputs(yr, mo)
        else:
            self.iass = 3
            self._assign(yr, mo)
        self._save_state(yr, mo)

    def _stmt_200(self, yr: int, mo: int):
        self.px1[yr, mo] = max(0.0, self._ar1(self.z[yr, mo], self.x1, mo))
        if self.px1[yr, mo] >= 1 and self.px3[yr, mo] == 0:
            self.x[yr, mo]   = self.px1[yr, mo]
            self.px3[yr, mo] = self.px1[yr, mo]
            self.px1[yr, mo] = 0.0
            self.iass = 1
            self._assign(yr, mo); self._save_state(yr, mo)
            return
        self.px2[yr, mo] = min(0.0, self._ar1(self.z[yr, mo], self.x2, mo))
        if self.px2[yr, mo] <= -1 and self.px3[yr, mo] == 0:
            self.x[yr, mo]   = self.px2[yr, mo]
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
        if self.k8 >= len(self.sx) - 1:
            self._grow_buffers()
        self.sx1[self.k8] = self.px1[yr, mo]
        self.sx2[self.k8] = self.px2[yr, mo]
        self.sx3[self.k8] = self.px3[yr, mo]
        self.x[yr, mo] = self.px3[yr, mo]
        self.k8 += 1; self.k8max = self.k8
        self._save_state(yr, mo)

    def _stmt_190(self, yr: int, mo: int):
        q = self.ze if self.pro == 100 else self.ze + self.v
        self.ppr[yr, mo] = min((self.pv / q) * 100, 100.0)
        if self.ppr[yr, mo] >= 100:
            self.ppr[yr, mo] = 100.0
            self.px3[yr, mo] = 0.0
        else:
            self.px3[yr, mo] = self._ar1(self.z[yr, mo], self.x3, mo)
        self._stmt_200(yr, mo)

    def _stmt_180(self, yr: int, mo: int):
        self.uw = self.z[yr, mo] + 0.15
        self.pv = self.uw + max(self.v, 0.0)
        if self.pv <= 0:
            self._stmt_210(yr, mo); return
        self.ze = -2.691 * self.x3 - 1.5
        self._stmt_190(yr, mo)

    def _stmt_170(self, yr: int, mo: int):
        self.ud = self.z[yr, mo] - 0.15
        self.pv = self.ud + min(self.v, 0.0)
        if self.pv >= 0:
            self._stmt_210(yr, mo); return
        self.ze = -2.691 * self.x3 + 1.5
        self._stmt_190(yr, mo)

    def run(self, z: np.ndarray):
        self.z = z
        for yr in range(z.shape[0]):
            for mo in range(12):
                k8 = self.k8
                if k8 >= len(self.indexj) - 1:
                    self._grow_buffers()
                self.indexj[k8] = yr
                self.indexm[k8] = mo
                self.ze = self.ud = self.uw = 0.0
                z_val        = z[yr, mo]
                x3           = self.x3
                no_abatement = self.pro in (0.0, 100.0)
                if no_abatement:
                    if -0.5 <= x3 <= 0.5:
                        self.pv = 0.0
                        self.ppr[yr, mo] = 0.0
                        self.px3[yr, mo] = 0.0
                        self._stmt_200(yr, mo)
                    elif x3 > 0.5:
                        self._stmt_210(yr, mo) if z_val >= 0.15 else self._stmt_170(yr, mo)
                    else:
                        self._stmt_210(yr, mo) if z_val <= -0.15 else self._stmt_180(yr, mo)
                else:
                    self._stmt_170(yr, mo) if x3 > 0 else self._stmt_180(yr, mo)

    def finish(self):
        n_yr = self.n_years
        for k in range(self.k8max):
            i, j = int(self.indexj[k]), int(self.indexm[k])
            self.tmdsi[i, j] = self.x[i, j]
            self.phdi[i, j]  = self.px3[i, j] or self.x[i, j]
            self.pmdi[i, j]  = _case(
                self.ppr[n_yr - 1, 11],
                self.px1[n_yr - 1, 11],
                self.px2[n_yr - 1, 11],
                self.px3[n_yr - 1, 11],
            )


# --------------------------------------------------------------------------- #
# Internal: compute raw d = P - P_hat over full record
# --------------------------------------------------------------------------- #

def _compute_d(
    precip : np.ndarray,
    pet    : np.ndarray,
    wb     : dict,
    cafec  : dict,
) -> np.ndarray:
    """Raw moisture departure d = P − P̂, shape (n_years, 12)."""
    n_years = precip.shape[0]
    d = np.zeros((n_years, 12))
    for yr in range(n_years):
        for mo in range(12):
            phat = (
                cafec["alpha"][mo] * pet[yr, mo]
                + cafec["beta"][mo]  * wb["pr"][yr, mo]
                + cafec["gamma"][mo] * wb["sp"][yr, mo]
                - cafec["delta"][mo] * wb["pl"][yr, mo]
            )
            d[yr, mo] = precip[yr, mo] - phat
    return d


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
PETMethod = Literal["thornthwaite", "hargreaves", "penman_monteith"]
DistMethod = Literal["normal", "pearson3", "best"]


def compute(
    precip                 : np.ndarray,
    tmax                   : np.ndarray,
    tmin                   : np.ndarray,
    awc                    : float,
    latitude               : float,
    data_start_year        : int,
    calibration_start_year : int,
    calibration_end_year   : int,
    pet_method             : PETMethod  = "thornthwaite",
    dist_method            : DistMethod = "best",
    fitting_params         : dict | None = None,
    **pet_kwargs,
) -> TMDSIResult:
    """
    Compute TMDSI (Tripathy-Mishra Drought Severity Index).

    Steps 1–5 are identical to scPDSI (water balance, CAFEC).
    Steps 6–9 use analytically derived parameters — no Palmer constants.

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
    dist_method            : 'normal' | 'pearson3' | 'best'
                             Distribution to fit to d_m in Step 6.
                             'best' selects between Normal and Pearson III
                             using the KS test p-value.
    fitting_params         : optional precomputed CAFEC coefficients
    **pet_kwargs           : forwarded to PET function if needed

    Returns
    -------
    TMDSIResult
        .tmdsi : drought index time series
        .z     : standardized Z (≈ N(0,1) by construction)
        .phi   : derived monthly φ, shape (12,)  ← key output for paper
        .w     : derived monthly w, shape (12,)  ← key output for paper
        .dist  : distribution chosen per month (length-12 list)
        .d_raw : raw moisture departure P − P̂
    """
    if np.all(np.isnan(precip)):
        nan = np.full(precip.shape, np.nan)
        return TMDSIResult(nan, nan, nan, nan, {}, np.zeros(12), np.zeros(12), [], nan)

    orig_len = precip.size
    precip   = np.clip(precip, 0.0, None)

    # --- PET (Step 1) -------------------------------------------------------
    if pet_method == "thornthwaite":
        pet = thornthwaite_pet(tmax, tmin, latitude, data_start_year)
    elif pet_method == "hargreaves":
        pet = hargreaves_pet(tmax, tmin, latitude, data_start_year, **pet_kwargs)
    elif pet_method == "penman_monteith":
        raise NotImplementedError(
            "Penman-Monteith PET not yet implemented. "
            "Use 'thornthwaite' or 'hargreaves'."
        )
    else:
        raise ValueError(f"Unknown pet_method '{pet_method}'.")

    # --- reshape (n_years, 12) ----------------------------------------------
    n_years   = orig_len // 12
    precip_2d = precip[: n_years * 12].reshape(n_years, 12)
    pet_2d    = pet   [: n_years * 12].reshape(n_years, 12)
    cs        = calibration_start_year - data_start_year
    ce        = calibration_end_year   - data_start_year
    n_calib   = ce - cs + 1

    # --- Water balance + CAFEC (Steps 2–5) ----------------------------------
    wb = run_water_balance(precip_2d, pet_2d, awc, cs, ce)
    if fitting_params and all(k in fitting_params for k in ("alpha", "beta", "gamma", "delta")):
        cafec = {k: np.asarray(fitting_params[k]) for k in ("alpha", "beta", "gamma", "delta")}
    else:
        cafec = cafec_coefficients(wb, n_calib)

    # --- Raw d = P − P̂ ------------------------------------------------------
    d = _compute_d(precip_2d, pet_2d, wb, cafec)

    # --- Step 6: Probability-based Z-index ----------------------------------
    z, dist_names = _compute_standardized_z(d, cs, ce, dist_method)

    # --- Step 7: Derive φ from linear reservoir model ----------------------
    phi = _derive_phi(wb, awc, n_calib)

    # --- Step 8: Derive w from AR(1) stationarity condition ----------------
    w = _derive_w(phi)

    # --- Step 9: AR(1) recursion with local φ, w ----------------------------
    state = _TMDSIBacktrackState(n_years, phi=phi, w=w)
    state.run(z)
    state.finish()

    def flat(arr): return arr.flatten()[:orig_len]

    return TMDSIResult(
        tmdsi  = flat(state.tmdsi),
        phdi   = flat(state.phdi),
        pmdi   = flat(state.pmdi),
        z      = flat(z),
        cafec  = cafec,
        phi    = phi,
        w      = w,
        dist   = dist_names,
        d_raw  = flat(d),
    )