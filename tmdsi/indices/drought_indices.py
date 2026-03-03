"""
drought_indices.py
==================
Three drought indices in one file. All share Steps 1-5.
They differ ONLY in how they compute Z and the AR(1) coefficients.

  compute_pdsi   : Palmer (1965)  — fixed K, fixed p=0.897, q=1/3
  compute_scpdsi : Wells (2004)   — local K (17.67 rescaling), local p,q from m,b fit
  compute_tmdsi  : Tripathy & Mishra (in prep.) — prob. Z, physics-derived phi,w

SHARED PIPELINE (Steps 1–5):
  1. PET via Thornthwaite
  2. Palmer two-layer water balance → ET, R, RO, L
  3. CAFEC coefficients α, β, γ, δ
  4. CAFEC precipitation P_hat
  5. Moisture departure d = P - P_hat

STEP 6 differences:
  PDSI   : K = 1.5*log10((PE+R+RO)/(P+L) + 2.8) / D_bar + 0.5   [global calibration constants]
  scPDSI : same K formula BUT rescaled: K_final = 17.67 * K / sum(D_bar*K)   [local rescaling]
  TMDSI  : no K at all — Z = Phi^{-1}(F(d))  [probability transform]

STEP 7 differences:
  PDSI + scPDSI : Z = K * d
  TMDSI         : Z ~ N(0,1) by construction from Step 6

STEP 8 differences:
  PDSI   : p = 0.897, q = 1/3   [Palmer's empirical constants, Kansas/Iowa only]
  scPDSI : fit line cumZ = m*t + b to extreme spell durations
             p = 1 - m/(m+b),  q = C/(m+b)   C=-4 dry, C=+4 wet
  TMDSI  : phi_m = 1 - (RO_m + L_m)/AWC   (linear reservoir)
             w_m  = sqrt(1 - phi_m^2)       (AR(1) stationarity)

STEP 9: AR(1) recursion with backtracking — same logic for all three.
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Literal

from tmdsi.pet.thornthwaite import thornthwaite_pet
# from tmdsi.pet.hargreaves   import hargreaves_pet

from tmdsi.water_balance    import run_water_balance, cafec_coefficients

PETMethod  = Literal["thornthwaite", "hargreaves"]
DistMethod = Literal["normal", "pearson3", "best"]

# ============================================================
# RESULT CONTAINERS
# ============================================================

@dataclass
class DroughtResult:
    """Returned by all three index functions."""
    index : np.ndarray    # the main drought index time series
    phdi  : np.ndarray    # Palmer Hydrological Drought Index
    pmdi  : np.ndarray    # Palmer Modified Drought Index
    z     : np.ndarray    # Z-index
    cafec : dict          # CAFEC coefficients {alpha,beta,gamma,delta}
    # scPDSI / PDSI specific
    p     : float | np.ndarray = 0.897   # AR(1) persistence
    q     : float | np.ndarray = 1/3     # AR(1) scaling
    # TMDSI specific
    phi   : np.ndarray = None   # monthly phi, shape (12,)
    w     : np.ndarray = None   # monthly w,   shape (12,)
    dist  : list       = None   # distribution per month


# ============================================================
# SHARED STEPS 1–5
# ============================================================

def _compute_pet(tmax, tmin, lat, year0, method, **kw):
    if method == "thornthwaite":
        return thornthwaite_pet(tmax, tmin, lat, year0)
    elif method == "hargreaves":
        return hargreaves_pet(tmax, tmin, lat, year0, **kw)
    raise ValueError(f"Unknown PET method: {method}")


def _compute_d(precip_2d, pet_2d, wb, cafec):
    """Moisture departure d = P - P_hat for every month."""
    n_years = precip_2d.shape[0]
    d = np.zeros((n_years, 12))
    for yr in range(n_years):
        for mo in range(12):
            phat = (cafec["alpha"][mo] * pet_2d[yr, mo]
                  + cafec["beta"][mo]  * wb["pr"][yr, mo]
                  + cafec["gamma"][mo] * wb["sp"][yr, mo]
                  - cafec["delta"][mo] * wb["pl"][yr, mo])
            d[yr, mo] = precip_2d[yr, mo] - phat
    return d


def _steps_1_to_5(precip, tmax, tmin, awc, lat, year0, cs, ce, pet_method, fitting_params, kw):
    """Run shared water balance pipeline. Returns (precip_2d, pet_2d, wb, cafec, d, n_years)."""
    n_years = len(precip) // 12
    pet      = _compute_pet(tmax, tmin, lat, year0, pet_method, **kw)
    p2d      = precip[:n_years*12].reshape(n_years, 12)
    pet2d    = pet   [:n_years*12].reshape(n_years, 12)
    n_calib  = ce - cs + 1
    wb       = run_water_balance(p2d, pet2d, awc, cs, ce)
    if fitting_params and all(k in fitting_params for k in ("alpha","beta","gamma","delta")):
        cafec = {k: np.asarray(fitting_params[k]) for k in ("alpha","beta","gamma","delta")}
    else:
        cafec = cafec_coefficients(wb, n_calib)
    d = _compute_d(p2d, pet2d, wb, cafec)
    return p2d, pet2d, wb, cafec, d, n_years, n_calib


# ============================================================
# STEP 6–7: Three ways to get Z
# ============================================================

def _palmer_K(wb, precip_2d, pet_2d, cafec, d, cs, ce, n_calib, rescale=False):
    """
    Compute Palmer's K factor and Z = K*d.
    rescale=False → original PDSI (global calibration)
    rescale=True  → scPDSI (17.67 local rescaling, Wells 2004)
    """
    # mean demand/supply ratio per calendar month
    trat = (wb["petsum"] + wb["rsum"] + wb["rosum"]) / np.maximum(
            wb["psum"] + wb["tlsum"], 1e-9)

    # mean |d| per calendar month over calibration period
    sabsd = np.zeros(12)
    n_years = precip_2d.shape[0]
    for yr in range(cs, ce + 1):
        for mo in range(12):
            phat = (cafec["alpha"][mo] * pet_2d[yr, mo]
                  + cafec["beta"][mo]  * wb["pr"][yr, mo]
                  + cafec["gamma"][mo] * wb["sp"][yr, mo]
                  - cafec["delta"][mo] * wb["pl"][yr, mo])
            sabsd[mo] += abs(precip_2d[yr, mo] - phat)
    dbar  = sabsd / n_calib

    # Palmer's K formula (Eq. 5 in Palmer 1965)
    Khat = 1.5 * np.log10((trat + 2.8) / np.maximum(dbar, 1e-9)) + 0.5

    if rescale:
        # Wells (2004) self-calibrating rescaling: forces ~2% global extreme frequency
        swtd = np.sum(dbar * Khat)
        K = 17.67 * Khat / swtd if swtd != 0 else Khat
    else:
        K = Khat

    # Z = K * d
    Z = K[np.newaxis, :] * d   # broadcast over years

    return K, Z


def _prob_Z(d, cs, ce, dist_method):
    """
    TMDSI Step 6: Z_m = Phi^{-1}(F_{d,m}(d_m))
    Fit distribution to d over calibration period per calendar month,
    then probability-transform to standard normal.
    Z is exactly N(0,1) by construction.
    """
    n_years = d.shape[0]
    Z = np.zeros((n_years, 12))
    dist_names = []
    eps = 1e-6

    for mo in range(12):
        d_cal = d[cs:ce+1, mo]
        d_cal = d_cal[~np.isnan(d_cal)]

        # fit distribution
        if dist_method == "normal":
            mu, sig = np.mean(d_cal), np.std(d_cal, ddof=1)
            dist = stats.norm(loc=mu, scale=max(sig, 1e-9))
            name = "normal"
        elif dist_method == "pearson3":
            sk, loc, sc = stats.pearson3.fit(d_cal)
            dist = stats.pearson3(sk, loc=loc, scale=sc)
            name = "pearson3"
        else:  # best
            mu, sig = np.mean(d_cal), np.std(d_cal, ddof=1)
            dn = stats.norm(loc=mu, scale=max(sig,1e-9))
            pn = stats.kstest(d_cal, dn.cdf).pvalue
            sk, loc, sc = stats.pearson3.fit(d_cal)
            dp = stats.pearson3(sk, loc=loc, scale=sc)
            pp = stats.kstest(d_cal, dp.cdf).pvalue
            dist, name = (dp, "pearson3") if pp > pn else (dn, "normal")

        dist_names.append(name)
        for yr in range(n_years):
            p = float(np.clip(dist.cdf(d[yr, mo]), eps, 1-eps))
            Z[yr, mo] = stats.norm.ppf(p)

    return Z, dist_names


# ============================================================
# STEP 8: Three ways to get p, q (AR(1) coefficients)
# ============================================================

def _palmer_pq():
    """PDSI: global fixed empirical constants (Palmer 1965)."""
    return 0.897, 1.0/3.0   # p, q


def _fit_mb(Z_flat, max_dur=None):
    """
    scPDSI Step 8: fit slope m and intercept b to the
    (duration, cumulative_Z) relationship for extreme spells.

    For each spell duration t = 1 ... max_dur:
      Find the single t-month window with minimum (dry) and maximum (wet) cumulative Z.
    Then fit: cumZ = m*t + b  via least squares.

    Returns (m_dry, b_dry, m_wet, b_wet)
    """
    z = Z_flat.copy()
    n = len(z)
    if max_dur is None:
        max_dur = min(n // 4, 72)   # up to 72 months (6 years)

    durations    = np.arange(1, max_dur + 1)
    cumZ_dry     = np.zeros(max_dur)
    cumZ_wet     = np.zeros(max_dur)

    for t in durations:
        t = int(t)
        # rolling sum of length t
        windows = np.array([z[i:i+t].sum() for i in range(n - t + 1)])
        cumZ_dry[t-1] = windows.min()
        cumZ_wet[t-1] = windows.max()

    # Least squares fit: cumZ = m*t + b
    def _fit_line(x, y):
        A = np.column_stack([x, np.ones_like(x)])
        result = np.linalg.lstsq(A, y, rcond=None)
        m, b = result[0]
        return m, b

    m_dry, b_dry = _fit_line(durations.astype(float), cumZ_dry)
    m_wet, b_wet = _fit_line(durations.astype(float), cumZ_wet)

    return m_dry, b_dry, m_wet, b_wet


def _mb_to_pq(m, b, C):
    denom = m + b
    if abs(denom) < 1e-9:
        return 0.897, 1/3
    p = 1.0 - m / denom
    q = abs(C / denom)      # force positive
    p = float(np.clip(p, 0.5,  0.99))
    q = float(np.clip(q, 0.01, 0.50))
    return p, q

def _derive_phi_w(wb, awc, n_calib):
    """
    TMDSI Steps 7–8:
      phi_m = 1 - (RO_m + L_m) / AWC     (linear reservoir)
      w_m   = sqrt(1 - phi_m^2)           (AR(1) stationarity → Var(X)=1)
    """
    k   = (wb["rosum"]/n_calib + wb["tlsum"]/n_calib) / max(awc, 1e-6)
    phi = np.clip(1.0 - k, 0.10, 0.99)
    w   = np.sqrt(1.0 - phi**2)
    return phi, w


# ============================================================
# STEP 9: AR(1) recursion + backtracking (shared by all three)
# ============================================================

def _case(prob, x1, x2, x3):
    if x3 == 0:
        return x1 if abs(x1) > abs(x2) else x2
    if prob <= 0 or prob >= 100:
        return x3
    frac = prob / 100.0
    return (1-frac)*x3 + frac*(x1 if x3 <= 0 else x2)


def _ar1_recursion(Z, p_dry, q_dry, p_wet, q_wet, n_years, monthly=False, phi=None, w=None):
    """
    Full Palmer AR(1) recursion with spell-transition backtracking.

    For PDSI/scPDSI: p_dry/q_dry/p_wet/q_wet are scalars.
    For TMDSI: pass monthly=True and phi,w arrays shape (12,).
    Returns (index, phdi, pmdi) each shape (n_years, 12).
    """
    index = np.full((n_years, 12), np.nan)
    phdi  = np.full((n_years, 12), np.nan)
    pmdi  = np.full((n_years, 12), np.nan)
    ppr   = np.zeros((n_years, 12))
    px1   = np.zeros((n_years, 12))
    px2   = np.zeros((n_years, 12))
    px3   = np.zeros((n_years, 12))
    X     = np.zeros((n_years, 12))

    # backtrack buffer
    BUF = 200
    sx  = np.zeros(BUF); sx1 = np.zeros(BUF)
    sx2 = np.zeros(BUF); sx3 = np.zeros(BUF)
    idxj = np.full(BUF, -1, dtype=int)
    idxm = np.full(BUF, -1, dtype=int)
    k8 = 0; k8max = 0

    v = 0.0; pro = 0.0
    x1s = 0.0; x2s = 0.0; x3s = 0.0
    iass = 0; ze = 0.0; pv = 0.0

    def get_pq(mo, spell_type):
        if monthly:
            p_ = float(phi[mo]); q_ = float(w[mo])
            return p_, q_
        if spell_type == "dry":
            return p_dry, q_dry
        return p_wet, q_wet

    def write_out(yr, mo):
        index[yr, mo] = X[yr, mo]
        phdi [yr, mo] = px3[yr, mo] if px3[yr, mo] != 0 else X[yr, mo]
        pmdi [yr, mo] = _case(ppr[yr, mo], px1[yr, mo], px2[yr, mo], px3[yr, mo])

    def assign(yr, mo):
        nonlocal k8, iass
        sx[k8] = X[yr, mo]
        isave  = iass
        if k8 == 0:
            write_out(yr, mo); return
        if iass == 3:
            for i in range(k8):
                sx[i] = sx3[i]
        else:
            for i in range(k8-1, -1, -1):
                if isave == 2:
                    if sx2[i] == 0: isave = 1; sx[i] = sx1[i]
                    else:           sx[i] = sx2[i]
                else:
                    if sx1[i] == 0: isave = 2; sx[i] = sx2[i]
                    else:           sx[i] = sx1[i]
        for idx in range(k8+1):
            j_, m_ = idxj[idx], idxm[idx]
            index[j_, m_] = sx[idx]
            phdi [j_, m_] = px3[j_, m_] if px3[j_, m_] != 0 else sx[idx]
            pmdi [j_, m_] = _case(ppr[j_, m_], px1[j_, m_], px2[j_, m_], px3[j_, m_])
        k8 = 0

    def save_state(yr, mo):
        nonlocal v, pro, x1s, x2s, x3s
        v = pv; pro = ppr[yr, mo]
        x1s = px1[yr, mo]; x2s = px2[yr, mo]; x3s = px3[yr, mo]

    def stmt210(yr, mo):
        nonlocal pv, k8, iass
        pv = 0.0
        px1[yr, mo] = 0.0; px2[yr, mo] = 0.0; ppr[yr, mo] = 0.0
        p_, q_ = get_pq(mo, "dry")
        px3[yr, mo] = p_ * x3s + q_ * Z[yr, mo]
        X  [yr, mo] = px3[yr, mo]
        if k8 == 0: write_out(yr, mo)
        else:       iass = 3; assign(yr, mo)
        save_state(yr, mo)

    def stmt200(yr, mo):
        nonlocal k8, k8max, iass
        p_d, q_d = get_pq(mo, "dry")
        p_w, q_w = get_pq(mo, "wet")
        px1[yr, mo] = max(0.0, p_d * x1s + q_d * Z[yr, mo])
        if px1[yr, mo] >= 1 and px3[yr, mo] == 0:
            X[yr, mo] = px1[yr, mo]; px3[yr, mo] = px1[yr, mo]; px1[yr, mo] = 0.0
            iass = 1; assign(yr, mo); save_state(yr, mo); return
        px2[yr, mo] = min(0.0, p_w * x2s + q_w * Z[yr, mo])
        if px2[yr, mo] <= -1 and px3[yr, mo] == 0:
            X[yr, mo] = px2[yr, mo]; px3[yr, mo] = px2[yr, mo]; px2[yr, mo] = 0.0
            iass = 2; assign(yr, mo); save_state(yr, mo); return
        if px3[yr, mo] == 0:
            if px1[yr, mo] == 0:
                X[yr, mo] = px2[yr, mo]; iass = 2; assign(yr, mo); save_state(yr, mo); return
            if px2[yr, mo] == 0:
                X[yr, mo] = px1[yr, mo]; iass = 1; assign(yr, mo); save_state(yr, mo); return
        # undetermined — buffer
        sx1[k8] = px1[yr, mo]; sx2[k8] = px2[yr, mo]; sx3[k8] = px3[yr, mo]
        X[yr, mo] = px3[yr, mo]; idxj[k8] = yr; idxm[k8] = mo
        k8 += 1; k8max = max(k8max, k8); save_state(yr, mo)

    def stmt190(yr, mo):
        nonlocal pv, ppr, px3
        q_ = ze if pro == 100 else ze + v
        ppr[yr, mo] = min(pv / q_ * 100, 100.0) if abs(q_) > 1e-9 else 100.0
        if ppr[yr, mo] >= 100:
            ppr[yr, mo] = 100.0; px3[yr, mo] = 0.0
        else:
            p_, q_ar = get_pq(mo, "dry")
            px3[yr, mo] = p_ * x3s + q_ar * Z[yr, mo]
        stmt200(yr, mo)

    def stmt180(yr, mo):
        nonlocal pv, ze
        uw = Z[yr, mo] + 0.15
        pv = uw + max(v, 0.0)
        if pv <= 0: stmt210(yr, mo); return
        # for TMDSI derive ze from phi/w, for PDSI/scPDSI use Palmer's -2.691
        if monthly:
            ze = x3s * (phi[mo] - 1.0) / w[mo]  - 0.5
        else:
            ze = -2.691 * x3s - 1.5
        stmt190(yr, mo)

    def stmt170(yr, mo):
        nonlocal pv, ze
        ud = Z[yr, mo] - 0.15
        pv = ud + min(v, 0.0)
        if pv >= 0: stmt210(yr, mo); return
        if monthly:
            ze = x3s * (phi[mo] - 1.0) / w[mo] + 0.5
        else:
            ze = -2.691 * x3s + 1.5
        stmt190(yr, mo)

    for yr in range(n_years):
        for mo in range(12):
            if k8 >= BUF - 1:   # expand buffer if needed
                BUF_ = BUF + 50
                for arr_ in (sx, sx1, sx2, sx3):
                    arr_.resize(BUF_, refcheck=False)
                idxj.resize(BUF_); idxm.resize(BUF_)
            idxj[k8] = yr; idxm[k8] = mo
            no_aba = pro in (0.0, 100.0)
            if no_aba:
                if -0.5 <= x3s <= 0.5:
                    pv = 0.0; ppr[yr,mo] = 0.0; px3[yr,mo] = 0.0; stmt200(yr, mo)
                elif x3s > 0.5:
                    stmt210(yr, mo) if Z[yr,mo] >= 0.15 else stmt170(yr, mo)
                else:
                    stmt210(yr, mo) if Z[yr,mo] <= -0.15 else stmt180(yr, mo)
            else:
                stmt170(yr, mo) if x3s > 0 else stmt180(yr, mo)

    # flush remaining buffer
    for k in range(k8max):
        j_, m_ = idxj[k], idxm[k]
        index[j_, m_] = X[j_, m_]
        phdi [j_, m_] = px3[j_, m_] if px3[j_, m_] != 0 else X[j_, m_]
        pmdi [j_, m_] = _case(ppr[n_years-1, 11], px1[n_years-1, 11],
                               px2[n_years-1, 11], px3[n_years-1, 11])

    return index, phdi, pmdi


# ============================================================
# PUBLIC API — three compute functions
# ============================================================

def _base_args(precip, tmax, tmin, awc, latitude, data_start_year,
               calibration_start_year, calibration_end_year,
               pet_method, fitting_params, pet_kwargs):
    """Validate inputs and compute shared pipeline."""
    precip = np.clip(np.asarray(precip, float), 0, None)
    if np.all(np.isnan(precip)):
        nan = np.full(precip.shape, np.nan)
        return None
    cs = calibration_start_year - data_start_year
    ce = calibration_end_year   - data_start_year
    return _steps_1_to_5(precip, tmax, tmin, awc, latitude,
                         data_start_year, cs, ce,
                         pet_method, fitting_params, pet_kwargs)


def compute_pdsi(
    precip, tmax, tmin, awc, latitude,
    data_start_year, calibration_start_year, calibration_end_year,
    pet_method: PETMethod = "thornthwaite",
    fitting_params=None, **pet_kwargs
) -> DroughtResult:
    """
    Palmer (1965) PDSI.
    Uses FIXED global constants: p=0.897, q=1/3, Palmer's original K calibration.
    Same K formula as Wells but WITHOUT the 17.67 local rescaling.
    """
    precip = np.clip(np.asarray(precip, float), 0, None)
    cs = calibration_start_year - data_start_year
    ce = calibration_end_year   - data_start_year
    p2d, pet2d, wb, cafec, d, n_years, n_calib = _steps_1_to_5(
        precip, tmax, tmin, awc, latitude, data_start_year,
        cs, ce, pet_method, fitting_params, pet_kwargs)

    # Step 6–7: Palmer K (no rescaling), Z = K*d
    K, Z = _palmer_K(wb, p2d, pet2d, cafec, d, cs, ce, n_calib, rescale=False)

    # Step 8: FIXED p, q (Palmer's empirical constants)
    p, q = _palmer_pq()

    # Step 9: AR(1) backtracking
    orig = len(precip)
    idx, phdi_arr, pmdi_arr = _ar1_recursion(Z, p, q, p, q, n_years)

    flat = lambda a: a.flatten()[:orig]
    return DroughtResult(
        index=flat(idx), phdi=flat(phdi_arr), pmdi=flat(pmdi_arr),
        z=flat(Z), cafec=cafec, p=p, q=q
    )


def compute_scpdsi(
    precip, tmax, tmin, awc, latitude,
    data_start_year, calibration_start_year, calibration_end_year,
    pet_method: PETMethod = "thornthwaite",
    fitting_params=None, **pet_kwargs
) -> DroughtResult:
    """
    Wells et al. (2004) Self-Calibrating PDSI (scPDSI).

    Two key improvements over PDSI:
    1. K is rescaled locally via 17.67 factor (forces ~2% extreme frequency globally)
    2. p and q are derived PER LOCATION by fitting a line to
       (spell duration, cumulative Z) for the most extreme wet/dry spells.
       From Wells (2004) Eqs. (19),(20):
         p = 1 - m/(m+b),  q = C/(m+b)   where C=-4 (dry) or +4 (wet)
    """
    precip = np.clip(np.asarray(precip, float), 0, None)
    cs = calibration_start_year - data_start_year
    ce = calibration_end_year   - data_start_year
    p2d, pet2d, wb, cafec, d, n_years, n_calib = _steps_1_to_5(
        precip, tmax, tmin, awc, latitude, data_start_year,
        cs, ce, pet_method, fitting_params, pet_kwargs)

    # Step 6–7: Local K with 17.67 rescaling, Z = K*d
    K, Z = _palmer_K(wb, p2d, pet2d, cafec, d, cs, ce, n_calib, rescale=True)

    # Step 8: Derive p, q from the data via m, b fit
    # --- use only calibration period Z to fit m, b ---
    Z_calib = Z[cs:ce+1, :].flatten()
    m_dry, b_dry, m_wet, b_wet = _fit_mb(Z_calib)

    p_dry, q_dry = _mb_to_pq(m_dry, b_dry, C=-4.0)   # dry spell factors
    p_wet, q_wet = _mb_to_pq(m_wet, b_wet, C=+4.0)   # wet spell factors

    # Step 9: AR(1) backtracking with local p,q
    orig = len(precip)
    idx, phdi_arr, pmdi_arr = _ar1_recursion(Z, p_dry, q_dry, p_wet, q_wet, n_years)

    flat = lambda a: a.flatten()[:orig]
    return DroughtResult(
        index=flat(idx), phdi=flat(phdi_arr), pmdi=flat(pmdi_arr),
        z=flat(Z), cafec=cafec, p=p_dry, q=q_dry
    )


def compute_tmdsi(
    precip, tmax, tmin, awc, latitude,
    data_start_year, calibration_start_year, calibration_end_year,
    pet_method: PETMethod = "thornthwaite",
    dist_method: DistMethod = "best",
    fitting_params=None, **pet_kwargs
) -> DroughtResult:
    """
    Tripathy & Mishra (in prep.) TMDSI.

    Replaces ALL empirical constants with derived values:
    Step 6: Z = Phi^{-1}(F(d))  — probability transform, Z ~ N(0,1) everywhere
    Step 7: phi_m = 1 - (RO_m + L_m)/AWC  — from linear reservoir model
    Step 8: w_m = sqrt(1 - phi_m^2)       — from AR(1) stationarity condition
    """
    precip = np.clip(np.asarray(precip, float), 0, None)
    cs = calibration_start_year - data_start_year
    ce = calibration_end_year   - data_start_year
    p2d, pet2d, wb, cafec, d, n_years, n_calib = _steps_1_to_5(
        precip, tmax, tmin, awc, latitude, data_start_year,
        cs, ce, pet_method, fitting_params, pet_kwargs)

    # Step 6: Probability-based standardization → Z ~ N(0,1)
    Z, dist_names = _prob_Z(d, cs, ce, dist_method)

    # Steps 7–8: Physics-derived phi and w (monthly, shape 12)
    phi, w = _derive_phi_w(wb, awc, n_calib)

    # Step 9: AR(1) with month-specific phi, w
    orig = len(precip)
    idx, phdi_arr, pmdi_arr = _ar1_recursion(
        Z, None, None, None, None, n_years, monthly=True, phi=phi, w=w)

    flat = lambda a: a.flatten()[:orig]
    return DroughtResult(
        index=flat(idx), phdi=flat(phdi_arr), pmdi=flat(pmdi_arr),
        z=flat(Z), cafec=cafec,
        phi=phi, w=w, dist=dist_names
    )