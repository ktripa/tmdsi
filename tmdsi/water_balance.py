"""
Author: Kumar Puran Tripathy
Palmer two-layer bucket model: soil water balance.
Computes ET, recharge, runoff, loss for each month.
"""

import numpy as np

AWC_TOP = 25.4  # surface layer capacity fixed at 1 inch = 25.4 mm


def _potential_loss(pet: float, ss: float, su: float, awc_bot: float) -> float:
    """
    ss: surface storage (mm)
    su: storage underlaying layer (mm)
    pet: potential evapotranspiration (mm)
    awc_bot: available water capacity of underlying layer (mm)
    """
    if ss >= pet:
        return pet
    return min(ss + su, ((pet - ss) * su) / (awc_bot + AWC_TOP) + ss)


def _recharge_step(
    p: float, pet: float, ss: float, su: float, awc_bot: float
) -> tuple[float, float, float, float, float, float]:
    """
    Single-month water balance.
    Returns: (et, loss, recharge, runoff, new_ss, new_su)
    """
    if p >= pet:
        et, loss = pet, 0.0
        excess = p - pet
        if excess > (AWC_TOP - ss):
            rs = AWC_TOP - ss
            new_ss = AWC_TOP
            ru = min(excess - rs, awc_bot - su)
            ro = max(excess - rs - ru, 0.0)
            new_su = su + ru
            r = rs + ru
        else:
            r, ro = excess, 0.0
            new_ss = ss + excess
            new_su = su
    else:
        r, ro = 0.0, 0.0
        deficit = pet - p
        if ss >= deficit:
            sl, new_ss = deficit, ss - deficit
            ul, new_su = 0.0, su
        else:
            sl, new_ss = ss, 0.0
            awc = awc_bot + AWC_TOP
            ul = min(su, (deficit - sl) * su / awc)
            new_su = su - ul
        loss = sl + ul
        et = p + sl + ul

    return et, loss, r, ro, new_ss, new_su


def run_water_balance(
    precip: np.ndarray,
    pet: np.ndarray,
    awc: float,
    calib_start_idx: int,
    calib_end_idx: int,
) -> dict:
    """
    Run Palmer two-layer water balance over full record.

    Parameters
    ----------
    precip : np.ndarray, shape (n_years, 12), mm
    pet    : np.ndarray, shape (n_years, 12), mm
    awc    : float, total available water capacity (mm)
    calib_start_idx : int, calibration start year index
    calib_end_idx   : int, calibration end year index (inclusive)

    Returns
    -------
    dict with monthly arrays (n_years, 12) for:
      sp, pl, pr, r, loss, et, ro, ss, su
    and calibration-period monthly sums (shape 12):
      psum, spsum, petsum, plsum, prsum, rsum, tlsum, etsum, rosum
    """
    n_years, _ = precip.shape
    awc_bot = max(awc - AWC_TOP, 0.0)

    # output arrays
    keys = ["sp", "pl", "pr", "r", "loss", "et", "ro", "ss", "su"]
    out = {k: np.full((n_years, 12), np.nan) for k in keys}
    sums = {k: np.zeros(12) for k in
            ["psum", "spsum", "petsum", "plsum", "prsum", "rsum", "tlsum", "etsum", "rosum"]}

    ss, su = AWC_TOP, awc_bot  # initialise full

    for yr in range(n_years):
        for mo in range(12):
            p, pe = precip[yr, mo], pet[yr, mo]
            sp = ss + su
            pr = awc - sp
            pl = _potential_loss(pe, ss, su, awc_bot)
            et, loss, r, ro, ss, su = _recharge_step(p, pe, ss, su, awc_bot)

            out["sp"][yr, mo] = sp
            out["pl"][yr, mo] = pl
            out["pr"][yr, mo] = pr
            out["r"][yr, mo]  = r
            out["loss"][yr, mo] = loss
            out["et"][yr, mo] = et
            out["ro"][yr, mo] = ro
            out["ss"][yr, mo] = ss
            out["su"][yr, mo] = su

            if calib_start_idx <= yr <= calib_end_idx:
                sums["psum"][mo]   += p
                sums["spsum"][mo]  += sp
                sums["petsum"][mo] += pe
                sums["plsum"][mo]  += pl
                sums["prsum"][mo]  += pr
                sums["rsum"][mo]   += r
                sums["tlsum"][mo]  += loss
                sums["etsum"][mo]  += et
                sums["rosum"][mo]  += ro

    return {**out, **sums}


def cafec_coefficients(wb: dict, n_calib_years: int) -> dict:
    """
    Compute CAFEC coefficients alpha, beta, gamma, delta
    from water balance calibration sums.

    α = ET_mean / PET_mean
    β = R_mean  / PR_mean
    γ = RO_mean / SP_mean
    δ = L_mean  / PL_mean
    """
    def safe_ratio(num, den, default=1.0):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(den != 0, num / den, np.where(num == 0, default, 0.0))
        return r

    nc = n_calib_years
    alpha = safe_ratio(wb["etsum"] / nc, wb["petsum"] / nc)
    beta  = safe_ratio(wb["rsum"]  / nc, wb["prsum"]  / nc)
    gamma = safe_ratio(wb["rosum"] / nc, wb["spsum"]  / nc)
    delta = safe_ratio(wb["tlsum"] / nc, wb["plsum"]  / nc)

    return {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta}