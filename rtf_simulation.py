#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI Timing Framework (RTF): simulation study and figures.

Discrete-time model (daily grid, monthly reporting).
All random seeds are fixed. Run:  python rtf_simulation.py
Outputs: fig1..fig6 (PNG, 300 dpi) and results.json with every number quoted in the paper.

Environment used for the paper: Python 3.11, NumPy >= 1.26, SciPy >= 1.12, Matplotlib >= 3.8.
"""
import json, math
import numpy as np
from scipy import stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20260915
rng = np.random.default_rng(SEED)
OUT = {}

# ----------------------------------------------------------------------------
# 0. Base-case parameters (Table 3)
# ----------------------------------------------------------------------------
B        = 100_000.0     # total campaign spend
T_DAYS   = 365           # active campaign horizon (days)
H_DAYS   = 540           # reporting horizon (18 months)
DAY      = 30            # days per reporting month
A_LR     = 1.9           # long-run ROAS (total contribution / total spend)
C_LOCKED = 20_000.0      # non-cancellable commitments accrued over the campaign
C_CANCEL = 10_000.0      # cancellable commitments accrued over the campaign
PHI      = 0.5           # cancellation penalty share
SIGMA    = 0.07          # multiplicative revenue noise (monthly level)
N_SIM    = 2000
EPS_TAIL = 0.05
L_MAX    = 730           # kernel truncation for computation (days)

C_EFF_TOTAL = C_LOCKED + PHI * C_CANCEL          # 25 000
KAPPA = C_EFF_TOTAL / B                          # marginal commitment per unit spend = 0.25

# ----------------------------------------------------------------------------
# 1. Kernel families (Table 2), discretised on the daily grid
# ----------------------------------------------------------------------------
def kernel(family, L=L_MAX):
    ell = np.arange(L + 1)
    if family == "exp":
        w = stats.expon.pdf(ell + 0.5, scale=15.0)
    elif family == "gamma":
        w = stats.gamma.pdf(ell + 0.5, a=3.0, scale=25.0)
    elif family == "lognorm":
        sigma = 0.55; mu = math.log(90.0) - sigma**2 / 2
        w = stats.lognorm.pdf(ell + 0.5, s=sigma, scale=math.exp(mu))
    else:
        raise ValueError(family)
    return w / w.sum()

def kernel_stats(w):
    ell = np.arange(len(w))
    mean = (ell * w).sum()
    sd = math.sqrt(((ell - mean) ** 2 * w).sum())
    cdf = np.cumsum(w)
    L_eps = int(np.searchsorted(cdf, 1 - EPS_TAIL))     # practical lag horizon
    return dict(mean=mean, cv=sd / mean, L_eps=L_eps)

W = {f: kernel(f) for f in ["exp", "gamma", "lognorm"]}
OUT["kernels"] = {f: kernel_stats(W[f]) for f in W}

# ----------------------------------------------------------------------------
# 2. Deterministic paths for the base case (uniform daily spend)
# ----------------------------------------------------------------------------
def paths(w, A=A_LR, spend=None, horizon=H_DAYS, T=T_DAYS, kappa=KAPPA, A_profile=None):
    """Return daily cumulative spend C, realised contribution R, pipeline R_pipe, commitments C_eff.
    A_profile: optional array of long-run ROAS by spend date (fatigue / non-monotone scenarios)."""
    days = np.arange(1, horizon + 1)
    if spend is None:
        spend = np.where(days <= T, B / T, 0.0)
    if A_profile is None:
        A_profile = np.full(horizon, A)
    Wcdf = np.cumsum(w)
    C = np.cumsum(spend)
    R = np.zeros(horizon); Rp = np.zeros(horizon)
    for t in range(horizon):
        lags = t - np.arange(t + 1)                    # lag of each past spend day
        real = Wcdf[np.minimum(lags, len(w) - 1)]      # share of response realised by t
        contrib = A_profile[:t + 1] * spend[:t + 1]
        R[t] = (contrib * real).sum()
        Rp[t] = (contrib * (1 - real)).sum()
    C_eff = kappa * np.minimum(C, B)
    return dict(days=days, spend=spend, C=C, R=R, Rp=Rp, C_eff=C_eff)

def rois(P):
    C, R, Rp, Ce = P["C"], P["R"], P["Rp"], P["C_eff"]
    S = C + Ce
    with np.errstate(divide="ignore", invalid="ignore"):
        out = dict(
            naive   = (R - C) / C,
            contract= (R - S) / S,
            lag     = (R + Rp - C) / C,
            rtf     = (R + Rp - S) / S,
            delta_C = R * Ce / (C * S),
            delta_L = Rp / S,
            roas    = R / C,
            r       = Ce / C,
            p       = Rp / C,
        )
    out["delta"] = out["delta_C"] - out["delta_L"]
    return out

base = paths(W["gamma"])
rb = rois(base)
months = np.arange(1, H_DAYS // DAY + 1)
midx = months * DAY - 1

def at_month(arr, m): return float(arr[m * DAY - 1])
OUT["base_case"] = {}
for m in [3, 6, 12, 18]:
    OUT["base_case"][f"month_{m}"] = {k: at_month(v, m) for k, v in rb.items()} | {
        "C": at_month(base["C"], m), "R": at_month(base["R"], m),
        "R_pipe": at_month(base["Rp"], m), "C_eff": at_month(base["C_eff"], m)}

# ----------------------------------------------------------------------------
# 3. Monte-Carlo corridors for Figure 2 (H1 illustration)
# ----------------------------------------------------------------------------
def monte_carlo(P, n=N_SIM, sigma=SIGMA, rng=rng):
    R_m = P["R"][midx]; Rp_m = P["Rp"][midx]; C_m = P["C"][midx]; Ce_m = P["C_eff"][midx]
    S_m = C_m + Ce_m
    eps = rng.normal(0, sigma, size=(n, len(midx)))
    R_obs = R_m * (1 + eps)
    naive = (R_obs - C_m) / C_m
    contract = (R_obs - S_m) / S_m
    rtf = (R_obs + Rp_m - S_m) / S_m
    q = lambda a: np.percentile(a, [15, 50, 85], axis=0)
    return dict(naive=q(naive), contract=q(contract), rtf=q(rtf), mean_naive=naive.mean(0),
                mean_contract=contract.mean(0), mean_rtf=rtf.mean(0))

mc = monte_carlo(base)
OUT["mc_month18"] = dict(naive=float(mc["mean_naive"][-1]), contract=float(mc["mean_contract"][-1]),
                        rtf=float(mc["mean_rtf"][-1]),
                        naive_q15=float(mc["naive"][0][-1]), naive_q85=float(mc["naive"][2][-1]),
                        contract_q15=float(mc["contract"][0][-1]), contract_q85=float(mc["contract"][2][-1]))
OUT["mc_month12"] = dict(naive=float(mc["mean_naive"][11]), contract=float(mc["mean_contract"][11]),
                        rtf=float(mc["mean_rtf"][11]))

# ----------------------------------------------------------------------------
# 4. Baseline comparison of four ROI measures vs the full-horizon benchmark
# ----------------------------------------------------------------------------
bench = OUT["base_case"]["month_18"]["contract"]     # ROI at full realisation (pipeline exhausted)
OUT["benchmark_full_horizon"] = bench
OUT["measure_comparison"] = {}
for m in [3, 6, 9, 12, 15, 18]:
    d = OUT["base_case"].get(f"month_{m}") or {k: at_month(v, m) for k, v in rb.items()}
    OUT["measure_comparison"][f"month_{m}"] = {k: d[k] for k in ["naive", "contract", "lag", "rtf"]}

# ----------------------------------------------------------------------------
# 5. Tail risk by kernel family (H2): pipeline share and tail mass beyond L_eps at end of campaign
# ----------------------------------------------------------------------------
OUT["tail_by_family"] = {}
for f in W:
    P = paths(W[f]); r = rois(P)
    w = W[f]; L_eps = OUT["kernels"][f]["L_eps"]
    OUT["tail_by_family"][f] = dict(
        p_month12=float(r["p"][12 * DAY - 1]),
        delta_L_month12=float(r["delta_L"][12 * DAY - 1]),
        delta_month12=float(r["delta"][12 * DAY - 1]),
        tail_mass_beyond_L_eps=float(w[L_eps + 1:].sum()),
        L_eps=L_eps)

# ----------------------------------------------------------------------------
# 6. Stopping rule: fatigue (monotone) and non-monotone effectiveness (H4)
# ----------------------------------------------------------------------------
def stopping_analysis(A_profile, label):
    P = paths(W["gamma"], A_profile=A_profile)
    # Full-response net value Pi(t) if stopped after day t (expected, noise-free)
    Pi = P["R"] + P["Rp"] - P["C"] - P["C_eff"]
    # Monthly decision grid during the active campaign
    grid = np.arange(DAY, T_DAYS + 1, DAY)
    Pi_g = Pi[grid - 1]
    g = np.diff(Pi_g)                                   # one-step expected gain of continuing
    osla = next((i for i, x in enumerate(g) if x <= 0), len(g))   # index in grid where OSLA stops
    bellman = int(np.argmax(Pi_g))                      # finite-horizon backward induction == argmax (deterministic)
    S = P["C"] + P["C_eff"]
    roi_full = (P["R"] + P["Rp"] - S) / S
    return dict(label=label,
                osla_stop_day=int(grid[osla]), bellman_stop_day=int(grid[bellman]),
                Pi_osla=float(Pi_g[osla]), Pi_bellman=float(Pi_g[bellman]), Pi_full=float(Pi_g[-1]),
                roi_full_at_osla=float(roi_full[grid[osla] - 1]), roi_full_end=float(roi_full[grid[-1] - 1]),
                C_at_osla=float(P["C"][grid[osla] - 1]),
                monotone=bool(all(g[i] <= 0 for i in range(osla, len(g)))),
                g=[float(x) for x in g])

days = np.arange(1, H_DAYS + 1)
Cpath = np.cumsum(np.where(days <= T_DAYS, B / T_DAYS, 0.0))
K_FAT = 60_000.0
A0 = 1.9 * (B / K_FAT) / (1 - math.exp(-B / K_FAT))       # so that the campaign-average long-run ROAS is 1.9
A_fatigue = A0 * np.exp(-Cpath / K_FAT)
OUT["fatigue"] = dict(A0=A0, K=K_FAT, stop=stopping_analysis(A_fatigue, "fatigue"))
# non-monotone: fatigue plus a late effectiveness bump (e.g. seasonal demand peak in months 9-11)
bump = 1 + 2.0 * np.exp(-((days - 330) / 20.0) ** 2)
A_nonmono = A_fatigue * bump
OUT["nonmonotone"] = dict(stop=stopping_analysis(A_nonmono, "nonmonotone"))
OUT["linear_constant"] = dict(stop=stopping_analysis(np.full(H_DAYS, A_LR), "constant"))

# ----------------------------------------------------------------------------
# 7. Kernel recovery: NLS, metrics, consistency (H3), stress scenarios
# ----------------------------------------------------------------------------
def gamma_w(theta, L=L_MAX):
    a, b = theta
    w = stats.gamma.pdf(np.arange(L + 1) + 0.5, a=a, scale=b)
    return w / w.sum()

def simulate_series(T, true_w, A=A_LR, rng=rng, noise=0.30, scenario="base"):
    """Daily spend (lognormal), contribution flow = A * conv(spend, w) * extras + noise."""
    ell = len(true_w)
    # Spend: monthly budget levels (lognormal, sigma 0.5) held for 30 days, plus 10 % daily noise.
    # Budget variation at the time scale of the kernel is what identifies its shape (Assumption 3).
    sig_m = 0.05 if scenario == "lowvar" else 0.5
    levels = rng.lognormal(mean=math.log(B / T_DAYS), sigma=sig_m, size=T // 30 + 1)
    X = np.repeat(levels, 30)[:T] * rng.lognormal(0, 0.1, size=T)
    eps = rng.normal(0, 1, size=T)
    if scenario == "endog":       # persistent demand shock u_t drives both spend and revenue
        u = np.zeros(T); e = rng.normal(0, 1, size=T)
        for t in range(1, T): u[t] = 0.99 * u[t - 1] + math.sqrt(1 - 0.99 ** 2) * e[t]
        X = X * np.exp(0.6 * u)
        eps = 0.8 * u + 0.6 * eps
    conv = np.convolve(X, true_w)[:T]
    y = A * conv
    if scenario == "break":       # effectiveness drops by 40% in the second half
        y = y * np.where(np.arange(T) < T // 2, 1.0, 0.6)
    if scenario == "season":      # unmodelled multiplicative seasonality (annual)
        y = y * (1 + 0.3 * np.sin(2 * np.pi * np.arange(T) / 365))
    if scenario == "margin":      # unmodelled margin change (m: 1.0 -> 0.8 after day T/2)
        y = y * np.where(np.arange(T) < T // 2, 1.0, 0.8)
    if scenario == "omitted":     # omitted second channel with exponential kernel
        lv2 = rng.lognormal(mean=math.log(0.5 * B / T_DAYS), sigma=0.5, size=T // 30 + 1)
        X2 = np.repeat(lv2, 30)[:T] * rng.lognormal(0, 0.1, size=T)
        y = y + 1.2 * np.convolve(X2, W["exp"])[:T]
    sd = noise * y.mean()
    y = y + sd * eps
    return X, y

BOUNDS_WIDE = ([0, 0.2, 1.0], [50, 30, 400])
BOUNDS_NARROW = ([0, 0.5, 5.0], [10, 10, 150])
def fit_gamma(X, y, start=(1.5, 60.0), bounds=BOUNDS_WIDE):
    T = len(y)
    def model(_, A, a, b):
        return A * np.convolve(X, gamma_w((a, b)))[:T]
    p0 = (y.sum() / X.sum(), *start)
    popt, pcov = optimize.curve_fit(model, None, y, p0=p0, bounds=bounds, maxfev=20000)
    return popt, pcov

def sse_gamma(X, y, p):
    T = len(y)
    return float(((y - p[0] * np.convolve(X, gamma_w(p[1:]))[:T]) ** 2).sum())

def metrics(theta_hat, true_w, A_hat=None, true_A=A_LR):
    w_hat = gamma_w(theta_hat)
    ise = float(((w_hat - true_w) ** 2).sum())
    rmse = math.sqrt(ise / len(true_w))
    ks = kernel_stats(true_w); kh = kernel_stats(w_hat)
    Le = ks["L_eps"]
    return dict(ise=ise, rmse=rmse, mean_lag_err=kh["mean"] - ks["mean"],
                tail_mass_err=float(w_hat[Le + 1:].sum() - true_w[Le + 1:].sum()),
                A_bias=(None if A_hat is None else float(A_hat - true_A)))

true_w = W["gamma"]; theta0 = (3.0, 25.0)

# 7a. Single illustrative fit (T = 720)
X, y = simulate_series(720, true_w)
popt, pcov = fit_gamma(X, y)
OUT["recovery_T720"] = dict(A_hat=float(popt[0]), alpha_hat=float(popt[1]), beta_hat=float(popt[2]),
                            **metrics(popt[1:], true_w, popt[0]))

# 7b. Consistency across T with coverage of 95% Wald intervals (H3)
REPS = 200
Ts = [90, 180, 360, 720, 1440, 2880]
cons = {}
for T in Ts:
    rows = []; cover_a = 0; cover_b = 0; sign_ok = 0
    for r in range(REPS):
        Xr, yr = simulate_series(T, true_w)
        try:
            p, cov = fit_gamma(Xr, yr)
        except Exception:
            continue
        se = np.sqrt(np.diag(cov))
        cover_a += abs(p[1] - theta0[0]) <= 1.96 * se[1]
        cover_b += abs(p[2] - theta0[1]) <= 1.96 * se[2]
        m = metrics(p[1:], true_w, p[0]); rows.append(m)
        # decision metric: sign of total bias at month 12 computed with estimated vs true kernel
        Ph = paths(gamma_w(p[1:])); rh = rois(Ph)
        sign_ok += np.sign(rh["delta"][12 * DAY - 1]) == np.sign(rb["delta"][12 * DAY - 1])
    n = len(rows)
    cons[T] = dict(n=n,
                   rmse_mean=float(np.mean([m["rmse"] for m in rows])),
                   rmse_q15=float(np.percentile([m["rmse"] for m in rows], 15)),
                   rmse_q85=float(np.percentile([m["rmse"] for m in rows], 85)),
                   ise_mean=float(np.mean([m["ise"] for m in rows])),
                   mean_lag_err_mean=float(np.mean([m["mean_lag_err"] for m in rows])),
                   mean_lag_err_sd=float(np.std([m["mean_lag_err"] for m in rows])),
                   tail_mass_err_mean=float(np.mean([m["tail_mass_err"] for m in rows])),
                   A_bias_mean=float(np.mean([m["A_bias"] for m in rows])),
                   coverage_alpha=cover_a / n, coverage_beta=cover_b / n,
                   sign_decision_correct=sign_ok / n)
OUT["consistency"] = cons

# 7c. Stress scenarios at T = 720, 100 replications each
scen = ["base", "wrongfamily", "endog", "break", "season", "margin", "omitted", "lowvar"]
stress = {}
for s in scen:
    rows = []
    for r in range(REPS):
        tw = W["lognorm"] if s == "wrongfamily" else true_w
        Xr, yr = simulate_series(720, tw, scenario=("base" if s == "wrongfamily" else s))
        try:
            p, cov = fit_gamma(Xr, yr)
        except Exception:
            continue
        m = metrics(p[1:], tw, p[0]);
        # pipeline share p at month 12 under estimated vs true kernel (uniform spend, A irrelevant for p)
        Ph = paths(gamma_w(p[1:])); Pt = paths(tw)
        m["p12_err"] = float(rois(Ph)["p"][12 * DAY - 1] - rois(Pt)["p"][12 * DAY - 1])
        rows.append(m)
    stress[s] = dict(n=len(rows),
                     rmse=float(np.mean([m["rmse"] for m in rows])),
                     mean_lag_err=float(np.mean([m["mean_lag_err"] for m in rows])),
                     mean_lag_err_sd=float(np.std([m["mean_lag_err"] for m in rows])),
                     tail_mass_err=float(np.mean([m["tail_mass_err"] for m in rows])),
                     A_bias=float(np.mean([m["A_bias"] for m in rows])),
                     p12_err=float(np.mean([m["p12_err"] for m in rows])))
OUT["stress"] = stress

# 7d. Separability: recovering h_conv and h_rep from their sum with / without a shifter W_t
def two_kernel_series(T, rng, with_shifter):
    levels = rng.lognormal(mean=math.log(B / T_DAYS), sigma=0.5, size=T // 30 + 1)
    X = np.repeat(levels, 30)[:T] * rng.lognormal(0, 0.1, size=T)
    w_conv = gamma_w((3.0, 25.0)); w_rep = gamma_w((6.0, 30.0))   # mean 75 vs 180 days
    conv = np.convolve(X, w_conv)[:T]; rep = np.convolve(X, w_rep)[:T]
    if with_shifter:
        Wt = rng.uniform(0.5, 1.5, size=T)      # observed shifter scaling only the repeat component
    else:
        Wt = np.ones(T)
    y = 1.2 * conv + 0.7 * Wt * rep
    y = y + 0.05 * y.mean() * rng.normal(size=T)
    return X, Wt, y, w_conv, w_rep

def fit_two(X, Wt, y, start):
    T = len(y)
    def model(_, A1, a1, b1, A2, a2, b2):
        return A1 * np.convolve(X, gamma_w((a1, b1)))[:T] + A2 * Wt * np.convolve(X, gamma_w((a2, b2)))[:T]
    popt, _ = optimize.curve_fit(model, None, y, p0=start,
                                 bounds=([0, 0.2, 1, 0, 0.2, 1], [50, 30, 400, 50, 30, 400]), maxfev=40000)
    return popt

sep = {}
for with_shifter in [False, True]:
    errs_conv = []; errs_rep = []; errs_sum = []
    for r in range(100):
        X, Wt, y, wc, wr = two_kernel_series(1440, rng, with_shifter)
        starts = [(1.0, 2.0, 40.0, 1.0, 4.0, 40.0), (1.0, 5.0, 20.0, 1.0, 2.0, 60.0)]
        best = None
        for st in starts:
            try:
                p = fit_two(X, Wt, y, st)
            except Exception:
                continue
            T = len(y)
            resid = y - (p[0] * np.convolve(X, gamma_w(p[1:3]))[:T] + p[3] * Wt * np.convolve(X, gamma_w(p[4:6]))[:T])
            sse = (resid ** 2).sum()
            if best is None or sse < best[0]:
                best = (sse, p)
        if best is None: continue
        p = best[1]
        wc_h = gamma_w(p[1:3]); wr_h = gamma_w(p[4:6])
        # component-level ISE (scaled kernels) and aggregate ISE
        sc = 1.2 * wc; sr = 0.7 * wr
        errs_conv.append(float(((p[0] * wc_h - sc) ** 2).sum() / (sc ** 2).sum()))
        errs_rep.append(float(((p[3] * wr_h - sr) ** 2).sum() / (sr ** 2).sum()))
        errs_sum.append(float((((p[0] * wc_h + p[3] * wr_h) - (sc + sr)) ** 2).sum() / ((sc + sr) ** 2).sum()))
    sep["with_shifter" if with_shifter else "no_shifter"] = dict(
        n=len(errs_sum), rel_ise_conv=float(np.median(errs_conv)), rel_ise_rep=float(np.median(errs_rep)),
        rel_ise_sum=float(np.median(errs_sum)))
OUT["separability"] = sep


# ----------------------------------------------------------------------------
# 7e. Estimation diagnostics: multi-start, convergence, boundary hits, bounds sensitivity, error distribution
# ----------------------------------------------------------------------------
STARTS = [(1.5, 60.0), (6.0, 10.0), (1.0, 150.0)]
diag = dict(n=0, converged=0, boundary_hits=0, start_disagreement=0, best_not_default=0,
            mean_lag_err_q=None, narrow_mean_lag_err_sd=None, narrow_boundary_hits=0)
errs_default = []; errs_best = []; errs_narrow = []; spreads = []
for r in range(REPS):
    Xr, yr = simulate_series(720, true_w)
    sols = []
    for st in STARTS:
        try:
            p, _ = fit_gamma(Xr, yr, start=st); sols.append((sse_gamma(Xr, yr, p), p, st))
        except Exception:
            sols.append((np.inf, None, st))
    ok = [sl for sl in sols if sl[1] is not None]
    diag["n"] += 1
    if len(ok) == len(STARTS): diag["converged"] += 1
    if not ok: continue
    best = min(ok, key=lambda z: z[0]); pb = best[1]
    lo, hi = BOUNDS_WIDE
    if any(abs(pb[i] - lo[i]) < 0.01 * (hi[i] - lo[i]) or abs(pb[i] - hi[i]) < 0.01 * (hi[i] - lo[i]) for i in (1, 2)):
        diag["boundary_hits"] += 1
    lags = [kernel_stats(gamma_w(sl[1][1:]))["mean"] for sl in ok]
    spreads.append(max(lags) - min(lags))
    if max(lags) - min(lags) > 1.0: diag["start_disagreement"] += 1
    if best[2] != STARTS[0]: diag["best_not_default"] += 1
    errs_best.append(kernel_stats(gamma_w(pb[1:]))["mean"] - 74.5)
    d = [sl for sl in ok if sl[2] == STARTS[0]]
    if d: errs_default.append(kernel_stats(gamma_w(d[0][1][1:]))["mean"] - 74.5)
    try:
        pn, _ = fit_gamma(Xr, yr, bounds=BOUNDS_NARROW)
        errs_narrow.append(kernel_stats(gamma_w(pn[1:]))["mean"] - 74.5)
        lo, hi = BOUNDS_NARROW
        if any(abs(pn[i] - lo[i]) < 0.01 * (hi[i] - lo[i]) or abs(pn[i] - hi[i]) < 0.01 * (hi[i] - lo[i]) for i in (1, 2)):
            diag["narrow_boundary_hits"] += 1
    except Exception:
        pass
qs = [5, 25, 50, 75, 95]
diag["mean_lag_err_q"] = {str(q): float(np.percentile(errs_best, q)) for q in qs}
diag["mean_lag_err_q_default"] = {str(q): float(np.percentile(errs_default, q)) for q in qs}
diag["spread_median"] = float(np.median(spreads)); diag["spread_q95"] = float(np.percentile(spreads, 95))
diag["narrow_mean_lag_err_sd"] = float(np.std(errs_narrow)); diag["wide_mean_lag_err_sd"] = float(np.std(errs_best))
diag["narrow_mean_lag_err_mean"] = float(np.mean(errs_narrow)); diag["wide_mean_lag_err_mean"] = float(np.mean(errs_best))
OUT["diagnostics"] = diag

# ----------------------------------------------------------------------------
# 7f. Stopping rule: grid of late-peak amplitude and timing, OSLA loss relative to Bellman
# ----------------------------------------------------------------------------
grid_res = []
for amp in [0.5, 1.0, 1.5, 2.0, 2.5]:
    for centre in [270, 300, 330, 360]:
        bmp = 1 + amp * np.exp(-((days - centre) / 20.0) ** 2)
        st = stopping_analysis(A_fatigue * bmp, f"amp{amp}_c{centre}")
        loss = (st["Pi_bellman"] - st["Pi_osla"]) / st["Pi_bellman"]
        grid_res.append(dict(amp=amp, centre=centre, osla_month=st["osla_stop_day"] // DAY,
                             bellman_month=st["bellman_stop_day"] // DAY, loss_pct=100 * loss, monotone=st["monotone"]))
OUT["stopping_grid"] = grid_res

# ----------------------------------------------------------------------------
# 8. Figures
# ----------------------------------------------------------------------------
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
RED, BLUE, GREEN, GREY = "#b22222", "#1f4e9c", "#2e8b57", "#666666"

# Fig 1: kernels
fig, ax = plt.subplots(figsize=(10, 3.2))
for f, col, lab in [("exp", RED, "Exponential (B2C)"), ("gamma", BLUE, "Gamma (SaaS/B2B)"), ("lognorm", GREEN, "Lognormal (healthcare)")]:
    ax.plot(np.arange(301), W[f][:301], color=col, lw=2, label=lab)
    ax.axvline(OUT["kernels"][f]["mean"], color=col, ls="--", lw=1)
ax.set_xlabel("lag ℓ, days"); ax.set_ylabel("weight h_ℓ"); ax.legend(); ax.set_xlim(0, 300)
ax.set_title("Three lag-kernel families (daily discretisation)")
fig.tight_layout(); fig.savefig("fig1_kernels.png", dpi=300); plt.close(fig)

# Fig 2: ROI dynamics
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.fill_between(months, 100 * mc["naive"][0], 100 * mc["naive"][2], color=RED, alpha=0.12)
ax.fill_between(months, 100 * mc["contract"][0], 100 * mc["contract"][2], color=BLUE, alpha=0.12)
ax.plot(months, 100 * mc["mean_naive"], color=RED, lw=3, label="Observed (naive) ROI")
ax.plot(months, 100 * mc["mean_contract"], color=BLUE, lw=3, label="Commitment-adjusted ROI (no expected future contribution)")
ax.plot(months, 100 * mc["mean_rtf"], color=GREEN, lw=3, ls="-.", label="Stop-now ROI (RTF: commitments + expected future contribution)")
ax.axvline(12, color=GREY, ls="--")
ax.set_xlabel("campaign month"); ax.set_ylabel("ROI, %"); ax.set_xticks(months[1::2])
ax.set_title(f"Dynamics of the three ROI measures (N = {N_SIM}, 70 % band)")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout(); fig.savefig("fig2_roi_dynamics.png", dpi=300); plt.close(fig)

# Fig 3: contract-bias map
roas_grid = np.linspace(0.5, 4.0, 200); r_grid = np.linspace(0.0, 1.0, 200)
RR, rr = np.meshgrid(roas_grid, r_grid)
Zc = 100 * RR * rr / (1 + rr)
fig, ax = plt.subplots(figsize=(10, 6.5))
cs = ax.contourf(RR, rr, Zc, levels=20, cmap="Reds"); cb = fig.colorbar(cs, ax=ax); cb.set_label("commitment component Δ_C, pp")
cl = ax.contour(RR, rr, Zc, levels=[10, 20, 38, 60, 100, 150], colors="k", linewidths=0.8); ax.clabel(cl, fmt="%.0f")
ax.plot(1.9, 0.25, marker="*", color="k", ms=16); ax.annotate("base case: Δ_C = 38 pp", (1.9, 0.25), (2.3, 0.15), arrowprops=dict(arrowstyle="->"))
ax.set_xlabel("ROAS = R / C"); ax.set_ylabel("commitment ratio r = C_eff / C")
ax.set_title("Commitment component Δ_C = ROAS · r / (1 + r)")
fig.tight_layout(); fig.savefig("fig3_bias_map.png", dpi=300); plt.close(fig)

# Fig 4: recovery + consistency
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
a1.plot(np.arange(301), true_w[:301], color=BLUE, lw=2.5, label="true Γ(3, 25)")
a1.plot(np.arange(301), gamma_w(popt[1:])[:301], color=RED, lw=2, ls="--", label=f"NLS estimate Γ({popt[1]:.2f}, {popt[2]:.1f})")
a1.set_xlabel("lag, days"); a1.set_ylabel("weight"); a1.legend(); a1.set_title("(a) Kernel recovery, T = 720")
Tv = np.array(Ts); a2.plot(Tv, [cons[t]["rmse_mean"] for t in Ts], "o-", color=BLUE, lw=2)
a2.fill_between(Tv, [cons[t]["rmse_q15"] for t in Ts], [cons[t]["rmse_q85"] for t in Ts], color=BLUE, alpha=0.15)
a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xlabel("series length T, days"); a2.set_ylabel("RMSE of normalised kernel")
a2.set_title(f"(b) Error decreases with T ({REPS} replications)")
fig.tight_layout(); fig.savefig("fig4_recovery.png", dpi=300); plt.close(fig)

# Fig 5: decomposition over time (base case, deterministic)
fig, ax = plt.subplots(figsize=(10, 5))
mm = months
ax.plot(mm, 100 * rb["delta_C"][midx], color=RED, lw=2.5, label="commitment component Δ_C (overstatement)")
ax.plot(mm, -100 * rb["delta_L"][midx], color=BLUE, lw=2.5, label="lag component −Δ_L (understatement)")
ax.plot(mm, 100 * rb["delta"][midx], color="k", lw=3, label="total gap Δ = Δ_C − Δ_L")
ax.axhline(0, color=GREY, lw=1); ax.axvline(12, color=GREY, ls="--")
ax.set_xlabel("campaign month"); ax.set_ylabel("gap of observed ROI, pp"); ax.set_xticks(mm[1::2])
ax.set_title("Decomposition of the observed-ROI gap over time (base case, no noise)")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout(); fig.savefig("fig5_decomposition.png", dpi=300); plt.close(fig)

# Fig 6: stopping: Pi(t) for fatigue (monotone) and non-monotone profiles
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, key, title in [(a1, "fatigue", "(a) Monotone case: audience fatigue"), (a2, "nonmonotone", "(b) Non-monotone case: late effectiveness peak")]:
    st = OUT[key]["stop"]
    grid = np.arange(DAY, T_DAYS + 1, DAY)
    P = paths(W["gamma"], A_profile=(A_fatigue if key == "fatigue" else A_nonmono))
    Pi = (P["R"] + P["Rp"] - P["C"] - P["C_eff"])[grid - 1]
    ax.plot(grid / DAY, Pi / 1000, "o-", color=BLUE, lw=2, label="Π(t): expected net contribution of the position if stopped at t")
    ax.axvline(st["osla_stop_day"] / DAY, color=RED, ls="--", label=f"OSLA: month {st['osla_stop_day']//DAY}")
    ax.axvline(st["bellman_stop_day"] / DAY, color=GREEN, ls=":", lw=2.5, label=f"Bellman: month {st['bellman_stop_day']//DAY}")
    ax.set_xlabel("stopping month"); ax.set_ylabel("Π, thousand m.u."); ax.set_title(title, fontsize=10); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("fig6_stopping.png", dpi=300); plt.close(fig)

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: OUT[k] for k in ["kernels", "base_case", "mc_month18", "mc_month12", "tail_by_family", "fatigue", "nonmonotone", "linear_constant", "recovery_T720", "consistency", "stress", "separability"]}, ensure_ascii=False, indent=1, default=float))
