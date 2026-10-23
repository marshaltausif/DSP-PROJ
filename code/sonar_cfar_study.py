#!/usr/bin/env python3
"""
sonar_cfar_study.py
===================
Monte-Carlo study of CFAR detectors for ACTIVE SONAR in heavy-tailed
(K-distributed) reverberation, including a proposed detector:

    ACK-CFAR  =  Automatic-Censoring, log-moment K-distribution CFAR
                 (ACK: analytic threshold; ACK-cal: estimator-aware calibrated threshold)

Detectors compared
------------------
  CA, GO, SO, OS      classical (thresholds designed for Gaussian background)
  VI                  Variability-Index CFAR (Smith & Varshney style)
  K-LM                K-distribution CFAR, log-moment shape estimate, NO censoring
                      (a simple stand-in for K-CFAR-type detectors)

  ORACLE              K threshold with the true shape/scale (upper bound)

Scenarios (per K shape parameter nu; nu = inf means Gaussian/Rayleigh)
---------------------------------------------------------------------
  homog       homogeneous reverberation
  multi       3 interfering targets (15 dB) inside the reference window
  edge_high   reverberation edge, CUT on the HIGH side (worst case for Pfa)
  edge_low    reverberation edge, CUT on the LOW  side (worst case for Pd)

Outputs (in --out directory)
----------------------------
  results.csv          Pfa and Pd-vs-SCR for every detector/scenario/nu
  summary.csv          Pfa and SCR needed for Pd = 0.9
  summary_table.tex    LaTeX (booktabs) table ready for the paper
  pd_nu<...>.png       Pd vs SCR figures
  pfa_bars.png         measured Pfa figure

Usage
-----
  python sonar_cfar_study.py --quick          # ~1 min sanity run
  python sonar_cfar_study.py                  # full run (some minutes)
  python sonar_cfar_study.py --pfa 1e-4 --pfa_samples 4000000 --pd_samples 100000

Requires: numpy, scipy, matplotlib
"""
import argparse
import csv
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.special import gammaln, kve, polygamma

# --------------------------------------------------------------------------
# Window geometry:  [ lead(HALF) | guard | CUT | guard | lag(HALF) ]
# --------------------------------------------------------------------------
HALF, GUARD = 16, 2


def set_geometry(half, guard):
    global HALF, GUARD, N_REF, W, CUT, LEAD, LAG, REF_IDX
    HALF, GUARD = half, guard
    N_REF = 2 * HALF
    W = 2 * (HALF + GUARD) + 1
    CUT = HALF + GUARD
    LEAD = slice(0, HALF)
    LAG = slice(W - HALF, W)
    REF_IDX = np.r_[0:HALF, W - HALF:W]


set_geometry(HALF, GUARD)


# --------------------------------------------------------------------------
# K-distribution (intensity) tables, unit mean.
#   I = tau * E,  tau ~ Gamma(nu, 1/nu),  E ~ Exp(1)
#   P(I > t) = 2/Gamma(nu) * (nu t)^(nu/2) * K_nu(2 sqrt(nu t))
#   Var[ln I] = trigamma(nu) + pi^2/6 ;  E[ln I] = psi(nu) - ln(nu) + psi(1)
# --------------------------------------------------------------------------
def log_tail_K(t, nu):
    z = 2.0 * np.sqrt(nu * t)
    return np.log(2.0) - gammaln(nu) + 0.5 * nu * np.log(nu * t) + np.log(kve(nu, z)) - z


class KTables:
    def __init__(self, pfa, p_cens, nu_min=0.2, nu_max=100.0, n=90):
        self.nu = np.geomspace(nu_min, nu_max, n)
        self.lnu = np.log(self.nu)
        self.kappa = np.array([self._solve(v, pfa) for v in self.nu])      # detection multiplier
        self.cens = np.array([self._solve(v, p_cens) for v in self.nu])    # censoring multiplier
        self.r_trunc = np.array([self._trunc_mean(v, c, p_cens)
                                 for v, c in zip(self.nu, self.cens)])     # E[I | I<cens]
        self.trig = polygamma(1, self.nu)                                  # decreasing in nu

    @staticmethod
    def _solve(nu, p):
        return brentq(lambda t: log_tail_K(t, nu) - np.log(p), 0.3, 1e7, xtol=1e-9)

    @staticmethod
    def _trunc_mean(nu, t, p):
        u = np.linspace(np.log(t), np.log(t) + 20.0, 4000)
        s = np.exp(u)
        y = np.exp(log_tail_K(s, nu)) * s
        tail_int = np.sum((y[1:] + y[:-1]) * np.diff(u)) / 2.0
        tail_mass = t * np.exp(log_tail_K(t, nu)) + tail_int               # E[I ; I>t]
        return (1.0 - tail_mass) / (1.0 - p)

    def interp(self, nu_hat, arr):
        return np.interp(np.log(nu_hat), self.lnu, arr)


# --------------------------------------------------------------------------
# Data generator
# --------------------------------------------------------------------------
def gen_windows(rng, M, nu, sc, scr_db=None):
    """Return X (M, W) intensities and ctx with the true local mean at CUT."""
    tau = np.ones((M, W)) if np.isinf(nu) else rng.gamma(nu, 1.0 / nu, (M, W))
    mu = np.ones((M, W))
    if sc.get("edge"):
        r = 10 ** (sc["ratio_db"] / 10.0)
        idx = CUT if sc["edge"] == "high" else CUT + 1
        mu[:, idx:] = r
    add = np.zeros((M, W))
    n_int = sc.get("n_int", 0)
    if n_int:
        pick = rng.random((M, N_REF)).argsort(1).argsort(1) < n_int
        add[:, REF_IDX] = pick * (10 ** (sc["inr_db"] / 10.0)) * mu[:, REF_IDX]
    if scr_db is not None:
        add[:, CUT] += 10 ** (scr_db / 10.0) * mu[:, CUT]
    X = (tau * mu + add) * rng.exponential(1.0, (M, W))   # Swerling-I style target
    return X, {"nu": nu, "mu_cut": mu[:, CUT]}


# --------------------------------------------------------------------------
# PROPOSED ESTIMATOR: log-moment K fit with model-based automatic censoring
# --------------------------------------------------------------------------
def estimate_K(R, tab, iters=2, censor=True):
    """R (M, n) reference intensities -> (nu_hat, mu_hat) robust to interferers.

    1. ln-moment estimate of shape nu (variance of ln I) on the retained cells.
    2. Censor cells above  mean * c(nu)  (c = K-tail quantile at p_cens), where
       c(nu) grows automatically for spiky reverberation, so genuine clutter
       spikes are NOT censored but interfering targets are.
    3. Re-estimate; correct the mean for truncation with r(nu).
    """
    L = np.log(np.maximum(R, 1e-12))
    mask = np.ones(R.shape, dtype=bool)
    r_used = np.ones(R.shape[0])
    for it in range(iters + 1):
        cnt = mask.sum(1)
        m = (L * mask).sum(1) / cnt
        v = (((L - m[:, None]) ** 2) * mask).sum(1) / np.maximum(cnt - 1, 1)
        psi1 = v - np.pi ** 2 / 6.0
        nu = np.exp(np.interp(psi1, tab.trig[::-1], tab.lnu[::-1]))       # invert trigamma
        mu = (R * mask).sum(1) / cnt / r_used
        if (not censor) or it == iters:
            break
        c = tab.interp(nu, tab.cens)
        r_used = tab.interp(nu, tab.r_trunc)
        mask = R <= (mu * c)[:, None]
        mu = mu  # (mean re-computed next loop with truncation correction)
    return nu, mu


# --------------------------------------------------------------------------
# Detectors.  Every detector returns the STATISTIC  s = I_cut / threshold_unit
# (decision: s > 1).  Returning the ratio lets us also evaluate Pd at a
# MATCHED measured Pfa (ROC-style), which is the fair way to compare detectors
# that do not hold the design Pfa in non-Gaussian reverberation.
# --------------------------------------------------------------------------
def train_ack_table(pfa, tab, rng, n_train, n_bins):
    """Estimator-aware threshold: learn quantile of I_cut/mu_hat as a function of nu_hat.

    Training data: homogeneous K-reverberation with nu ~ log-uniform[0.4, 60]
    (80%) plus pure Rayleigh/Gaussian (20%).  Equal-count bins of nu_hat; the
    (1-pfa) quantile of the ratio in each bin becomes the threshold multiplier.
    """
    nu_hat, ratio, done = [], [], 0
    while done < n_train:
        m = min(100_000, n_train - done)
        nu_row = np.exp(rng.uniform(np.log(0.4), np.log(60.0), m))
        gauss = rng.random(m) < 0.2
        tau = rng.gamma(nu_row[:, None], 1.0 / nu_row[:, None], (m, W))
        tau[gauss] = 1.0
        X = tau * rng.exponential(1.0, (m, W))
        nh, mh = estimate_K(X[:, REF_IDX], tab)
        nu_hat.append(nh); ratio.append(X[:, CUT] / mh)
        done += m
    nu_hat, ratio = np.concatenate(nu_hat), np.concatenate(ratio)
    order = np.argsort(nu_hat)
    nu_hat, ratio = nu_hat[order], ratio[order]
    edges = np.linspace(0, len(nu_hat), n_bins + 1).astype(int)
    xs, ys = [], []
    for a_, b_ in zip(edges[:-1], edges[1:]):
        xs.append(np.log(np.median(nu_hat[a_:b_])))
        ys.append(np.log(np.quantile(ratio[a_:b_], 1 - pfa)))
    xs, ys = np.array(xs), np.array(ys)
    # smooth with a low-order polynomial (monotone-ish, removes quantile noise)
    coef = np.polyfit(xs, ys, 3)
    grid = np.linspace(xs.min(), xs.max(), 200)
    return grid, np.exp(np.polyval(coef, grid))


def build_detectors(pfa, tab, rng, cal_n, train_n, n_bins):
    N = N_REF

    def calib(est_fn):
        ratios, done = [], 0
        while done < cal_n:
            m = min(500_000, cal_n - done)
            X = rng.exponential(1.0, (m, W))
            ratios.append(X[:, CUT] / est_fn(X))
            done += m
        return np.quantile(np.concatenate(ratios), 1 - pfa)

    ml = lambda X: X[:, LEAD].mean(1)
    mg = lambda X: X[:, LAG].mean(1)
    k_os = int(round(0.75 * N))
    est_go = lambda X: np.maximum(ml(X), mg(X))
    est_so = lambda X: np.minimum(ml(X), mg(X))
    est_os = lambda X: np.partition(X[:, REF_IDX], k_os - 1, axis=1)[:, k_os - 1]

    a_ca = N * (pfa ** (-1.0 / N) - 1.0)
    a_ca_h = HALF * (pfa ** (-1.0 / HALF) - 1.0)
    a_go, a_so, a_os = calib(est_go), calib(est_so), calib(est_os)
    cal_x, cal_y = train_ack_table(pfa, tab, rng, train_n, n_bins)

    K_VI, K_MR = 4.76, 1.806   # Smith & Varshney constants for 16-cell half windows (verify)

    def det_vi(X, ctx):
        Ld, Gd = X[:, LEAD], X[:, LAG]
        mL, mG = Ld.mean(1), Gd.mean(1)
        viL = 1 + Ld.var(1, ddof=1) / mL ** 2
        viG = 1 + Gd.var(1, ddof=1) / mG ** 2
        homL, homG = viL <= K_VI, viG <= K_VI
        mr = mL / mG
        mr_ok = (mr >= 1 / K_MR) & (mr <= K_MR)
        thr = np.where(homL & homG & mr_ok, a_ca * (mL + mG) / 2,
              np.where(homL & homG, a_go * np.maximum(mL, mG),
              np.where(homL, a_ca_h * mL,
              np.where(homG, a_ca_h * mG, a_so * np.minimum(mL, mG)))))
        return X[:, CUT] / thr

    def det_klm(X, ctx):
        nu, mu = estimate_K(X[:, REF_IDX], tab, censor=False)
        return X[:, CUT] / (mu * tab.interp(nu, tab.kappa))

    def det_ack(X, ctx):                       # analytic plug-in threshold
        nu, mu = estimate_K(X[:, REF_IDX], tab)
        return X[:, CUT] / (mu * tab.interp(nu, tab.kappa))

    def det_ack_cal(X, ctx):                   # estimator-aware calibrated threshold
        nu, mu = estimate_K(X[:, REF_IDX], tab)
        return X[:, CUT] / (mu * np.interp(np.log(nu), cal_x, cal_y))

    def det_oracle(X, ctx):
        nu = ctx["nu"]
        kap = -np.log(pfa) if np.isinf(nu) else np.interp(np.log(nu), tab.lnu, tab.kappa)
        return X[:, CUT] / (ctx["mu_cut"] * kap)

    return {
        "CA": lambda X, c: X[:, CUT] / (a_ca * X[:, REF_IDX].mean(1)),
        "GO": lambda X, c: X[:, CUT] / (a_go * est_go(X)),
        "SO": lambda X, c: X[:, CUT] / (a_so * est_so(X)),
        "OS": lambda X, c: X[:, CUT] / (a_os * est_os(X)),
        "VI": det_vi,
        "K-LM": det_klm,
        "ACK (proposed)": det_ack,
        "ACK-cal (proposed)": det_ack_cal,
        "ORACLE": det_oracle,
    }


# --------------------------------------------------------------------------
# Experiment driver
# --------------------------------------------------------------------------
SCENARIOS = {
    "homog":     {},
    "multi":     {"n_int": 3, "inr_db": 15.0},
    "edge_high": {"edge": "high", "ratio_db": 10.0},
    "edge_low":  {"edge": "low", "ratio_db": 10.0},
}
STYLE = {
    "CA": ("tab:blue", "o", "-"), "GO": ("tab:cyan", "s", "-"), "SO": ("tab:olive", "^", "-"),
    "OS": ("tab:orange", "v", "-"), "VI": ("tab:purple", "D", "-"), "K-LM": ("tab:brown", "x", "--"),
    "ACK (proposed)": ("salmon", "*", "-"), "ACK-cal (proposed)": ("crimson", "P", "-"),
    "ORACLE": ("k", None, ":"),
}


def collect(det, rng, M, nu, sc, scr_db, chunk=100_000):
    out, done = [], 0
    while done < M:
        m = min(chunk, M - done)
        X, ctx = gen_windows(rng, m, nu, sc, scr_db)
        out.append(det(X, ctx))
        done += m
    return np.concatenate(out)


def scr_at(pd_curve, scr, level=0.9):
    pd_curve = np.maximum.accumulate(np.asarray(pd_curve))
    if pd_curve.max() < level or pd_curve.min() > level:
        return np.nan
    return float(np.interp(level, pd_curve, scr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pfa", type=float, default=1e-3)
    ap.add_argument("--pfa_samples", type=int, default=1_000_000, help="H0 windows per detector/scenario")
    ap.add_argument("--pd_samples", type=int, default=40_000, help="H1 windows per SCR point")
    ap.add_argument("--cal_samples", type=int, default=3_000_000)
    ap.add_argument("--train_samples", type=int, default=3_000_000, help="ACK-cal training windows")
    ap.add_argument("--nus", type=float, nargs="+", default=[1.0, 3.0, np.inf])
    ap.add_argument("--half", type=int, default=16)
    ap.add_argument("--guard", type=int, default=2)
    ap.add_argument("--p_cens", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results_sonar_cfar")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.pfa_samples, a.pd_samples, a.cal_samples, a.train_samples = 300_000, 8_000, 800_000, 1_000_000
        a.nus = [1.0, np.inf]
    n_bins = int(np.clip(a.train_samples * a.pfa / 150, 8, 30))   # ~150 exceedances per bin
    set_geometry(a.half, a.guard)
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    scr_grid = np.arange(0, 41, 4.0)

    t0 = time.time()
    print(f"Building K tables, calibrating baselines, training ACK-cal (Pfa={a.pfa:g}) ...")
    tab = KTables(a.pfa, a.p_cens)
    dets = build_detectors(a.pfa, tab, rng, a.cal_samples, a.train_samples, n_bins)
    print(f"  done in {time.time() - t0:.1f}s")

    rows_pd, rows_sum = [], []
    for nu in a.nus:
        nu_lbl = "inf" if np.isinf(nu) else f"{nu:g}"
        for scn, sc in SCENARIOS.items():
            print(f"[nu={nu_lbl}] scenario={scn}")
            for name, det in dets.items():
                s0 = collect(det, rng, a.pfa_samples, nu, sc, None)
                pfa_hat = float((s0 > 1).mean())
                q = np.quantile(s0, 1 - a.pfa)                         # matched-Pfa threshold
                pd_d, pd_m = [], []
                for s in scr_grid:
                    s1 = collect(det, rng, a.pd_samples, nu, sc, s)
                    pd_d.append(float((s1 > 1).mean())); pd_m.append(float((s1 > q).mean()))
                    rows_pd.append([nu_lbl, scn, name, s, pd_d[-1], pd_m[-1]])
                d90, m90 = scr_at(pd_d, scr_grid), scr_at(pd_m, scr_grid)
                rows_sum.append([nu_lbl, scn, name, pfa_hat, d90, m90])
                print(f"   {name:19s} Pfa={pfa_hat:.2e} ({pfa_hat / a.pfa:5.1f}x)  "
                      f"SCR@Pd0.9: design={d90:5.1f} dB  matched-Pfa={m90:5.1f} dB")

    with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["nu", "scenario", "detector", "SCR_dB", "Pd_design", "Pd_matchedPfa"])
        w.writerows(rows_pd)
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["nu", "scenario", "detector", "Pfa_measured", "SCR90_design_dB", "SCR90_matchedPfa_dB"])
        w.writerows(rows_sum)

    # ---- figures ------------------------------------------------------------
    for col, tag, ylabel in [(5, "matched", f"Pd at matched Pfa={a.pfa:g}"),
                             (4, "design", f"Pd with design threshold (Pfa={a.pfa:g})")]:
        for nu in a.nus:
            nu_lbl = "inf" if np.isinf(nu) else f"{nu:g}"
            fig, axs = plt.subplots(1, 4, figsize=(19, 4.2), sharey=True)
            for ax, scn in zip(axs, SCENARIOS):
                for name in dets:
                    sel = [r for r in rows_pd if r[0] == nu_lbl and r[1] == scn and r[2] == name]
                    c, mk, ls = STYLE[name]
                    ax.plot([r[3] for r in sel], [r[col] for r in sel], color=c, marker=mk, ls=ls,
                            ms=5, lw=1.4, label=name)
                ax.set_title(f"{scn}  (nu={nu_lbl})"); ax.set_xlabel("SCR (dB)"); ax.grid(alpha=.3)
            axs[0].set_ylabel(ylabel); axs[0].legend(fontsize=7)
            fig.tight_layout(); fig.savefig(os.path.join(a.out, f"pd_{tag}_nu{nu_lbl}.png"), dpi=200)
            plt.close(fig)

    fig, axs = plt.subplots(1, len(a.nus), figsize=(5.5 * len(a.nus), 4), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, nu in zip(axs, a.nus):
        nu_lbl = "inf" if np.isinf(nu) else f"{nu:g}"
        for i, scn in enumerate(SCENARIOS):
            for j, name in enumerate(dets):
                v = [r[3] for r in rows_sum if r[0] == nu_lbl and r[1] == scn and r[2] == name][0]
                ax.bar(i + 0.09 * (j - len(dets) / 2), max(v, 1e-7), 0.085, color=STYLE[name][0],
                       label=name if i == 0 else None)
        ax.axhline(a.pfa, color="k", ls="--", lw=1)
        ax.set_yscale("log"); ax.set_xticks(range(len(SCENARIOS))); ax.set_xticklabels(SCENARIOS, fontsize=8)
        ax.set_title(f"measured Pfa, nu={nu_lbl}"); ax.grid(alpha=.3, axis="y")
    axs[0].legend(fontsize=6); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "pfa_bars.png"), dpi=200); plt.close(fig)

    with open(os.path.join(a.out, "summary_table.tex"), "w") as f:
        f.write("\\begin{table*}[t]\\centering\\caption{Measured $P_{fa}$ (design %g) / SCR (dB) for "
                "$P_d=0.9$ at matched $P_{fa}$}\n" % a.pfa)
        f.write("\\begin{tabular}{ll" + "c" * len(dets) + "}\\toprule\n")
        f.write("$\\nu$ & Scenario & " + " & ".join(dets) + " \\\\\\midrule\n")
        for nu in a.nus:
            nu_lbl = "inf" if np.isinf(nu) else f"{nu:g}"
            for scn in SCENARIOS:
                cells = []
                for name in dets:
                    r = [x for x in rows_sum if x[0] == nu_lbl and x[1] == scn and x[2] == name][0]
                    cells.append(f"{r[3]:.1e} / {r[5]:.1f}")
                f.write(f"{nu_lbl} & {scn} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table*}\n")

    print(f"\nAll done in {time.time() - t0:.0f}s -> {a.out}/")


if __name__ == "__main__":
    main()
