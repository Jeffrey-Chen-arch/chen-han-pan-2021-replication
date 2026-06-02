"""Table IV: Fama-MacBeth of next-month excess return and alpha on sentiment beta.

MAIN alpha (SAS-aligned): per-fund FULL-SAMPLE 9-factor loadings; alpha_it = excess_it - sum_k
beta_i,k * factor_k,t  (intercept NOT subtracted). Controls are SAS-aligned: lagged log size,
lagged log age, management fee, incentive fee, HWM, lockup dummy, notice/30, + style dummies.
Estimator = monthly cross-sectional OLS, Newey-West(2) on the coefficient series.
"""
import numpy as np
import pandas as pd

from .regression import fama_macbeth
from .utils import winsorize_series
from .config import load_benchmarks

ALPHA_FACTORS = ["mktrf", "smb", "umd", "ptfsbd", "ptfsfx", "ptfscom", "term_s", "credit_s", "liq_v"]
SAS_CONTROLS = ["lag_log_fund_size", "lag_log_fund_age", "management_fee", "incentive_fee",
                "high_water_mark", "lockup_dummy", "notice_period_months"]
STYLES = ["style_ca", "style_emn", "style_gm", "style_lseh", "style_ms", "style_fof"]
WINSOR = ["lag_log_fund_size", "lag_log_fund_age", "management_fee", "incentive_fee", "notice_period_months"]


def fullsample_byfund_alpha(panel: pd.DataFrame, factors: pd.DataFrame, min_obs: int = 30) -> pd.DataFrame:
    d = panel.merge(factors[["month"] + ALPHA_FACTORS], on="month", how="left").dropna(subset=["excess_return"] + ALPHA_FACTORS)
    out = []
    for fid, g in d.groupby("fund_id", sort=False):
        if len(g) < min_obs:
            continue
        F = g[ALPHA_FACTORS].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(g)), F])
        y = g["excess_return"].to_numpy(dtype=float)
        if np.linalg.matrix_rank(X) < X.shape[1]:
            continue
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        alpha = y - F @ beta[1:]          # intercept beta[0] intentionally NOT subtracted
        t = g[["fund_id", "month"]].copy()
        t["alpha_main"] = alpha
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["fund_id", "month", "alpha_main"])


def run_table4(cfg: dict, panel: pd.DataFrame, betas: pd.DataFrame, factors: pd.DataFrame,
               alpha_construction: str = "sas_full", winsor_bounds: tuple | None = None,
               controls_set: str = "sas_lagged"):
    """alpha_construction: 'sas_full' (T0: full-sample by-fund loadings) or
    'paper_rolling' (T1: rolling 36-month loadings per paper text).
    winsor_bounds: optional (lower, upper) override for control winsorization (T5 = (0.01, 0.99);
                   T6 = (0.0, 1.0) = no winsorization).
    controls_set : 'sas_lagged' (lagged size/age + fees/HWM/lockup/notice + 6 styles),
                   'contemporaneous' (non-lagged size/age, else same), or
                   'full_style' (sas_lagged + one-hot of ALL primarycategory levels present)."""
    lags = int(cfg["regressions"]["newey_west_lags"])
    if alpha_construction == "paper_rolling":
        from .rolling import rolling_alpha_paper_text
        alpha = rolling_alpha_paper_text(cfg, panel, factors).rename(columns={"alpha_rolling": "alpha_main"})
    else:
        alpha = fullsample_byfund_alpha(panel, factors)

    p = panel.sort_values(["fund_id", "month"]).copy()
    p["lag_log_fund_size"] = p.groupby("fund_id")["log_fund_size"].shift(1)
    p["lag_log_fund_age"] = p.groupby("fund_id")["log_fund_age"].shift(1)

    b = betas.rename(columns={"return_month": "month"})[["fund_id", "month", "sentiment_beta"]]
    reg = (b.merge(p, on=["fund_id", "month"], how="inner")
             .merge(alpha, on=["fund_id", "month"], how="left"))
    wlo = winsor_bounds[0] if winsor_bounds else cfg["sample"]["winsorize_controls_lower"]
    whi = winsor_bounds[1] if winsor_bounds else cfg["sample"]["winsorize_controls_upper"]
    for c in WINSOR:
        if c in reg:
            reg[c] = winsorize_series(reg[c], wlo, whi)

    # Build the control list per controls_set.
    if controls_set == "contemporaneous":
        size_age = ["log_fund_size", "log_fund_age"]
        other = ["management_fee", "incentive_fee", "high_water_mark", "lockup_dummy", "notice_period_months"]
        ctrls = size_age + other
        style_cols = STYLES
    elif controls_set == "full_style":
        ctrls = SAS_CONTROLS
        cat = reg["primarycategory"].astype(str).str.strip().str.lower() if "primarycategory" in reg else None
        style_cols = list(STYLES)
        if cat is not None:
            dummies = pd.get_dummies(cat, prefix="catdum").astype(float)
            # drop one to avoid collinearity; merge in
            if dummies.shape[1] > 1:
                dummies = dummies.iloc[:, 1:]
            reg = pd.concat([reg, dummies], axis=1)
            style_cols = [c for c in dummies.columns]
    else:  # sas_lagged (default)
        ctrls = SAS_CONTROLS
        style_cols = STYLES

    specs = {
        "excess_univariate": ("excess_return", ["sentiment_beta"]),
        "excess_multivariate": ("excess_return", ["sentiment_beta"] + ctrls + style_cols),
        "alpha_univariate": ("alpha_main", ["sentiment_beta"]),
        "alpha_multivariate": ("alpha_main", ["sentiment_beta"] + ctrls + style_cols),
    }
    results, rows = {}, []
    for name, (dep, regs) in specs.items():
        sub = reg.dropna(subset=[dep])
        res = fama_macbeth(sub, dep, regs, date_col="month", lags=lags)
        results[name] = res
        if res is None:
            rows.append({"model": name, "sentiment_beta_coef": np.nan, "sentiment_beta_t": np.nan,
                         "adj_r2": np.nan, "n_months": 0, "avg_monthly_n": np.nan})
            continue
        avg_n = int(round(res["_avg_nobs"]))  # actual avg regression N (after dropping missing regressors)
        rows.append({"model": name,
                     "sentiment_beta_coef": round(res["sentiment_beta"]["coef"], 3),
                     "sentiment_beta_t": round(res["sentiment_beta"]["t"], 2),
                     "adj_r2": round(res["_adj_r2_avg"], 3), "n_months": res["_n_months"],
                     "avg_monthly_n": avg_n})
    tbl = pd.DataFrame(rows)

    # Full coefficient tables for the multivariate models (all controls + style dummies).
    full_rows = []
    for name in ["excess_multivariate", "alpha_multivariate"]:
        res = results.get(name)
        if not res:
            continue
        for rn, v in res.items():
            if rn.startswith("_"):
                continue
            full_rows.append({"model": name, "regressor": rn,
                              "coef": round(v["coef"], 3), "t": round(v["t"], 2)})
    full = pd.DataFrame(full_rows)

    miss = pd.DataFrame({"control": SAS_CONTROLS + STYLES,
                         "pct_missing_in_reg_frame": [round(float(reg[c].isna().mean()) * 100, 1) if c in reg else None
                                                      for c in SAS_CONTROLS + STYLES]})

    bench = load_benchmarks()["table4"]["models"]
    brows = [{"model": k, "paper_coef": v["coefficients"]["sentiment_beta"]["coeff"],
              "paper_t": v["coefficients"]["sentiment_beta"]["t"], "paper_adj_r2": v["adjusted_r2"]}
             for k, v in bench.items()]
    comp = tbl.merge(pd.DataFrame(brows), on="model", how="left")
    return tbl, comp, full, miss, reg
