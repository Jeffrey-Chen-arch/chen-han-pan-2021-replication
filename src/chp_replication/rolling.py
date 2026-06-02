"""Rolling estimation with NO look-ahead.

Sentiment beta for return month m: regress fund excess return on factors over the 36 months
[m-36, m-1] (>=30 obs). Coefficient on sentiment_change = sentiment beta. This is the sort
variable for Table III and the regressor for Table IV. First return month = 1997-01 (monthid 37).
"""
import numpy as np
import pandas as pd

BETA_FACTORS = ["sentiment_change", "mktrf", "smb", "umd", "ptfsbd", "ptfsfx",
                "ptfscom", "term_s", "credit_s", "liq_v", "INF", "def"]
ALPHA_FACTORS = ["mktrf", "smb", "umd", "ptfsbd", "ptfsfx", "ptfscom", "term_s", "credit_s", "liq_v"]
FIRST_RETURN_MID = 37  # 1997-01


def _monthid(m: pd.Series) -> pd.Series:
    return (m.dt.year - 1994) * 12 + (m.dt.month - 1) + 1


def _mid_to_month(mid):
    return (pd.Period("1994-01", "M") + (int(mid) - 1)).to_timestamp("M")


def _ols_beta_t(Xw, yk, idx):
    """OLS coefficient vector + t-stat of column `idx` (plain OLS SE, no NW)."""
    beta, *_ = np.linalg.lstsq(Xw, yk, rcond=None)
    resid = yk - Xw @ beta
    dof = len(yk) - Xw.shape[1]
    t = np.nan
    if dof > 0:
        sigma2 = float(resid @ resid) / dof
        try:
            xtx_inv = np.linalg.inv(Xw.T @ Xw)
            se = np.sqrt(sigma2 * np.diag(xtx_inv))
            with np.errstate(divide="ignore", invalid="ignore"):
                tvec = beta / se
            t = float(tvec[idx]) if se[idx] > 0 else np.nan
            return beta, t, tvec
        except np.linalg.LinAlgError:
            return beta, np.nan, None
    return beta, t, None


def rolling_sentiment_betas(cfg: dict, panel: pd.DataFrame, factors: pd.DataFrame,
                            beta_method: str = "single_step") -> pd.DataFrame:
    """Rolling sentiment beta over [m-36, m-1].

    beta_method:
      'single_step'  : one regression on [const, sentiment_change, all controls];
                       sentiment_beta = coeff on sentiment_change (P0/P2/P4/P5).
      'two_step_5pct': Internet-Appendix IA.IV two-step. Step 1 regresses on [const,
                       sentiment_change, ALL controls] and records each CONTROL factor's |t|.
                       Step 2 re-regresses on [const, sentiment_change, controls with |t|>=1.96
                       in step 1] and takes sentiment_beta from step 2. sentiment_change is ALWAYS
                       retained; selection is two-sided |t|>1.96 on the CONTROLS only (plain OLS t).
    """
    win = int(cfg["sample"]["rolling_window_months"])
    minobs = int(cfg["sample"]["min_obs_rolling"])
    fac_cols = [f for f in BETA_FACTORS if f in factors.columns]
    ctrl_idx = [i for i, f in enumerate(BETA_FACTORS) if f != "sentiment_change"]  # control columns (0-based in Xfw)
    df = panel.merge(factors[["month"] + fac_cols], on="month", how="left")
    df["mid"] = _monthid(df["month"])
    win_df = df.dropna(subset=["excess_return"] + BETA_FACTORS)
    grid = df.dropna(subset=["excess_return"])
    wgroups = {fid: g for fid, g in win_df.groupby("fund_id", sort=False)}

    rows = []
    two_step = (beta_method == "two_step_5pct")
    for fid, gd in grid.groupby("fund_id", sort=False):
        wd = wgroups.get(fid)
        if wd is None or len(wd) < minobs:
            continue
        mids_w = wd["mid"].to_numpy()
        Yw = wd["excess_return"].to_numpy(dtype=float)
        Xfw = wd[BETA_FACTORS].to_numpy(dtype=float)  # col 0 = sentiment_change, 1.. = controls
        for k in np.unique(gd["mid"].to_numpy()):
            if k < FIRST_RETURN_MID:
                continue
            mask = (mids_w >= k - win) & (mids_w <= k - 1)
            nobs = int(mask.sum())
            if nobs < minobs:
                continue
            Xf = Xfw[mask]
            yk = Yw[mask]
            # Step 1: full model [const, sentiment(col0), all controls]
            X1 = np.column_stack([np.ones(nobs), Xf])
            if np.linalg.matrix_rank(X1) < X1.shape[1]:
                continue
            beta1, t_sent, tvec1 = _ols_beta_t(X1, yk, 1)  # idx 1 = sentiment_change
            n_sel = len(ctrl_idx)
            if not two_step:
                rows.append((fid, int(k), float(beta1[1]), t_sent, nobs, n_sel))
                continue
            # Step 2: keep sentiment + controls significant at 5% in step 1.
            if tvec1 is None:
                rows.append((fid, int(k), float(beta1[1]), t_sent, nobs, n_sel))
                continue
            # tvec1 columns: 0=const,1=sentiment,2..=controls (in BETA_FACTORS control order)
            keep_ctrl = [j for jj, j in enumerate(range(2, X1.shape[1])) if abs(tvec1[j]) >= 1.96]
            sel_cols = [0, 1] + keep_ctrl  # const + sentiment + significant controls
            X2 = X1[:, sel_cols]
            if np.linalg.matrix_rank(X2) < X2.shape[1]:
                rows.append((fid, int(k), float(beta1[1]), t_sent, nobs, len(keep_ctrl)))
                continue
            beta2, t_sent2, _ = _ols_beta_t(X2, yk, 1)
            rows.append((fid, int(k), float(beta2[1]), t_sent2, nobs, len(keep_ctrl)))

    res = pd.DataFrame(rows, columns=["fund_id", "return_mid", "sentiment_beta", "beta_t", "beta_nobs", "n_selected_controls"])
    res["return_month"] = res["return_mid"].map(_mid_to_month)
    return res


def rolling_alpha_paper_text(cfg: dict, panel: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Paper-text Table IV alpha: for each (fund, return_month m), estimate 9-factor loadings
    over [m-36, m-1] (>=30 obs), then alpha_m = excess_return_m - sum_k beta_i,k * factor_k,m.
    Intercept NOT subtracted. No look-ahead: loadings use only historical data; factors at m are
    contemporaneous realizations (not future)."""
    win = int(cfg["sample"]["rolling_window_months"])
    minobs = int(cfg["sample"]["min_obs_rolling"])
    fac_cols = [f for f in ALPHA_FACTORS if f in factors.columns]
    df = panel.merge(factors[["month"] + fac_cols], on="month", how="left")
    df["mid"] = _monthid(df["month"])
    # Window-eligible rows: excess + alpha factors complete in the historical window.
    win_df = df.dropna(subset=["excess_return"] + ALPHA_FACTORS)
    # Grid: need excess at m AND alpha factors at m (to compute alpha_m).
    grid = df.dropna(subset=["excess_return"] + ALPHA_FACTORS)
    wgroups = {fid: g for fid, g in win_df.groupby("fund_id", sort=False)}

    rows = []
    for fid, gd in grid.groupby("fund_id", sort=False):
        wd = wgroups.get(fid)
        if wd is None or len(wd) < minobs:
            continue
        mids_w = wd["mid"].to_numpy()
        Yw = wd["excess_return"].to_numpy(dtype=float)
        Xfw = wd[ALPHA_FACTORS].to_numpy(dtype=float)
        gm = gd["mid"].to_numpy()
        gE = gd["excess_return"].to_numpy(dtype=float)
        gF = gd[ALPHA_FACTORS].to_numpy(dtype=float)
        for i, k in enumerate(gm):
            if k < FIRST_RETURN_MID:
                continue
            mask = (mids_w >= k - win) & (mids_w <= k - 1)
            nobs = int(mask.sum())
            if nobs < minobs:
                continue
            X = np.column_stack([np.ones(nobs), Xfw[mask]])
            if np.linalg.matrix_rank(X) < X.shape[1]:
                continue
            beta, *_ = np.linalg.lstsq(X, Yw[mask], rcond=None)
            loadings = beta[1:]  # intercept NOT subtracted
            alpha_m = float(gE[i] - loadings @ gF[i])
            rows.append((fid, int(k), alpha_m))

    res = pd.DataFrame(rows, columns=["fund_id", "return_mid", "alpha_rolling"])
    res["month"] = res["return_mid"].map(_mid_to_month)
    return res[["fund_id", "month", "alpha_rolling"]]
