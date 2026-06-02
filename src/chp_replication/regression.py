"""Regression utilities: Newey-West means, time-series alphas, Fama-MacBeth, decile sorts."""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def newey_west_mean(series, lags: int = 2, small_sample_correction: bool = False) -> dict:
    y = pd.to_numeric(pd.Series(series), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(y) <= lags + 5:
        return {"mean": np.nan, "se": np.nan, "t": np.nan, "nobs": int(len(y))}
    X = np.ones((len(y), 1))
    res = sm.OLS(y.values, X).fit(cov_type="HAC",
                                  cov_kwds={"maxlags": lags, "use_correction": small_sample_correction})
    return {"mean": float(res.params[0]), "se": float(res.bse[0]), "t": float(res.tvalues[0]), "nobs": int(res.nobs)}


def time_series_alpha(y, X, lags: int = 2, small_sample_correction: bool = False):
    """Intercept (alpha) of y on X with Newey-West(lags) HAC SEs."""
    data = pd.concat([pd.Series(np.asarray(y), name="y"), pd.DataFrame(X).reset_index(drop=True)], axis=1)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) <= data.shape[1] + lags + 5:
        return None
    Y = data["y"]
    XX = sm.add_constant(data.drop(columns=["y"]), has_constant="add")
    if np.linalg.matrix_rank(XX.values) < XX.shape[1]:
        return None
    return sm.OLS(Y, XX).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": small_sample_correction})


def fama_macbeth(panel: pd.DataFrame, dep: str, regressors: list[str], date_col: str = "return_month",
                 lags: int = 2, small_sample_correction: bool = False) -> dict | None:
    """Monthly cross-sectional OLS, then Newey-West(lags) on the coefficient time series."""
    coef_rows, r2s, nobs_list = [], [], []
    for _, d in panel.groupby(date_col):
        sub = d[[dep] + regressors].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) <= len(regressors) + 2:
            continue
        X = sm.add_constant(sub[regressors], has_constant="add")
        if np.linalg.matrix_rank(X.values) < X.shape[1]:
            continue
        res = sm.OLS(sub[dep], X).fit()
        coef_rows.append(res.params)
        r2s.append(res.rsquared_adj)
        nobs_list.append(int(res.nobs))  # actual regression N (after dropping missing regressors)
    if not coef_rows:
        return None
    cdf = pd.DataFrame(coef_rows)
    out = {"_adj_r2_avg": float(np.nanmean(r2s)), "_n_months": int(len(cdf)),
           "_avg_nobs": float(np.mean(nobs_list)) if nobs_list else float("nan")}
    for c in cdf.columns:
        nw = newey_west_mean(cdf[c], lags=lags, small_sample_correction=small_sample_correction)
        out[c] = {"coef": nw["mean"], "t": nw["t"]}
    return out


def assign_deciles_ascending(values, n: int = 10) -> pd.Series:
    """Ascending deciles: 1 = lowest, n = highest (published Table III labels).
    Group membership equals SAS `proc rank groups=10` (descending only flips labels)."""
    s = pd.to_numeric(pd.Series(values), errors="coerce")
    valid = s.dropna()
    m = len(valid)
    out = pd.Series(np.nan, index=s.index)
    if m < n:
        return out
    ranks = valid.rank(method="first")
    dec = np.ceil(ranks * n / m)
    dec = dec.clip(1, n).astype(int)
    out.loc[valid.index] = dec
    return out
