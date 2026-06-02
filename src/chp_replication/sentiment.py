"""Baker-Wurgler sentiment LEVEL reconstruction + CHANGES index.

Follows the workbook's STATA program: nipo -> 12-month moving sum; ripo -> IPO-weighted 12-month
average; 12-month lags on ripo & pdnd; orthogonalize each proxy to macro growth; correlation-PCA
(proxies standardized) -> first PC. The LEVEL reconstruction is validated against the file's
SENT_ORTH (a high correlation confirms the transforms). The CHANGES index = first PC of the
CHANGES in the orthogonalized proxies.

Closest-feasible; per Prof. Arif it need not match Wurgler exactly. The turnover proxy is NOT in
the provided file (BW dropped it), so it is omitted (documented).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

MACRO = ["gindpro", "gconsdur", "gconsnon", "gconsserv", "gemploy", "recess1"]
PROXY_COLS = {"cef": "cef", "nipo": "nipo_am", "lagripo": "lagripo", "lagpdnd": "lagpdnd", "s": "s"}


def _resid(y: pd.Series, X: pd.DataFrame) -> pd.Series:
    d = pd.concat([y.rename("y"), X], axis=1)
    v = d.dropna()
    if len(v) < 24:
        return pd.Series(np.nan, index=y.index)
    res = sm.OLS(v["y"], sm.add_constant(v[X.columns])).fit()
    return d["y"] - res.predict(sm.add_constant(d[X.columns], has_constant="add"))


def _pca_first(M: np.ndarray) -> np.ndarray:
    M = M - M.mean(axis=0)
    w, V = np.linalg.eigh(np.cov(M, rowvar=False))
    return M @ V[:, int(np.argmax(w))]


def _std(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and not np.isnan(sd) else s


def build_sentiment_candidates(post_xlsx) -> pd.DataFrame:
    raw = pd.read_excel(post_xlsx, sheet_name="DATA")
    raw.columns = [str(c).strip() for c in raw.columns]
    df = raw.copy()
    for c in ["yearmo", "pdnd", "ripo", "nipo", "cefd", "s", "indpro", "consdur",
              "consnon", "consserv", "recess", "employ", "cpi", "SENT", "SENT_ORTH"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("yearmo").reset_index(drop=True)

    df["gindpro"] = df["indpro"] / df["indpro"].shift(12) - 1
    df["gemploy"] = df["employ"] / df["employ"].shift(12) - 1
    for c in ["consdur", "consnon", "consserv"]:
        df["g" + c] = (df[c] / df[c].shift(12)) / (df["cpi"] / df["cpi"].shift(12)) - 1
    df["recess1"] = df["recess"].shift(1)

    # STATA transforms: nipo 12m moving sum; ripo IPO-weighted 12m average
    df["nipo_am"] = df["nipo"].rolling(12, min_periods=12).sum()
    ripo_w = df["ripo"].where(df["nipo"] > 0, 0).fillna(0)
    num = (ripo_w * df["nipo"].fillna(0)).rolling(12, min_periods=12).sum()
    den = df["nipo"].fillna(0).rolling(12, min_periods=12).sum().replace(0, np.nan)
    df["ripo_am"] = num / den
    df["cef"] = df["cefd"]
    df["lagripo"] = df["ripo_am"].shift(12)
    df["lagpdnd"] = df["pdnd"].shift(12)

    Xmac = df[MACRO]
    E = pd.DataFrame({name: _resid(df[col], Xmac) for name, col in PROXY_COLS.items()})

    # LEVEL reconstruction (validation): correlation-PCA of orthogonalized proxy LEVELS
    EL = E.apply(_std)
    maskL = EL.notna().all(axis=1)
    lvl = pd.Series(np.nan, index=df.index, dtype=float)
    lvl[maskL] = _pca_first(EL[maskL].to_numpy())
    lvl = _std(lvl)
    if lvl.corr(df["SENT_ORTH"]) < 0:
        lvl = -lvl
    df["my_SENT_ORTH_recon"] = lvl

    # M1 CHANGES index: correlation-PCA of CHANGES in orthogonalized proxies (columns standardized).
    C = E.diff().apply(_std)
    maskC = C.notna().all(axis=1)
    chg = pd.Series(np.nan, index=df.index, dtype=float)
    chg[maskC] = _pca_first(C[maskC].to_numpy())
    chg = _std(chg)
    if chg.corr(df["SENT_ORTH"].diff()) < 0:
        chg = -chg
    df["sent_change_pca"] = chg

    # M2 covariance-PCA: PCA on the CHANGES of orthogonalized proxies WITHOUT per-column standardization.
    Craw = E.diff()
    maskC2 = Craw.notna().all(axis=1)
    chg2 = pd.Series(np.nan, index=df.index, dtype=float)
    chg2[maskC2] = _pca_first(Craw[maskC2].to_numpy())
    chg2 = _std(chg2)
    if chg2.corr(df["SENT_ORTH"].diff()) < 0:
        chg2 = -chg2
    df["sent_change_covpca"] = chg2

    # M3 orthogonalize-CHANGES-then-PCA (alternative ORTHOGONALIZATION ORDER vs M1): first-difference each
    # raw proxy, orthogonalize the CHANGES to macro, then correlation-PCA. (M1 orthogonalizes levels, then
    # differences, then PCAs; M3 differences first, then orthogonalizes the changes, then PCAs.)
    dP = pd.DataFrame({name: df[col].diff() for name, col in PROXY_COLS.items()})
    EC = pd.DataFrame({name: _resid(dP[name], Xmac) for name in PROXY_COLS}).apply(_std)
    maskEC = EC.notna().all(axis=1)
    chg3 = pd.Series(np.nan, index=df.index, dtype=float)
    chg3[maskEC] = _pca_first(EC[maskEC].to_numpy())
    chg3 = _std(chg3)
    if chg3.corr(df["SENT_ORTH"].diff()) < 0:
        chg3 = -chg3
    df["sent_change_orthchg_pca"] = chg3

    # M4 level-PC-DIFFERENCE: first difference of the corr-PCA LEVEL index (diagnostic, NOT BW-changes def).
    df["sent_change_levelpc_diff"] = _std(df["my_SENT_ORTH_recon"].diff())

    df["month"] = pd.to_datetime(df["yearmo"].astype("Int64").astype(str), format="%Y%m",
                                 errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    df["delta_SENT_ORTH"] = df["SENT_ORTH"].diff()
    df["delta_SENT"] = df["SENT"].diff()
    return df[["month", "yearmo", "sent_change_pca", "sent_change_covpca", "sent_change_orthchg_pca",
               "sent_change_levelpc_diff", "my_SENT_ORTH_recon", "delta_SENT_ORTH", "delta_SENT",
               "SENT", "SENT_ORTH"]]
