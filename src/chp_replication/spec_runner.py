"""Specification-tournament runner: resolve spec inheritance, translate sample specs into
build_sample kwargs, compute Table-I-like moments and sample/Table-II/Table-III/Table-IV scores."""
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from .paths import get_project_root


def load_spec_grid(path=None):
    path = Path(path) if path else get_project_root() / "config/spec_grid.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_spec(spec_id: str, all_specs: dict) -> dict:
    spec = dict(all_specs[spec_id])
    base = spec.pop("base", None)
    if base:
        b = resolve_spec(base, all_specs)
        b.update({k: v for k, v in spec.items() if k != "base"})
        return b
    return spec


def sample_spec_to_kwargs(spec: dict) -> dict:
    kw = {}
    direct_keys = ["redemption_keep_missing", "aum_rule", "currency_rule", "net_return_rule",
                   "include_categories", "duplicate_rule", "fund_identifier_rule"]
    for k in direct_keys:
        if k in spec:
            kw[k] = spec[k]
    if "backfill_drop_obs" in spec:
        kw["backfill_drop_obs"] = int(spec["backfill_drop_obs"])
    if "min_aum_threshold" in spec:
        kw["min_aum_threshold"] = float(spec["min_aum_threshold"])
    if spec.get("tass_source") == "tr_tass_old":
        rest = get_project_root() / "data/raw/restricted"
        kw["returns_path"] = rest / "tr_tass_old_returns.parquet"
        kw["info_path"] = rest / "tr_tass_old_info.parquet"
    return kw


TABLE1_TARGETS = {
    "fund_return_mean": 0.59,
    "fund_size_mean": 181.48, "fund_size_median": 55.99,
    "fund_age_mean": 80.66, "fund_age_median": 67,
    "management_fee_mean": 1.35, "management_fee_median": 1.50,
    "incentive_fee_mean": 15.05, "incentive_fee_median": 20,
    "high_water_mark_mean": 0.67,
    "lockup_dummy_mean": 0.32,
    "notice_period_months_mean": 1.44,
}


def compute_table1_moments(panel: pd.DataFrame) -> dict:
    def mm(col, kind):
        if col not in panel.columns:
            return float("nan")
        s = pd.to_numeric(panel[col], errors="coerce")
        return float(s.mean() if kind == "mean" else s.median())
    return {
        "fund_return_mean": mm("fund_return", "mean"),
        "fund_size_mean": mm("aum_million", "mean"),
        "fund_size_median": mm("aum_million", "median"),
        "fund_age_mean": mm("fund_age_months", "mean"),
        "fund_age_median": mm("fund_age_months", "median"),
        "management_fee_mean": mm("management_fee", "mean"),
        "management_fee_median": mm("management_fee", "median"),
        "incentive_fee_mean": mm("incentive_fee", "mean"),
        "incentive_fee_median": mm("incentive_fee", "median"),
        "high_water_mark_mean": mm("high_water_mark", "mean"),
        "lockup_dummy_mean": mm("lockup_dummy", "mean"),
        "notice_period_months_mean": mm("notice_period_months", "mean"),
    }


def table1_closeness(moments: dict) -> float:
    diffs = []
    for k, target in TABLE1_TARGETS.items():
        my = moments.get(k, float("nan"))
        if pd.notna(my) and target != 0:
            diffs.append(min(abs(my - target) / abs(target), 1.0))
    return float(1.0 - sum(diffs) / len(diffs)) if diffs else 0.0  # higher is better


def sample_score(fund_count: int, table1_close: float, months: int) -> dict:
    target_funds = 4073
    count_score = float(max(0.0, 1.0 - abs(fund_count - target_funds) / target_funds))
    months_score = float(min(months / 264.0, 1.0))
    total = 0.40 * count_score + 0.40 * table1_close + 0.20 * months_score
    return {"count_score": round(count_score, 3),
            "table1_close": round(float(table1_close), 3),
            "months_score": round(months_score, 3),
            "total": round(total, 3)}


# ----- Table III scoring -----
TABLE3_TARGETS = {"alpha": 0.59, "alpha_t": 3.55, "excess": 0.31, "excess_t": 3.16, "beta": 2.03}

# Alternative paper targets used when scoring non-baseline spec combinations.
# - UMich sentiment (M9) -> Table V
# - All-categories sample (S9) -> Internet Appendix IA.I
# - Alt-filter sample (S10) -> Internet Appendix IA.II
PAPER_TARGETS_BY_KEY = {
    "table_iii": {"alpha": 0.59, "alpha_t": 3.55, "excess": 0.31, "excess_t": 3.16, "beta": 2.03},
    "table_v_umich": {"alpha": 0.58, "alpha_t": 2.17, "excess": 0.26, "excess_t": 1.88, "beta": 2.03},
    "ia_i_all_categories": {"alpha": 0.52, "alpha_t": 2.61, "excess": 0.39, "excess_t": 3.04, "beta": 2.03},
    "ia_ii_alt_filters": {"alpha": 0.55, "alpha_t": 3.04, "excess": 0.34, "excess_t": 3.17, "beta": 2.03},
}


# ----- Primary-final eligibility (the main professor-facing Table III/IV replication) -----
# Primary = P0 immediate-next-month, BW-family sentiment (not UMich), main-paper sample (main or
# documented main sensitivity, NOT Internet-Appendix-only or diagnostic), paper/professor factor.
PRIMARY_SAMPLE_TIERS = {"main", "sensitivity_admissible"}
PRIMARY_FACTOR_TIERS = {"main_professor_guided", "sensitivity_hsieh", "sensitivity_professor", "main"}
BW_SENTIMENT_PREFIXES = ("M1", "M2", "M3", "M4", "M5", "M6")  # BW family; M9 (UMich) excluded


def primary_final_eligible(sample_adm: str, sentiment_id: str, factor_adm: str, table3_spec: str) -> tuple[bool, str]:
    reasons = []
    if not str(table3_spec).startswith("P0"):
        reasons.append(f"table3_spec={table3_spec} (not P0 immediate-next-month; IA robustness)")
    if not str(sentiment_id).split("_")[0] in BW_SENTIMENT_PREFIXES:
        reasons.append(f"sentiment={sentiment_id} (not BW family; UMich/other is diagnostic)")
    if sample_adm not in PRIMARY_SAMPLE_TIERS:
        reasons.append(f"sample admissibility={sample_adm} (IA-only/diagnostic, not main-paper)")
    if factor_adm not in PRIMARY_FACTOR_TIERS:
        reasons.append(f"factor admissibility={factor_adm} (not paper/professor admissible)")
    return (len(reasons) == 0), "; ".join(reasons)


def resolve_paper_target(sample_id: str, sentiment_id: str, sample_spec: dict, sentiment_spec: dict) -> tuple[str, dict]:
    """Return (target_label, target_dict) for the given (sample, sentiment) combination.
    Priority: UMich sentiment always -> Table V; IA samples -> IA targets; otherwise Table III.
    """
    if "Michigan" in sentiment_id or "UMCSENT" in sentiment_id or "M9" in sentiment_id:
        return "Table V (UMich)", PAPER_TARGETS_BY_KEY["table_v_umich"]
    if sample_id == "S9_all_11_categories" or sample_spec.get("admissibility") == "internet_appendix" and "all_11" in str(sample_spec.get("include_categories", "")):
        return "IA.I (all 11 categories)", PAPER_TARGETS_BY_KEY["ia_i_all_categories"]
    if sample_id == "S10_alternative_filters_24m_10m":
        return "IA.II (alt filters 24m/10m)", PAPER_TARGETS_BY_KEY["ia_ii_alt_filters"]
    return "Table III (baseline)", PAPER_TARGETS_BY_KEY["table_iii"]


def _closeness(x: float, target: float, scale: float | None = None) -> float:
    """1 when x==target; linear decay; 0 at distance==scale (default |target|)."""
    if pd.isna(x):
        return 0.0
    sc = abs(target) if scale is None else scale
    if sc == 0:
        return 1.0 if abs(x - target) < 1e-9 else 0.0
    return float(max(0.0, 1.0 - abs(x - target) / sc))


def table3_score(beta_spread: float, excess_spread: float, excess_t: float,
                 alpha_spread: float, alpha_t: float, targets: dict | None = None) -> dict:
    """Score a Table III combo against a target dict; defaults to baseline Table III (0.59 / 3.55 / 0.31 / 3.16)."""
    tgt = targets if targets is not None else TABLE3_TARGETS
    alpha_close = _closeness(alpha_spread, tgt["alpha"])
    alpha_t_close = _closeness(alpha_t, tgt["alpha_t"], scale=abs(tgt["alpha_t"]) or 3.55)
    excess_close = _closeness(excess_spread, tgt["excess"])
    excess_t_close = _closeness(excess_t, tgt["excess_t"], scale=abs(tgt["excess_t"]) or 3.16)
    beta_close = _closeness(beta_spread, tgt["beta"])
    sign_score = 1.0 if (pd.notna(alpha_spread) and alpha_spread > 0) else 0.0
    total = (0.35 * 0.5 * (alpha_close + alpha_t_close)
             + 0.25 * 0.5 * (excess_close + excess_t_close)
             + 0.10 * beta_close
             + 0.10 * sign_score
             + 0.20 * 0.5 * (alpha_close + excess_close))  # decile-fit proxy
    return {"alpha_close": round(alpha_close, 3), "alpha_t_close": round(alpha_t_close, 3),
            "excess_close": round(excess_close, 3), "excess_t_close": round(excess_t_close, 3),
            "beta_close": round(beta_close, 3), "sign_score": sign_score,
            "table3_score": round(total, 3)}


# ----- Table II sentiment moments + scaling -----
TABLE2_TARGETS = {"mean": -0.50, "sd": 1.39, "corr_mktrf": 0.16, "corr_smb": 0.22, "corr_INF": 0.22}
SENT_WIN_START = pd.Timestamp("1994-01-31")
SENT_WIN_END = pd.Timestamp("2018-12-31")


def sentiment_moments(factors: pd.DataFrame, sent_col: str) -> dict:
    """Compute Table-II-comparable moments of a sentiment-change series over 1994-2018."""
    if sent_col not in factors.columns:
        return {k: float("nan") for k in
                ["sentiment_mean", "sentiment_sd", "corr_mktrf", "corr_smb", "corr_INF", "n_obs"]}
    win = factors[(factors["month"] >= SENT_WIN_START) & (factors["month"] <= SENT_WIN_END)]
    s = win[sent_col]
    def corr(other):
        return float(s.corr(win[other])) if other in win.columns else float("nan")
    return {
        "sentiment_mean": float(s.mean()), "sentiment_sd": float(s.std()),
        "corr_mktrf": corr("mktrf"), "corr_smb": corr("smb"), "corr_INF": corr("INF"),
        "n_obs": int(s.dropna().shape[0]),
    }


def tableII_score(moments: dict) -> float:
    """Closeness of sentiment moments to Table II (mean/sd/correlations). Higher is better."""
    diffs = []
    diffs.append(_closeness(moments.get("sentiment_mean"), TABLE2_TARGETS["mean"]))
    diffs.append(_closeness(moments.get("sentiment_sd"), TABLE2_TARGETS["sd"]))
    diffs.append(_closeness(moments.get("corr_mktrf"), TABLE2_TARGETS["corr_mktrf"], scale=0.3))
    diffs.append(_closeness(moments.get("corr_smb"), TABLE2_TARGETS["corr_smb"], scale=0.3))
    diffs.append(_closeness(moments.get("corr_INF"), TABLE2_TARGETS["corr_INF"], scale=0.3))
    return round(float(np.mean(diffs)), 3)


def apply_tableII_scaling(factors: pd.DataFrame, sent_col: str, out_col: str | None = None) -> pd.DataFrame:
    """Linearly rescale a sentiment-change series so its 1994-2018 mean/sd match Table II (-0.50, 1.39).
    This is a monotone affine transform: it does NOT change decile sorting (Table III spread is
    scale-invariant) but it puts the beta column and Table IV coefficients on the paper's reported scale.
    """
    out_col = out_col or (sent_col + "_tableII_scaled")
    f = factors.copy()
    win = f[(f["month"] >= SENT_WIN_START) & (f["month"] <= SENT_WIN_END)][sent_col]
    mu, sd = float(win.mean()), float(win.std())
    if not np.isfinite(sd) or sd == 0:
        f[out_col] = f[sent_col]
        return f
    f[out_col] = (f[sent_col] - mu) / sd * TABLE2_TARGETS["sd"] + TABLE2_TARGETS["mean"]
    return f


# ----- Richer Table III decile metrics -----
def table3_decile_metrics(tbl: pd.DataFrame, benchmark_rows: list | None = None) -> dict:
    """Monotonicity of the decile alpha pattern + RMSE of replicated vs paper decile alphas.
    `tbl` is the run_table3 output (rows per decile + Spread). `benchmark_rows` = paper Table III
    rows with 'portfolio' and 'alpha' if available; else decile RMSE is NaN.
    """
    dec = tbl[tbl["portfolio"].str.match(r"^(\d|10)", na=False)].copy()
    alphas = pd.to_numeric(dec["alpha"], errors="coerce").to_numpy()
    # monotonicity: fraction of adjacent increases (decile1->10 alpha should rise)
    if len(alphas) >= 2 and np.isfinite(alphas).all():
        incr = np.sum(np.diff(alphas) > 0) / (len(alphas) - 1)
        # Spearman of alpha vs decile rank
        ranks = np.arange(1, len(alphas) + 1)
        mono_corr = float(pd.Series(alphas).corr(pd.Series(ranks), method="spearman"))
    else:
        incr, mono_corr = float("nan"), float("nan")
    rmse = float("nan")
    if benchmark_rows:
        bdf = pd.DataFrame(benchmark_rows)
        if "portfolio" in bdf and "alpha" in bdf:
            mrg = dec.merge(bdf[["portfolio", "alpha"]], on="portfolio", suffixes=("_repl", "_paper"))
            if len(mrg):
                d = pd.to_numeric(mrg["alpha_repl"], errors="coerce") - pd.to_numeric(mrg["alpha_paper"], errors="coerce")
                rmse = float(np.sqrt(np.nanmean(d.to_numpy() ** 2)))
    return {"monotonicity_frac_increasing": round(incr, 3) if pd.notna(incr) else None,
            "monotonicity_spearman": round(mono_corr, 3) if pd.notna(mono_corr) else None,
            "decile_alpha_rmse_vs_paper": round(rmse, 3) if pd.notna(rmse) else None}


# ----- Rolling-beta cache (VERSIONED + input-hashed, with integrity registry) -----
# code_version is a sha256 over the source files that affect beta/Table III/IV computation, so ANY
# edit to those files automatically invalidates the cache (no hand-maintained version string).
CODE_VERSION_SOURCE_FILES = [
    "src/chp_replication/rolling.py", "src/chp_replication/table3.py",
    "src/chp_replication/table4.py", "src/chp_replication/spec_runner.py",
    "config/spec_grid.yaml",
]


def _code_version_hash():
    import hashlib
    root = get_project_root()
    h = hashlib.sha256()
    for rel in CODE_VERSION_SOURCE_FILES:
        p = root / rel
        h.update(rel.encode())
        if p.exists():
            h.update(p.read_bytes())
        else:
            h.update(b"MISSING")
    return h.hexdigest()


ROLLING_BETA_CODE_VERSION = _code_version_hash()  # full sha256
ROLLING_BETA_CODE_VERSION_SHORT = ROLLING_BETA_CODE_VERSION[:12]
_CACHE_REGISTRY = []  # list of dicts; dump to cache_integrity_audit.xlsx after a run


def _hash_inputs(panel, factors, beta_method):
    """Hash the EXACT inputs to the beta regression so any change in the sentiment series, factor
    values, sample panel, or method invalidates the cache."""
    import hashlib
    from .rolling import BETA_FACTORS
    fac_cols = [c for c in (["sentiment_change"] + BETA_FACTORS) if c in factors.columns]
    fac_block = factors[["month"] + fac_cols].copy()
    fac_sig = int(pd.util.hash_pandas_object(fac_block.fillna(-9.99e9), index=False).sum() & 0xFFFFFFFFFFFF)
    pan_sig = (int(panel["fund_id"].nunique()), int(len(panel)),
               int(round(float(pd.to_numeric(panel["excess_return"], errors="coerce").fillna(0).sum()) * 100)))
    raw = f"{beta_method}|{ROLLING_BETA_CODE_VERSION_SHORT}|{fac_sig}|{pan_sig[0]}_{pan_sig[1]}_{pan_sig[2]}"
    return hashlib.md5(raw.encode()).hexdigest()[:12], fac_sig, pan_sig


def cached_rolling_betas(cfg, panel, factors, label: str, beta_method: str = "single_step", recompute=False):
    """Versioned cache of rolling sentiment betas. The filename embeds a hash of the exact regression
    inputs (sentiment series + factor values + panel signature) plus the beta_method and code version,
    so a stale cache from an older M3/two-step/F2 definition can NEVER be silently reused. Every call is
    logged to _CACHE_REGISTRY for the cache-integrity audit."""
    from .rolling import rolling_sentiment_betas
    from datetime import datetime
    cache_dir = get_project_root() / "data/processed/tournament/rolling_betas"
    cache_dir.mkdir(parents=True, exist_ok=True)
    h, fac_sig, pan_sig = _hash_inputs(panel, factors, beta_method)
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in label)
    cache_file = cache_dir / f"{safe}__{beta_method}__{h}.parquet"
    if cache_file.exists() and not recompute:
        betas = pd.read_parquet(cache_file)
        status = "reused_valid_hash_match"
    else:
        betas = rolling_sentiment_betas(cfg, panel, factors, beta_method=beta_method)
        try:
            betas.to_parquet(cache_file, index=False)
        except Exception:  # noqa: BLE001
            pass
        status = "recomputed"
    _CACHE_REGISTRY.append({
        "label": label, "sample_spec": label.split("__")[0] if "__" in label else label,
        "sentiment_spec": label.split("__")[1] if label.count("__") >= 1 else "",
        "factor_spec": label.split("__")[2] if label.count("__") >= 2 else "",
        "beta_method": beta_method, "code_version_hash": ROLLING_BETA_CODE_VERSION_SHORT,
        "code_version_source_files": ";".join(CODE_VERSION_SOURCE_FILES),
        "input_hash": h, "factor_hash": fac_sig, "sample_hash": f"{pan_sig[0]}_{pan_sig[1]}_{pan_sig[2]}",
        "n_funds": pan_sig[0], "panel_rows": pan_sig[1],
        "cache_file": cache_file.name, "rows": int(len(betas)),
        "status": status, "valid_for_current_run": True, "reason_if_invalid": "",
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    return betas


def dump_cache_registry(path):
    """Write the cache-integrity audit (every beta cache touched this run, with input + code-version hashes)."""
    from .utils import write_excel_with_sheets
    df = pd.DataFrame(_CACHE_REGISTRY)
    summary = pd.DataFrame([
        {"check": "code_version_hash (sha256, short)", "value": ROLLING_BETA_CODE_VERSION_SHORT},
        {"check": "code_version_full_sha256", "value": ROLLING_BETA_CODE_VERSION},
        {"check": "code_version_source_files", "value": ";".join(CODE_VERSION_SOURCE_FILES)},
        {"check": "n_cache_calls", "value": len(df)},
        {"check": "n_recomputed", "value": int((df["status"] == "recomputed").sum()) if len(df) else 0},
        {"check": "n_reused_valid", "value": int((df["status"] == "reused_valid_hash_match").sum()) if len(df) else 0},
        {"check": "all_valid_for_current_run", "value": bool(df["valid_for_current_run"].all()) if len(df) else True},
        {"check": "note", "value": "Cache key = md5(method + code_version_hash + factor_values_hash + panel_signature). "
                                    "code_version_hash = sha256 of rolling/table3/table4/spec_runner/spec_grid. "
                                    "Stale M3/two-step/F2 caches cannot be reused (different hash). Cache was purged before this run."},
    ])
    write_excel_with_sheets(path, {"cache_calls": df, "summary": summary})
    return df


def apply_factor_spec(factors: pd.DataFrame, factor_spec: dict, fred_raw_path: Path | None = None) -> pd.DataFrame:
    """Build a factors copy whose term_s/credit_s match the requested factor spec.
    Defaults to the current factors_main (= F0 professor FRED main)."""
    fac = factors.copy()
    ts = factor_spec.get("term_s")
    cs = factor_spec.get("credit_s")
    if ts in (None, "GS10_minus_TB3MS_level") and cs in (None, "delta_DBAA_minus_DGS10"):
        return fac  # F0 - already in factors_main
    if fred_raw_path is None:
        fred_raw_path = get_project_root() / "data/raw/public/fred_raw.csv"
    fred = pd.read_csv(fred_raw_path, parse_dates=["month"]).sort_values("month").reset_index(drop=True)
    if ts == "delta_DGS10":
        fred["term_s_new"] = fred["DGS10"].diff()
    elif ts == "GS10_minus_TB3MS_level":
        fred["term_s_new"] = fred["GS10"] - fred["TB3MS"]
    elif ts == "FRED_T10Y3M_level" and "T10Y3M" in fred.columns:
        fred["term_s_new"] = fred["T10Y3M"]  # FRED 10y-3m constant-maturity spread (level)
    elif ts == "DGS10_minus_DGS3MO_level" and {"DGS10", "DGS3MO"}.issubset(fred.columns):
        fred["term_s_new"] = fred["DGS10"] - fred["DGS3MO"]
    if cs == "delta_DBAA_minus_DGS10":
        spread = fred["DBAA"] - fred["DGS10"]
        fred["credit_s_new"] = spread.diff()
    keep = ["month"] + [c for c in ("term_s_new", "credit_s_new") if c in fred.columns]
    fred = fred[keep]
    fac = fac.merge(fred, on="month", how="left")
    if "term_s_new" in fac.columns:
        fac["term_s"] = fac["term_s_new"]
        fac = fac.drop(columns=["term_s_new"])
    if "credit_s_new" in fac.columns:
        fac["credit_s"] = fac["credit_s_new"]
        fac = fac.drop(columns=["credit_s_new"])
    return fac
