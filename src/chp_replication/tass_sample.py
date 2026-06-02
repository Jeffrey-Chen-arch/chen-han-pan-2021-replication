"""Build the CHP fund-month sample from raw TASS, with a documented sample funnel.

Applies the paper's screens in order and records counts at each step. Units are inferred from
the data. Output panel is restricted (data/processed) and never packaged.
"""
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

from .utils import write_excel_with_sheets, month_end_timestamp
from .tass_clean import (standardize_return_to_percent, standardize_aum_to_million,
                         standardize_fee_to_percent, parse_redemption_frequency_to_months)

EXCLUDE_STRATEGIES = {"emerging markets", "fixed income arbitrage", "managed futures", "dedicated short bias"}
STYLE_CODE = {
    "convertible arbitrage": "ca", "equity market neutral": "emn", "global macro": "gm",
    "long/short equity hedge": "lseh", "multi-strategy": "ms", "fund of funds": "fof",
}
RETURN_ERROR_ABS_PCT = 300.0  # drop |monthly return| > 300% as clear data errors (documented; NOT winsorization)


def _fund_signature(d: pd.DataFrame):
    return hash(tuple(zip(d["month"].astype("int64").tolist(), d["fund_return"].round(4).tolist())))


def _drop_highconf_duplicate_funds(r: pd.DataFrame):
    sigs = (r.sort_values(["fund_id", "month"])
              .groupby("fund_id", sort=False)
              .apply(_fund_signature, include_groups=False))
    bysig = defaultdict(list)
    for fid, sg in sigs.items():
        bysig[sg].append(fid)
    drop, dup_sets = [], 0
    maxaum = r.groupby("fund_id")["aum_million"].max()
    for sg, fids in bysig.items():
        if len(fids) > 1:
            dup_sets += 1
            keep = max(fids, key=lambda f: (maxaum.get(f, -1) if pd.notna(maxaum.get(f, np.nan)) else -1))
            drop += [f for f in fids if f != keep]
    out = r[~r["fund_id"].isin(drop)]
    return out, f"{dup_sets} identical-history duplicate sets; dropped {len(drop)} funds", drop


def _build_controls(r: pd.DataFrame) -> pd.DataFrame:
    r = r.copy()
    r["log_fund_size"] = np.log(r["aum_dollars"].where(r["aum_dollars"] > 0))
    inc = pd.to_datetime(r.get("inceptiondate"), errors="coerce")
    ps = pd.to_datetime(r.get("performancestartdate"), errors="coerce")
    start_date = inc.fillna(ps)
    first_month = r.groupby("fund_id")["month"].transform("min")
    start_date = start_date.fillna(first_month)
    age = (r["month"].dt.year - start_date.dt.year) * 12 + (r["month"].dt.month - start_date.dt.month)
    r["fund_age_months"] = age.clip(lower=1)
    r["log_fund_age"] = np.log(r["fund_age_months"])
    r["management_fee"], _ = standardize_fee_to_percent(pd.to_numeric(r.get("managementfee"), errors="coerce"))
    r["incentive_fee"], _ = standardize_fee_to_percent(pd.to_numeric(r.get("incentivefee"), errors="coerce"))
    r["high_water_mark"] = pd.to_numeric(r.get("highwatermark"), errors="coerce")  # already 0/1; NaN preserved
    lk = pd.to_numeric(r.get("lockupperiod"), errors="coerce")
    r["lockup_dummy"] = np.where(lk.notna(), (lk > 0).astype(float), np.nan)  # preserve missing (not 0)
    r["notice_period_months"] = pd.to_numeric(r.get("redemptionnoticeperiod"), errors="coerce") / 30.0
    cat = r["primarycategory"].astype(str).str.strip().str.lower()
    for name, code in STYLE_CODE.items():
        r[f"style_{code}"] = (cat == name).astype(float)
    return r


def build_sample(cfg: dict, redemption_keep_missing: bool = False, aum_rule: str = "hybrid", save: bool = True,
                 returns_path=None, info_path=None, min_obs_override=None,
                 currency_rule: str = "usd_code_only",
                 net_return_rule: str = "strict_net",
                 include_categories: str = "paper_7",
                 backfill_drop_obs=None, min_aum_threshold=None,
                 duplicate_rule: str = "exact_and_highconf",
                 fund_identifier_rule: str = "productreference"):
    root = Path(cfg["project"]["root"])
    rest = root / "data/raw/restricted"
    proc = root / "data/processed"; proc.mkdir(parents=True, exist_ok=True)
    diag = root / "output/diagnostics"; diag.mkdir(parents=True, exist_ok=True)
    s = cfg["sample"]
    start = pd.Timestamp(s["start_month"] + "-01") + pd.offsets.MonthEnd(0)
    end = pd.Timestamp(s["end_month"] + "-01") + pd.offsets.MonthEnd(0)
    min_aum = float(min_aum_threshold) if min_aum_threshold is not None else float(s["min_aum_million"])
    backfill = int(backfill_drop_obs) if backfill_drop_obs is not None else int(s["backfill_drop_observations"])
    min_obs = int(s["min_obs_fund_total"])
    if min_obs_override is not None:
        min_obs = int(min_obs_override)

    ret = pd.read_parquet(returns_path or (rest / "tass_returns.parquet"))
    info = pd.read_parquet(info_path or (rest / "tass_fund_info.parquet")).rename(columns={"productreference": "fund_id"})

    funnel, decisions = [], {}

    def step(code, desc, df, note=""):
        funnel.append({"step": code, "description": desc, "funds": int(df["fund_id"].nunique()),
                       "fund_months": int(len(df)), "notes": note})

    r = ret.rename(columns={"productreference": "fund_id", "rateofreturn": "ret_raw",
                            "estimatedassets": "aum_raw", "date": "date_raw"})
    r["month"] = month_end_timestamp(r["date_raw"])
    r["fund_return"], decisions["return_unit"] = standardize_return_to_percent(r["ret_raw"])
    r["aum_million"], decisions["aum_unit"] = standardize_aum_to_million(r["aum_raw"])
    r["aum_dollars"] = r["aum_million"] * 1_000_000.0  # derive from standardized millions (robust to raw units)
    step("01_raw_input", "raw productperformance rows", r)

    r = r.merge(info, on="fund_id", how="left")
    step("02_after_merge_fund_info", "merge wrds_productdetails (live+graveyard both kept)", r)

    # ----- S14 fund_identifier_rule: cluster productreferences into "funds" -----
    if fund_identifier_rule == "name_company_cluster":
        comp_path = rest / "tass_company.parquet"
        comp = pd.read_parquet(comp_path)
        mf = (comp[comp["companytype"] == "Management Firm"]
              [["productreference", "companyid", "companyname"]]
              .drop_duplicates("productreference")
              .rename(columns={"productreference": "fund_id"}))
        r = r.merge(mf, on="fund_id", how="left")
        # Cluster_id: prefer (companyid, primarycategory, currencycode) — fund-family + strategy + currency.
        # Funds without companyid fall back to their own productreference (keeps them as their own cluster).
        cat = r["primarycategory"].astype(str).str.lower().str.strip()
        ccy = r["currencycode"].astype(str).str.upper().str.strip()
        cluster_key = (r["companyid"].astype("Int64").astype(str)
                       + "|" + cat + "|" + ccy)
        cluster_key = cluster_key.where(r["companyid"].notna(),
                                        "NM|" + r["fund_id"].astype(str))
        # Map cluster_key strings to integer cluster_id (stable)
        codes, _ = pd.factorize(cluster_key, sort=True)
        r["cluster_id"] = codes.astype(np.int64)
        n_pre = int(r["fund_id"].nunique())
        n_post = int(r["cluster_id"].nunique())
        # Within (cluster_id, month) keep highest AUM (ties broken by latest first appearance)
        r = (r.sort_values(["cluster_id", "month", "aum_million"], ascending=[True, True, False])
              .drop_duplicates(["cluster_id", "month"], keep="first"))
        # Replace fund_id with cluster_id so downstream filters/counts use clusters as "funds"
        r["original_productreference"] = r["fund_id"]
        r["fund_id"] = r["cluster_id"]
        step("02b_after_cluster_identifier",
             f"cluster productrefs by companyid+category+currency ({n_pre}->{n_post} clusters)",
             r, f"fund_identifier_rule={fund_identifier_rule}")

    r = r[r["month"].notna() & (r["month"] >= pd.Timestamp("1990-01-31")) & (r["month"] <= pd.Timestamp("2026-12-31"))]
    step("03_after_date_standardization", "drop missing/sentinel dates (e.g. 1900)", r)

    n0 = len(r)
    r = r[r["fund_return"].abs() <= RETURN_ERROR_ABS_PCT]
    step("03b_after_return_error_guard", f"drop |monthly return|>{RETURN_ERROR_ABS_PCT}% as data errors", r, f"removed {n0 - len(r)} obs")

    if duplicate_rule != "none":
        r = r.sort_values(["fund_id", "month"]).drop_duplicates(["fund_id", "month"], keep="first")
    step("04_after_drop_duplicate_fund_month",
         f"exact fund-month de-duplication ({duplicate_rule})", r)

    r["obs_rank"] = r.groupby("fund_id").cumcount()
    r = r[r["obs_rank"] >= backfill]
    step("05_after_backfill_drop_first_12", f"drop first {backfill} reported months per fund", r)

    r = r[(r["month"] >= start) & (r["month"] <= end)]
    step("06_after_restrict_1994_2018", "restrict to sample period", r)

    cc = r["currencycode"].astype(str).str.upper()
    cd = r.get("currencydescription", pd.Series("", index=r.index)).astype(str).str.upper()
    if currency_rule == "usd_code_only":
        r = r[cc == "USD"]
    elif currency_rule == "usd_code_or_description":
        r = r[(cc == "USD") | cd.str.contains("DOLLAR", na=False)]
    elif currency_rule == "usd_or_missing":
        miss = cc.isin(["NONE", "NAN", ""]) | r["currencycode"].isna()
        r = r[(cc == "USD") | cd.str.contains("DOLLAR", na=False) | miss]
    step("07_after_usd_filter", f"currency_rule={currency_rule}", r)

    gn = r["grossnett"].astype(str).str.upper()
    if net_return_rule == "strict_net":
        r = r[gn == "N"]
    elif net_return_rule == "net_or_missing":
        miss = gn.isin(["NONE", "NAN", ""]) | r["grossnett"].isna()
        r = r[(gn == "N") | miss]
    step("08_after_net_of_fee_filter", f"net_return_rule={net_return_rule}", r)

    cat = r["primarycategory"].astype(str).str.strip().str.lower()
    if include_categories == "paper_7":
        r = r[~cat.isin(EXCLUDE_STRATEGIES)]
        step("09_after_strategy_filter", "paper_7: exclude EM/FIA/MF/DSB", r)
    else:  # "all_11"
        step("09_after_strategy_filter", "all_11: NO category exclusion (IA.I)", r)

    r["redemption_months"] = r["redemptionfrequency"].map(parse_redemption_frequency_to_months)
    if redemption_keep_missing:
        r = r[(r["redemption_months"] <= 1.0) | (r["redemption_months"].isna())]
        step("10_after_redemption_filter", "monthly-or-higher redemption (missing KEPT, sensitivity)", r)
    else:
        r = r[r["redemption_months"] <= 1.0]
        step("10_after_redemption_filter", "monthly-or-higher redemption (missing dropped, main)", r)

    if aum_rule == "strict":
        r = r[r["aum_million"].notna() & (r["aum_million"] >= min_aum)]
        step("11_after_aum_filter", f"strict: keep only nonmissing AUM>=${min_aum}m each month", r)
    elif aum_rule == "lenient":
        funds_ok = r.groupby("fund_id")["aum_million"].max().pipe(lambda s: s[s >= min_aum].index)
        r = r[r["fund_id"].isin(funds_ok)]
        step("11_after_aum_filter", f"lenient: fund ever AUM>=${min_aum}m; keep ALL months", r)
    else:  # hybrid (main)
        funds_ok = r.groupby("fund_id")["aum_million"].max().pipe(lambda s: s[s >= min_aum].index)
        r = r[r["fund_id"].isin(funds_ok)]
        r = r[~(r["aum_million"].notna() & (r["aum_million"] < min_aum))]
        step("11_after_aum_filter", f"hybrid: fund has any AUM>=${min_aum}m; drop nonmissing<${min_aum}m; keep missing", r)

    if duplicate_rule == "exact_and_highconf":
        r, dup_note, _ = _drop_highconf_duplicate_funds(r)
        step("11b_after_duplicate_fund_drop", "drop high-confidence duplicate funds (identical histories)", r, dup_note)
    else:
        step("11b_after_duplicate_fund_drop", f"skipped (duplicate_rule={duplicate_rule})", r)

    cnt = r.groupby("fund_id")["fund_return"].transform("count")
    r = r[cnt >= min_obs]
    step("12_after_min30_obs", f"require >= {min_obs} return obs per fund", r)

    fac = pd.read_parquet(proc / "factors_main.parquet")[["month", "rf"]]
    r = r.merge(fac, on="month", how="left")
    r["excess_return"] = r["fund_return"] - r["rf"]
    step("13_after_merge_rf_excess", "merge rf; excess_return = ret - rf", r, f"rf missing rows: {int(r['rf'].isna().sum())}")

    r = _build_controls(r)
    keep = ["fund_id", "month", "fund_return", "rf", "excess_return", "aum_million", "aum_dollars",
            "primarycategory", "log_fund_size", "fund_age_months", "log_fund_age",
            "management_fee", "incentive_fee", "high_water_mark", "lockup_dummy", "notice_period_months",
            "redemption_months", "live_graveyard"] + [f"style_{v}" for v in STYLE_CODE.values()]
    panel = r[[c for c in keep if c in r.columns]].sort_values(["fund_id", "month"]).reset_index(drop=True)
    fdf = pd.DataFrame(funnel)
    fdf["lost_funds"] = (fdf["funds"].shift(1) - fdf["funds"]).fillna(0).astype(int)
    fdf["lost_fund_months"] = (fdf["fund_months"].shift(1) - fdf["fund_months"]).fillna(0).astype(int)
    if save:
        panel.to_parquet(proc / "tass_sample_main.parquet", index=False)
        ddf = pd.DataFrame([{"item": k, "decision": v} for k, v in decisions.items()])
        tgt = pd.DataFrame([{"paper_final_funds": cfg["validation"]["paper_final_fund_count_target"],
                             "my_final_funds": int(panel["fund_id"].nunique()), "my_fund_months": int(len(panel)),
                             "month_min": f"{panel['month'].min():%Y-%m}", "month_max": f"{panel['month'].max():%Y-%m}"}])
        write_excel_with_sheets(diag / "sample_funnel.xlsx",
                                {"funnel": fdf, "unit_decisions": ddf, "final_vs_paper": tgt})
        print(fdf.to_string(index=False))
        print("\nunit decisions:", decisions)
        print("FINAL:", tgt.to_dict("records")[0])
    return panel, fdf
