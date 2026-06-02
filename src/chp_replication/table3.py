"""Table III: decile portfolio sorts on sentiment beta (SAS-aligned).

Each return month m, sort funds into 10 equal-weighted deciles by their sentiment beta
(estimated over [m-36, m-1]); 1=Low, 10=High (== SAS proc-rank groups; spread 10-1 == SAS _0-_9).
Per decile: avg sentiment beta, NW(2) mean excess return, and 9-factor alpha (mktrf, smb, umd,
ptfsbd, ptfsfx, ptfscom, term_s, credit_s, liq_v) intercept with NW(2) t - EXCLUDES sentiment/INF/def.
"""
import numpy as np
import pandas as pd

from .regression import newey_west_mean, time_series_alpha, assign_deciles_ascending
from .rolling import ALPHA_FACTORS
from .config import load_benchmarks

PORT_LABELS = {1: "1 (Low)", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10 (High)"}


def run_table3(cfg: dict, panel: pd.DataFrame, betas: pd.DataFrame, factors: pd.DataFrame,
               skip_months: int = 0, holding_months: int = 1, min_abs_beta_t: float | None = None,
               holding_method: str = "overlap"):
    """Table III decile sort. Variants (Internet Appendix robustness):
      skip_months   -> P2: skip k months between formation and the held return (k=1 = skip-one-month).
      holding_months-> P4/P5: h-month holding (h=3, h=6).
      holding_method-> 'overlap' (IA.V default; each calendar month = EW avg of the h most-recent
                       formation cohorts, Jegadeesh-Titman) or 'forward_average' (each formation month
                       sorted once; held return = avg of the fund's own next-h excess returns).
      min_abs_beta_t-> LEGACY diagnostic only (filter funds by |sentiment-beta t|). The genuine IA.IV
                       two-step is implemented as beta_method='two_step_5pct' in rolling_sentiment_betas
                       (per-window factor selection); pass those betas with P3 and DO NOT set this.
    """
    lags = int(cfg["regressions"]["newey_west_lags"])
    cols = ["fund_id", "return_month", "sentiment_beta"]
    if "beta_t" in betas.columns:
        cols.append("beta_t")
    b = betas.rename(columns={"return_month": "month"})[[c if c != "return_month" else "month" for c in cols]]
    if min_abs_beta_t is not None and "beta_t" in b.columns:  # legacy significant-beta filter
        b = b[b["beta_t"].abs() >= float(min_abs_beta_t)]
    panel_rm = panel[["fund_id", "month", "excess_return"]]
    base_shift = max(skip_months, 0)

    if holding_method == "forward_average" and (holding_months > 1 or base_shift > 0):
        # Held return at formation month m = mean of fund's excess returns over [m+skip, m+skip+h-1].
        rets = []
        for j in range(base_shift, base_shift + holding_months):
            tmp = panel_rm.copy()
            tmp["month"] = tmp["month"] - pd.offsets.MonthEnd(j)  # bring return at m+j back to formation m
            rets.append(tmp)
        fwd = (pd.concat(rets).groupby(["fund_id", "month"])["excess_return"].mean().reset_index())
        m = b.merge(fwd, on=["fund_id", "month"], how="inner").dropna(subset=["sentiment_beta", "excess_return"])
        m["decile"] = m.groupby("month")["sentiment_beta"].transform(lambda s: assign_deciles_ascending(s, 10))
        m = m.dropna(subset=["decile"]); m["decile"] = m["decile"].astype(int)
        g = (m.groupby(["month", "decile"])
               .agg(ew_ret=("excess_return", "mean"), avg_beta=("sentiment_beta", "mean"), n=("fund_id", "size"))
               .reset_index())
    else:
        # OVERLAP (default): shift formation beta forward by skip..skip+h-1; each calendar month is the
        # EW average across the active formation cohorts (overlapping calendar portfolio).
        parts = []
        for h in range(holding_months):
            shift = base_shift + h if holding_months > 1 else base_shift
            bb = b.copy()
            if shift > 0:
                bb["month"] = bb["month"] + pd.offsets.MonthEnd(shift)
            bb["_hlag"] = h
            parts.append(bb)
        bcat = pd.concat(parts, ignore_index=True)
        m = (bcat.merge(panel_rm, on=["fund_id", "month"], how="inner")
                 .dropna(subset=["sentiment_beta", "excess_return"]))
        m["decile"] = m.groupby(["month", "_hlag"])["sentiment_beta"].transform(lambda s: assign_deciles_ascending(s, 10))
        m = m.dropna(subset=["decile"]); m["decile"] = m["decile"].astype(int)
        g0 = (m.groupby(["month", "_hlag", "decile"])
                .agg(ew_ret=("excess_return", "mean"), avg_beta=("sentiment_beta", "mean"), n=("fund_id", "size"))
                .reset_index())
        g = (g0.groupby(["month", "decile"])
               .agg(ew_ret=("ew_ret", "mean"), avg_beta=("avg_beta", "mean"), n=("n", "mean"))
               .reset_index())
    ret_ts = g.pivot(index="month", columns="decile", values="ew_ret").sort_index()
    beta_ts = g.pivot(index="month", columns="decile", values="avg_beta").sort_index()
    cnt_ts = g.pivot(index="month", columns="decile", values="n").sort_index()
    ret_ts["spread"] = ret_ts[10] - ret_ts[1]

    af = factors[["month"] + ALPHA_FACTORS].set_index("month")
    rows = []
    for col in list(range(1, 11)) + ["spread"]:
        s = ret_ts[col].dropna()
        nw = newey_west_mean(s, lags=lags)
        merged = ret_ts[[col]].join(af, how="left").dropna()
        ar = time_series_alpha(merged[col].values, merged[ALPHA_FACTORS], lags=lags)
        alpha = float(ar.params["const"]) if ar is not None else np.nan
        alpha_t = float(ar.tvalues["const"]) if ar is not None else np.nan
        if col == "spread":
            avg_beta = beta_ts[10].mean() - beta_ts[1].mean()
            label = "Spread (10-1)"
        else:
            avg_beta = beta_ts[col].mean()
            label = PORT_LABELS[col]
        rows.append({"portfolio": label, "sentiment_beta": round(float(avg_beta), 2),
                     "excess_return": round(nw["mean"], 2), "excess_t": round(nw["t"], 2),
                     "alpha": round(alpha, 2), "alpha_t": round(alpha_t, 2), "n_months": nw["nobs"]})
    tbl = pd.DataFrame(rows)

    bdf = pd.DataFrame(load_benchmarks()["table3"]["rows"])
    comp = tbl.merge(bdf, on="portfolio", suffixes=("_repl", "_paper"))
    for c in ["sentiment_beta", "excess_return", "excess_t", "alpha", "alpha_t"]:
        comp[f"{c}_diff"] = (comp[f"{c}_repl"] - comp[f"{c}_paper"]).round(2)
    comp = comp[["portfolio", "sentiment_beta_repl", "sentiment_beta_paper",
                 "excess_return_repl", "excess_return_paper", "excess_return_diff",
                 "alpha_repl", "alpha_paper", "alpha_diff",
                 "alpha_t_repl", "alpha_t_paper"]]
    return tbl, comp, ret_ts.reset_index(), cnt_ts.reset_index()
