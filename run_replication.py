"""Reproduce the headline Table III and Table IV results of Chen, Han & Pan (2021),
"Sentiment Trading and Hedge Fund Returns" (Journal of Finance), for the closest-feasible
primary specification.

Primary specification
----------------------
  Sample     : U.S. equity-oriented hedge funds, monthly-or-better redemption (missing flags kept).
  Sentiment  : first principal component of orthogonalized Baker-Wurgler proxy CHANGES.
  Factors    : Fung-Hsieh seven factors + momentum + Pastor-Stambaugh liquidity
               (term/credit proxied from FRED; inflation and default spread in the beta regression).

Inputs (not distributed; see README "Data")
  data/processed/factors_main.parquet   - monthly factors + sentiment series
  data/raw/restricted/                  - Lipper/TASS fund return and info extracts (licensed)

Run
  python run_replication.py

Outputs
  results/table3_table4_results.xlsx    - Table III (deciles + spread) and Table IV, with paper values
  console summary of the headline numbers
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from chp_replication.config import load_config  # noqa: E402
from chp_replication.spec_runner import (load_spec_grid, resolve_spec,  # noqa: E402
                                         sample_spec_to_kwargs, apply_factor_spec)
from chp_replication.tass_sample import build_sample  # noqa: E402
from chp_replication.rolling import rolling_sentiment_betas  # noqa: E402
from chp_replication.table3 import run_table3  # noqa: E402
from chp_replication.table4 import run_table4  # noqa: E402

MAIN_SAMPLE = "S1_redemption_keep_missing"
MAIN_SENTIMENT_COLUMN = "sent_change_orthchg_pca"   # first PC of orthogonalized proxy changes
MAIN_FACTOR = "F0_professor_fred_main"
PAPER_T3 = {"beta_spread": 2.03, "excess_spread": 0.31, "excess_t": 3.16, "alpha_spread": 0.59, "alpha_t": 3.55}


def main():
    cfg = load_config()
    grid = load_spec_grid()

    spec = resolve_spec(MAIN_SAMPLE, grid["sample_specs"])
    panel, _ = build_sample(cfg, save=False, **sample_spec_to_kwargs(spec))

    factors_path = ROOT / "data/processed/factors_main.parquet"
    if not factors_path.exists():
        raise SystemExit(f"Missing {factors_path}. See README 'Data' - factor/sentiment inputs are "
                         "not distributed and must be prepared first.")
    factors = pd.read_parquet(factors_path)
    fac = apply_factor_spec(factors, grid["factor_specs"][MAIN_FACTOR]).copy()
    fac["sentiment_change"] = fac[MAIN_SENTIMENT_COLUMN]

    betas = rolling_sentiment_betas(cfg, panel, fac)
    tbl3, _, _, _ = run_table3(cfg, panel, betas, fac)
    tbl4, _, _, _, _ = run_table4(cfg, panel, betas, fac, alpha_construction="paper_rolling")

    spread = tbl3[tbl3["portfolio"] == "Spread (10-1)"].iloc[0]
    print("=" * 64)
    print(f"Funds: {panel['fund_id'].nunique()}   Months: {int(spread['n_months'])}")
    print("Table III high-minus-low spread (replication vs paper):")
    print(f"  excess spread {spread['excess_return']:.2f} (t {spread['excess_t']:.2f})  vs  "
          f"{PAPER_T3['excess_spread']} (t {PAPER_T3['excess_t']})")
    print(f"  alpha  spread {spread['alpha']:.2f} (t {spread['alpha_t']:.2f})  vs  "
          f"{PAPER_T3['alpha_spread']} (t {PAPER_T3['alpha_t']})")
    print("Table IV sentiment-beta coefficients (replication):")
    print(tbl4[["model", "sentiment_beta_coef", "sentiment_beta_t", "adj_r2"]].to_string(index=False))
    print("=" * 64)

    out = ROOT / "results/table3_table4_results.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out) as xl:
        tbl3.to_excel(xl, sheet_name="Table_III", index=False)
        tbl4.to_excel(xl, sheet_name="Table_IV", index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
