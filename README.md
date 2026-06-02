# Replication of Chen, Han & Pan (2021): Tables III & IV

A closest-feasible replication of *"Sentiment Trading and Hedge Fund Returns,"* Chen, Han, and Pan,
*Journal of Finance* (2021) — Table III (decile sorts on the rolling sentiment beta) and Table IV
(Fama-MacBeth of next-month performance on the sentiment beta).

## Result

The closest paper-eligible specification reproduces the Table III high-minus-low **excess-return spread
almost exactly (0.30% vs. 0.31% per month)** and about **73% of the alpha-spread magnitude (0.43% vs.
0.59%)**, with weaker t-statistics. For Table IV the sentiment-beta coefficients remain positive with
comparable implied effects but lower significance. The full results, methods, and discrepancy audit are
in [`results/CHP_replication_report.pdf`](results/CHP_replication_report.pdf).

## How to run

```bash
pip install -r requirements.txt
python run_replication.py
```

`run_replication.py` builds the primary specification and writes Table III and Table IV to
`results/table3_table4_results.xlsx`. The project root is detected automatically, so it runs from any
location after cloning.

## Data (not included)

Fund data is **Lipper/TASS via WRDS** (licensed) and is **not distributed** here; no credentials are
stored in this repository. Public factor inputs come from Kenneth French's data library, David Hsieh's
trend-following (PTFS) factors, FRED, the Pastor-Stambaugh liquidity factor, and the Baker-Wurgler
sentiment index. To run end-to-end, place the prepared monthly factor file at
`data/processed/factors_main.parquet` and the TASS extracts under `data/raw/restricted/`.

## Layout

```
run_replication.py        single entry point (primary specification -> Table III & IV)
src/chp_replication/      replication library
  tass_sample.py            sample construction (paper screens, funnel)
  sentiment.py              Baker-Wurgler sentiment-change constructions
  rolling.py                36-month rolling sentiment beta (no look-ahead)
  table3.py                 Table III decile sorts and 9-factor alpha
  table4.py                 Table IV Fama-MacBeth
  regression.py             Newey-West, Fama-MacBeth, decile assignment
  spec_runner.py, config.py, paths.py, utils.py, tass_clean.py
config/spec_grid.yaml     pre-registered admissible specification grid
results/                  report (PDF), results workbook, and figures
```

## Method (brief)

Each month, a fund's sentiment beta is the loading on the Baker-Wurgler sentiment-changes index in a
36-month rolling regression of excess returns on the sentiment-changes index and the Fung-Hsieh factors,
momentum, liquidity, inflation, and the default spread (paper equation 1). Funds are sorted into ten
equal-weighted deciles; portfolio alphas use the Fung-Hsieh seven factors, momentum, and liquidity
(excluding inflation and the default spread, per the paper's footnote 15), with Newey-West(2)
t-statistics. Term and credit factors are proxied from FRED because the paper's Barclays return series
are not publicly available.
