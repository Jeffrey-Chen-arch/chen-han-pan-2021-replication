"""TASS field standardizers (unit inference, strategy/redemption parsing).

Units are inferred from the data (not assumed from SampleData). Each function returns the
converted series AND a decision string for the audit trail.
"""
import numpy as np
import pandas as pd


def standardize_return_to_percent(s: pd.Series) -> tuple[pd.Series, str]:
    x = pd.to_numeric(s, errors="coerce")
    q99 = x.abs().quantile(0.99)
    if pd.notna(q99) and q99 <= 1.0:
        return x * 100.0, "decimal_to_percent"
    return x, "already_percent"


def standardize_aum_to_million(s: pd.Series) -> tuple[pd.Series, str]:
    x = pd.to_numeric(s, errors="coerce")
    med = x.dropna().median()
    if pd.isna(med):
        return x, "unknown_all_missing"
    if med > 1_000_000:
        return x / 1_000_000.0, "dollars_to_millions"
    if med > 1_000:
        return x / 1_000.0, "thousands_to_millions"
    return x, "already_millions"


def standardize_fee_to_percent(s: pd.Series) -> tuple[pd.Series, str]:
    x = pd.to_numeric(s, errors="coerce")
    q99 = x.abs().quantile(0.99)
    if pd.notna(q99) and q99 <= 1.0:
        return x * 100.0, "decimal_to_percent"
    return x, "already_percent"


def normalize_strategy(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip().lower().replace("_", " ").replace("-", " ")


def parse_redemption_frequency_to_months(x):
    """Convert a redemption-frequency label/number to month-equivalent (smaller = more liquid)."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if "daily" in s:
        return 1 / 21
    if "weekly" in s and "bi" not in s:
        return 1 / 4.33
    if "biweekly" in s or "bi-weekly" in s or "fortnight" in s:
        return 0.5
    if "semi-month" in s or "semimonth" in s:
        return 0.5
    if "monthly" in s or s == "month":
        return 1.0
    if "quarter" in s:
        return 3.0
    if "semiannual" in s or "semi-annual" in s or "semi annual" in s:
        return 6.0
    if "annual" in s or "year" in s:
        return 12.0
    try:
        val = float(s)
        return val / 30.0 if val > 31 else val  # days -> months heuristic
    except Exception:
        return np.nan


def yes_no_to_dummy(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ("1", "yes", "y", "true", "t"):
        return 1.0
    if s in ("0", "no", "n", "false", "f"):
        return 0.0
    try:
        return 1.0 if float(s) != 0 else 0.0
    except Exception:
        return np.nan
