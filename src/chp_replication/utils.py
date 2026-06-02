from pathlib import Path
import hashlib
import pandas as pd
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: str | Path, chunk_size: int = 65536) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def month_period(s) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.to_period("M")


def month_end_timestamp(s) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp("M")


def safe_numeric(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def winsorize_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    lo, hi = x.quantile(lower), x.quantile(upper)
    return x.clip(lower=lo, upper=hi)


def write_excel_with_sheets(path: str | Path, sheets: dict[str, pd.DataFrame]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            sheet = str(name)[:31]
            out = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
            out.to_excel(writer, index=False, sheet_name=sheet)
            worksheet = writer.sheets[sheet]
            worksheet.freeze_panes(1, 0)
            for i, col in enumerate(out.columns):
                width = min(max(len(str(col)) + 2, 12), 45)
                worksheet.set_column(i, i, width)
