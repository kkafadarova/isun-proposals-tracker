import os
import re
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


# ---- Config ----
EXPORT_URL = os.getenv(
    "EXPORT_URL",
    "https://2020.eufunds.bg/bg/0/0/ProjectProposals/ExportToHtml"
    "?ProgrammeId=yIyRFEzMEDyPTP0ZcYrk5g%3D%3D&ShowRes=True",
)

TARGET_PROCEDURE_CODE = os.getenv("TARGET_PROCEDURE_CODE", "BG16RFPR002-1.010")
TARGET_PROCEDURE_NAME = os.getenv(
    "TARGET_PROCEDURE_NAME",
    "Зелени и цифрови партньорства за интелигентна трансформация",
)

OUT_CSV = os.getenv("OUT_CSV", "data/isun_bg16rfpr002-1.010_history.csv")

DEBUG_DIR = Path(os.getenv("DEBUG_DIR", "debug"))
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


# --- Bulgarian column names (final CSV headers) ---
COL_TIMESTAMP = "timestamp_utc"
COL_CODE = "Номер на процедура"
COL_NAME = "Име на процедура"
COL_SUBMITTED = "Брой подадени проектни предложения"
COL_TOTAL = "Обща стойност на подадените проектни предложения"
COL_BFP = "Стойност на подадените проектни предложения БФП (в евро)"
COL_APPROVED = "Брой одобрени проектни предложения"
COL_RESERVE = "Брой проектни предложения в резервен списък"
COL_REJECTED = "Брой отхвърлени проектни предложения"

METRIC_COLS = [COL_SUBMITTED, COL_TOTAL, COL_BFP, COL_APPROVED, COL_RESERVE, COL_REJECTED]


# ---- Helpers ----
def _clean(s: str) -> str:
    # keep numbers readable for parsing
    return str(s).replace("\xa0", " ").strip()


def parse_int(x):
    s = _clean(x)
    if s == "" or s.lower() == "nan":
        return None
    s = re.sub(r"[^\d-]", "", s)
    return int(s) if s else None


def parse_float(x):
    s = _clean(x)
    if s == "" or s.lower() == "nan":
        return None

    # leave digits + separators only
    s = re.sub(r"[^\d\-,\.]", "", s)

    # "5,23" -> "5.23"
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")

    # "1,234.56" -> "1234.56"
    if "," in s and "." in s:
        s = s.replace(",", "")

    return float(s) if s else None


def fetch_bytes(url: str) -> bytes:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "bg-BG,bg;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    }
    r = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
    r.raise_for_status()

    content = r.content

    # quick “blocked / protected” detection
    head = content[:8000].decode("utf-8", errors="ignore").lower()
    if any(k in head for k in ["apm_do_not_touch", "tspd", "captcha", "please wait"]):
        # save for inspection
        (DEBUG_DIR / "last_response.html").write_bytes(content)
        raise RuntimeError(
            "ISUN returned a protected/anti-bot page (saved to debug/last_response.html)."
        )

    return content


def parse_html_tables(content_bytes: bytes) -> list[pd.DataFrame]:
    html = content_bytes.decode("utf-8", errors="replace")

    try:
        # IMPORTANT: force lxml (stable in CI)
        return pd.read_html(StringIO(html), flavor="lxml")
    except ValueError:
        # No tables found -> dump response to debug
        (DEBUG_DIR / "last_response.html").write_text(html, encoding="utf-8")
        raise RuntimeError(
            "No HTML tables found in export response. "
            "Saved debug/last_response.html (likely not the expected export page)."
        )


def find_procedure_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Find the table that contains rows with procedure code/name.
    We don't rely on headers (they may change).
    """
    needle_code = TARGET_PROCEDURE_CODE
    needle_name = TARGET_PROCEDURE_NAME

    for t in tables:
        tt = t.copy()
        tt.columns = [str(c).strip() for c in tt.columns]
        tt = tt.fillna("").astype(str)

        mask = tt.apply(
            lambda row: any((needle_code in cell) or (needle_name in cell) for cell in row),
            axis=1,
        )

        if mask.any():
            # return only matching rows
            return t.loc[mask].reset_index(drop=True)

    # debug: write column sets
    cols_preview = [list(map(str, t.columns)) for t in tables[:10]]
    raise RuntimeError(
        "Could not find any table containing the target procedure.\n"
        f"Looked for: {needle_code} / {needle_name}\n"
        f"First tables columns preview: {cols_preview}"
    )


def find_target_row(df: pd.DataFrame) -> pd.Series:
    """
    Pick the best row: exact code match if possible, else name contains.
    """
    df_str = df.fillna("").astype(str)

    # exact code match anywhere in row
    mask_code = df_str.apply(lambda r: any(TARGET_PROCEDURE_CODE == c.strip() for c in r), axis=1)
    hits = df.loc[mask_code]
    if not hits.empty:
        return hits.iloc[0]

    # name contains anywhere in row
    mask_name = df_str.apply(lambda r: any(TARGET_PROCEDURE_NAME in c for c in r), axis=1)
    hits = df.loc[mask_name]
    if not hits.empty:
        return hits.iloc[0]

    raise RuntimeError("Target row not found after table was already filtered (unexpected).")


def extract_metrics_from_row(row: pd.Series) -> dict:
    """
    Expected row format in HTML export table:
      [code] [name] [submitted] [total] [bfp] [approved] [reserve] [rejected]

    We find code position then take the NEXT 7 cells:
      name + 6 metrics
    This is more stable than scanning 'any digits' (because the name can contain digits).
    """
    cells = ["" if pd.isna(v) else str(v).strip() for v in row.tolist()]

    code_idx = None
    for i, v in enumerate(cells):
        if v.strip() == TARGET_PROCEDURE_CODE or TARGET_PROCEDURE_CODE in v:
            code_idx = i
            break
    if code_idx is None:
        raise RuntimeError("Could not locate procedure code in row.")

    # take: name + 6 metric cells
    tail = cells[code_idx + 1 : code_idx + 1 + 7]
    if len(tail) < 7:
        raise RuntimeError(f"Row too short after code. Got {len(tail)} cells: {tail}")

    name_cell = tail[0]
    m1, m2, m3, m4, m5, m6 = tail[1:7]

    submitted = parse_int(m1)
    total_val = parse_float(m2)
    bfp_val = parse_float(m3)
    approved = parse_int(m4)
    reserve = parse_int(m5)
    rejected = parse_int(m6)

    return {
        COL_NAME: name_cell if name_cell else TARGET_PROCEDURE_NAME,
        COL_SUBMITTED: int(submitted) if submitted is not None else None,
        COL_TOTAL: float(total_val) if total_val is not None else None,
        COL_BFP: float(bfp_val) if bfp_val is not None else None,
        COL_APPROVED: int(approved) if approved is not None else None,
        COL_RESERVE: int(reserve) if reserve is not None else None,
        COL_REJECTED: int(rejected) if rejected is not None else None,
    }


def make_snapshot(row: pd.Series) -> dict:
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    metrics = extract_metrics_from_row(row)

    return {
        COL_TIMESTAMP: ts,
        COL_CODE: TARGET_PROCEDURE_CODE,
        COL_NAME: metrics[COL_NAME],
        COL_SUBMITTED: metrics[COL_SUBMITTED],
        COL_TOTAL: metrics[COL_TOTAL],
        COL_BFP: metrics[COL_BFP],
        COL_APPROVED: metrics[COL_APPROVED],
        COL_RESERVE: metrics[COL_RESERVE],
        COL_REJECTED: metrics[COL_REJECTED],
    }


def load_existing(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=[COL_TIMESTAMP, COL_CODE, COL_NAME] + METRIC_COLS)
    return pd.read_csv(path)


def _num_equal(a, b) -> bool:
    # NaN/None considered equal
    if (pd.isna(a) and pd.isna(b)) or (a is None and b is None):
        return True
    try:
        return abs(float(a) - float(b)) < 1e-6
    except Exception:
        return str(a) == str(b)


def append_if_changed(existing: pd.DataFrame, new_row: dict) -> pd.DataFrame:
    new_df = pd.DataFrame([new_row])

    if existing is None or existing.empty:
        return new_df

    last = existing.iloc[-1]

    # Only compare the 6 metrics (NOT timestamp/name/code)
    for c in METRIC_COLS:
        if c not in existing.columns:
            return pd.concat([existing, new_df], ignore_index=True)
        if not _num_equal(last[c], new_row.get(c)):
            return pd.concat([existing, new_df], ignore_index=True)

    # identical metrics => do not append
    return existing


def main():
    print("Fetching export…")
    content = fetch_bytes(EXPORT_URL)

    print("Parsing tables…")
    tables = parse_html_tables(content)

    print("Locating procedure table…")
    filtered = find_procedure_table(tables)

    print("Finding target row…")
    row = find_target_row(filtered)

    snapshot = make_snapshot(row)
    print("Snapshot:", snapshot)

    out_path = Path(OUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_existing(str(out_path))
    updated = append_if_changed(existing, snapshot)

    if len(updated) == len(existing):
        print("No changes; nothing to append.")
        return

    updated.to_csv(out_path, index=False)
    print(f"Saved -> {OUT_CSV} (rows: {len(existing)} -> {len(updated)})")


if __name__ == "__main__":
    main()
