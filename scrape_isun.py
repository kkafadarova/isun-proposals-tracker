import os
import re
from datetime import datetime, timezone
from io import BytesIO
import pandas as pd
import requests


# Prefer Excel export (more machine-friendly than HTML)
EXPORT_URL = os.getenv(
    "EXPORT_URL",
    "https://2020.eufunds.bg/bg/0/0/ProjectProposals/ExportToExcel?ProgrammeId=yIyRFEzMEDyPTP0ZcYrk5g%3D%3D&ShowRes=True",
)

TARGET_PROCEDURE_CODE = os.getenv("TARGET_PROCEDURE_CODE", "BG16RFPR002-1.010")
TARGET_PROCEDURE_NAME = os.getenv(
    "TARGET_PROCEDURE_NAME",
    "Зелени и цифрови партньорства за интелигентна трансформация",
)

OUT_CSV = os.getenv("OUT_CSV", "data/isun_bg16rfpr002-1.010_history.csv")

# --- Final CSV headers (Bulgarian) ---
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
INT_COLS = [COL_SUBMITTED, COL_APPROVED, COL_RESERVE, COL_REJECTED]


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _clean_text(x) -> str:
    s = "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _num_str(x) -> str:
    # For comparisons, normalize numeric-like things to a canonical string.
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = _clean_text(x)
    # Keep digits, minus, comma, dot
    s = re.sub(r"[^\d\-,\.]", "", s)
    # If "5,23" -> "5.23"
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    # If "1,234.56" -> "1234.56"
    if "," in s and "." in s:
        s = s.replace(",", "")
    return s


def parse_int(x):
    s = _num_str(x)
    if not s:
        return None
    s = re.sub(r"[^\d-]", "", s)
    return int(s) if s else None


def parse_float(x):
    s = _num_str(x)
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def is_protected_page(content: bytes, content_type: str | None) -> bool:
    # Heuristics: anti-bot pages usually come as HTML and contain common markers
    ct = (content_type or "").lower()
    if "text/html" in ct or ct == "" or ct.startswith("text/"):
        head = content[:5000].decode("utf-8", errors="ignore").lower()
        markers = [
            "tspd",
            "apm_do_not_touch",
            "enable javascript",
            "please wait",
            "captcha",
            "checking your browser",
            "<html",
        ]
        return any(m in head for m in markers)
    return False


def fetch_bytes(url: str) -> tuple[bytes, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; github-actions; +https://github.com)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=90, allow_redirects=True)
    r.raise_for_status()
    return r.content, r.headers.get("Content-Type", "")


def fetch_with_debug(url: str) -> bytes:
    _ensure_dir("debug/x")
    content, ctype = fetch_bytes(url)

    if is_protected_page(content, ctype):
        _ensure_dir("debug/last_response.html")
        with open("debug/last_response.html", "wb") as f:
            f.write(content)
        raise RuntimeError(
            "ISUN returned a protected/anti-bot page. Saved to debug/last_response.html"
        )

    return content


from io import BytesIO, StringIO
import pandas as pd

def read_table_from_export(content: bytes) -> pd.DataFrame:
    """
    ISUN export sometimes returns:
      - Excel (.xlsx) => bytes start with b'PK\\x03\\x04'
      - HTML page => text with <html> and <table>
    This function detects and parses accordingly.
    """
    # 1) Excel (xlsx) signature
    if content[:4] == b"PK\x03\x04":
        xls = pd.read_excel(BytesIO(content))
        return xls

    # 2) Otherwise treat as HTML
    html = content.decode("utf-8", errors="replace")

    # quick sanity check
    if "<table" not in html.lower():
        # save for debugging
        os.makedirs("debug", exist_ok=True)
        with open("debug/last_non_table_response.html", "w", encoding="utf-8") as f:
            f.write(html)
        raise RuntimeError("HTML response has no <table>. Saved debug/last_non_table_response.html")

    tables = pd.read_html(StringIO(html))
    if not tables:
        raise RuntimeError("No tables found in HTML by pandas.")
    # return all tables concatenated OR pick the relevant one later
    return pd.concat(tables, ignore_index=True)


def find_target_row(df: pd.DataFrame) -> pd.Series:
    # robust search across all cells
    df_str = df.copy()
    df_str.columns = [str(c).strip() for c in df_str.columns]
    df_str = df_str.fillna("").astype(str)

    needle_code = TARGET_PROCEDURE_CODE
    needle_name = TARGET_PROCEDURE_NAME

    mask = df_str.apply(
        lambda row: any((needle_code in cell) or (needle_name in cell) for cell in row),
        axis=1,
    )
    hits = df.loc[mask]
    if hits.empty:
        raise RuntimeError(
            "Target procedure row not found in export.\n"
            f"Looked for code={needle_code} or name contains '{needle_name}'.\n"
            f"Columns: {list(df.columns)}"
        )
    return hits.iloc[0]


def extract_metrics_from_row(row: pd.Series) -> dict:
    """
    Expected row layout (as seen on the public page):
      code | name | submitted | total | bfp | approved | reserve | rejected

    We extract values by scanning cells after the code cell, grabbing 6 numeric tokens.
    """
    cells = ["" if pd.isna(v) else str(v) for v in row.tolist()]

    code_idx = None
    for i, v in enumerate(cells):
        if TARGET_PROCEDURE_CODE in v:
            code_idx = i
            break

    if code_idx is None:
        raise RuntimeError("Could not locate the procedure code cell in the target row.")

    scan = cells[code_idx + 1 : code_idx + 1 + 30]

    numeric_tokens: list[str] = []
    for v in scan:
        vv = _clean_text(v)
        if re.search(r"\d", vv):
            f = parse_float(vv)
            if f is not None:
                numeric_tokens.append(vv)
        if len(numeric_tokens) >= 6:
            break

    if len(numeric_tokens) < 6:
        raise RuntimeError(
            f"Could not extract 6 numeric cells from row. Extracted={numeric_tokens}"
        )

    submitted = parse_int(numeric_tokens[0])
    total_val = parse_float(numeric_tokens[1])
    bfp_val = parse_float(numeric_tokens[2])
    approved = parse_int(numeric_tokens[3])
    reserve = parse_int(numeric_tokens[4])
    rejected = parse_int(numeric_tokens[5])

    return {
        COL_SUBMITTED: submitted,
        COL_TOTAL: total_val,
        COL_BFP: bfp_val,
        COL_APPROVED: approved,
        COL_RESERVE: reserve,
        COL_REJECTED: rejected,
    }


def make_snapshot(row: pd.Series) -> dict:
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    metrics = extract_metrics_from_row(row)

    # enforce ints for count columns
    for c in INT_COLS:
        if metrics.get(c) is not None:
            metrics[c] = int(metrics[c])

    return {
        COL_TIMESTAMP: ts,
        COL_CODE: TARGET_PROCEDURE_CODE,
        COL_NAME: TARGET_PROCEDURE_NAME,
        **metrics,
    }


def load_existing(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _metric_equal(a, b) -> bool:
    # treat NaN/None as equal
    if (a is None or (isinstance(a, float) and pd.isna(a))) and (
        b is None or (isinstance(b, float) and pd.isna(b))
    ):
        return True

    # compare ints strictly when both are ints
    try:
        ia = int(float(a))
        ib = int(float(b))
        # if both "look like ints", use strict
        if abs(float(a) - ia) < 1e-9 and abs(float(b) - ib) < 1e-9:
            return ia == ib
    except Exception:
        pass

    # floats with tolerance
    try:
        fa = float(a)
        fb = float(b)
        return abs(fa - fb) < 1e-6
    except Exception:
        return _clean_text(a) == _clean_text(b)


def append_if_changed(existing: pd.DataFrame, snapshot: dict) -> pd.DataFrame:
    new_df = pd.DataFrame([snapshot])

    if existing is None or existing.empty:
        return new_df

    last = existing.iloc[-1]

    # If metrics are identical -> do not append
    for c in METRIC_COLS:
        if c not in existing.columns:
            # schema changed -> append
            return pd.concat([existing, new_df], ignore_index=True)
        if not _metric_equal(last[c], snapshot.get(c)):
            return pd.concat([existing, new_df], ignore_index=True)

    return existing


def main():
    print("Fetching export…")
    content = fetch_with_debug(EXPORT_URL)

    print("Parsing export…")
    df = read_table_from_export(content)

    print("Finding target procedure row…")
    row = find_target_row(df)

    snapshot = make_snapshot(row)
    print("Snapshot:", snapshot)

    _ensure_dir(OUT_CSV)
    existing = load_existing(OUT_CSV)
    updated = append_if_changed(existing, snapshot)

    if len(updated) == len(existing):
        print("No changes; nothing to append.")
        return

    updated.to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV} (rows: {len(existing)} -> {len(updated)})")


if __name__ == "__main__":
    main()
