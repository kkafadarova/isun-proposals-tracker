import os
import re
import sys
from datetime import datetime, timezone
from io import BytesIO

import pandas as pd
import requests


# -----------------------
# Config
# -----------------------
EXPORT_URL = os.getenv(
    "EXPORT_URL",
    # You can pass ExportToHtml / ExportToXml here; script will prefer ExportToExcel anyway
    "https://2020.eufunds.bg/bg/0/0/ProjectProposals/ExportToExcel?ProgrammeId=yIyRFEzMEDyPTP0ZcYrk5g%3D%3D&ShowRes=True",
)

TARGET_PROCEDURE_CODE = os.getenv("TARGET_PROCEDURE_CODE", "BG16RFPR002-1.010")
TARGET_PROCEDURE_NAME = os.getenv(
    "TARGET_PROCEDURE_NAME",
    "Зелени и цифрови партньорства за интелигентна трансформация",
)

OUT_CSV = os.getenv("OUT_CSV", "data/isun_bg16rfpr002-1.010_history.csv")

DEBUG_DIR = "debug"
os.makedirs(DEBUG_DIR, exist_ok=True)


# -----------------------
# Final CSV headers (Bulgarian)
# -----------------------
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


# -----------------------
# Utils
# -----------------------
def to_excel_url(url: str) -> str:
    # If user passes ExportToHtml/Xml, swap to ExportToExcel
    url = url.replace("ExportToHtml", "ExportToExcel").replace("ExportToXml", "ExportToExcel")
    return url


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).replace("\xa0", " ")).strip()


def parse_int(x):
    s = _clean_ws(x)
    if s == "" or s.lower() == "nan":
        return None
    # remove everything except digits and minus
    s = re.sub(r"[^\d-]", "", s)
    if s == "":
        return None
    return int(s)


def parse_float(x):
    s = _clean_ws(x)
    if s == "" or s.lower() == "nan":
        return None

    # keep digits, comma, dot, minus
    s = re.sub(r"[^\d\-,\.]", "", s)

    # Bulgarian decimal "5,23" -> "5.23"
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")

    # thousands separator "1,234.56" -> "1234.56"
    if "," in s and "." in s:
        s = s.replace(",", "")

    if s == "" or s == "-" or s == ".":
        return None
    return float(s)


def is_probably_protected_html(content: bytes, content_type: str | None) -> bool:
    # common cases:
    # - content-type text/html
    # - HTML includes "APM_DO_NOT_TOUCH", "TSPD", "Incapsula", etc.
    if content_type and "text/html" in content_type.lower():
        text = content[:5000].decode("utf-8", errors="ignore")
        markers = ["APM_DO_NOT_TOUCH", "TSPD", "Incapsula", "cloudflare", "captcha", "<html"]
        if any(m.lower() in text.lower() for m in markers):
            return True

    # if it looks like HTML anyway
    head = content[:20].lstrip()
    if head.startswith(b"<") or head.startswith(b"<!DOCTYPE"):
        text = content[:5000].decode("utf-8", errors="ignore")
        if "APM_DO_NOT_TOUCH" in text or "TSPD" in text or "captcha" in text.lower():
            return True

    return False


def looks_like_xlsx(content: bytes) -> bool:
    # xlsx is a zip: starts with PK\x03\x04
    return content[:2] == b"PK"


def requests_fetch(url: str, timeout: int = 60) -> bytes:
    session = requests.Session()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;q=0.9,"
            "*/*;q=0.8"
        ),
        "Accept-Language": "bg-BG,bg;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Referer": "https://2020.eufunds.bg/bg/0/0/ProjectProposals",
    }

    r = session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    ct = r.headers.get("content-type", "")
    content = r.content

    # save last response for debug always
    with open(os.path.join(DEBUG_DIR, "last_response.bin"), "wb") as f:
        f.write(content)
    with open(os.path.join(DEBUG_DIR, "last_response_headers.txt"), "w", encoding="utf-8") as f:
        f.write(f"URL: {r.url}\nSTATUS: {r.status_code}\nCONTENT-TYPE: {ct}\n\nHEADERS:\n{dict(r.headers)}\n")

    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} from ISUN")

    if is_probably_protected_html(content, ct):
        with open(os.path.join(DEBUG_DIR, "last_response.html"), "wb") as f:
            f.write(content)
        raise RuntimeError("ISUN returned a protected/anti-bot page.")

    return content


def playwright_download(url: str) -> bytes:
    """
    Uses Playwright to handle cases where requests gets blocked.
    Works well for ExportToExcel (download).
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            locale="bg-BG",
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        # Expect a download
        with page.expect_download(timeout=120000) as dl_info:
            page.goto(url, wait_until="domcontentloaded", timeout=120000)
        download = dl_info.value
        path = download.path()
        if not path:
            # sometimes path is None; save via download.save_as
            save_to = os.path.join(DEBUG_DIR, "last_download.xlsx")
            download.save_as(save_to)
            with open(save_to, "rb") as f:
                data = f.read()
        else:
            with open(path, "rb") as f:
                data = f.read()

        # Save for debug
        with open(os.path.join(DEBUG_DIR, "last_playwright_download.bin"), "wb") as f:
            f.write(data)

        browser.close()
        return data


def fetch_with_fallback(url: str) -> bytes:
    """
    Prefer Excel export. If requests gets blocked, fallback to Playwright download.
    """
    excel_url = to_excel_url(url)
    print(f"Using export URL: {excel_url}")

    # Try requests first
    try:
        content = requests_fetch(excel_url)
        return content
    except Exception as e:
        print(f"[requests] failed: {e}")
        print("Falling back to Playwright…")
        content = playwright_download(excel_url)

        # If still HTML (blocked), save and fail
        if is_probably_protected_html(content, "text/html"):
            with open(os.path.join(DEBUG_DIR, "last_playwright.html"), "wb") as f:
                f.write(content)
            raise RuntimeError("Playwright also got a protected/anti-bot HTML page.")

        return content


# -----------------------
# Parsing
# -----------------------
def read_table_from_export(content: bytes) -> pd.DataFrame:
    """
    ISUN ExportToExcel returns .xlsx (zip). Parse it.
    """
    if not looks_like_xlsx(content):
        # if it is HTML, save it and raise
        with open(os.path.join(DEBUG_DIR, "last_non_excel_response.html"), "wb") as f:
            f.write(content)
        raise RuntimeError("Export response is not XLSX (does not start with PK).")

    # Most exports have a single sheet with the table. Read first sheet.
    xls = pd.read_excel(BytesIO(content), sheet_name=0)
    # Normalize columns to strings
    xls.columns = [str(c).strip() for c in xls.columns]
    return xls


def find_target_row(df: pd.DataFrame) -> pd.Series:
    """
    Robust search: find first row where ANY cell contains the procedure code or procedure name.
    """
    df_str = df.fillna("").astype(str)
    needle_code = TARGET_PROCEDURE_CODE
    needle_name = TARGET_PROCEDURE_NAME

    mask = df_str.apply(
        lambda row: any((needle_code in cell) or (needle_name in cell) for cell in row),
        axis=1,
    )
    hits = df.loc[mask]
    if hits.empty:
        raise RuntimeError(
            f"Target row not found.\nColumns: {list(df.columns)}\n"
            f"Searched: {needle_code} / {needle_name}"
        )
    return hits.iloc[0]


def extract_metrics_from_row(row: pd.Series) -> dict:
    """
    Expected row format (from Excel export):
      Procedure code | name | submitted | total | bfp | approved | reserve | rejected
    We extract the 6 metrics robustly by scanning numeric cells after the code.
    """
    cells = ["" if pd.isna(v) else str(v) for v in row.tolist()]

    code_idx = None
    for i, v in enumerate(cells):
        if TARGET_PROCEDURE_CODE in v:
            code_idx = i
            break
    if code_idx is None:
        # Sometimes code is in different cell formatting, fallback name search
        for i, v in enumerate(cells):
            if TARGET_PROCEDURE_NAME in v:
                code_idx = max(0, i - 1)  # heuristic
                break
    if code_idx is None:
        raise RuntimeError("Could not locate code/name cell inside target row to extract numbers.")

    scan = cells[code_idx + 1 : code_idx + 1 + 30]

    numeric_tokens = []
    for v in scan:
        vv = _clean_ws(v)
        if re.search(r"\d", vv):
            # accept both ints and floats
            f = parse_float(vv)
            if f is not None:
                numeric_tokens.append(vv)
        if len(numeric_tokens) >= 6:
            break

    if len(numeric_tokens) < 6:
        raise RuntimeError(f"Could not extract 6 metric cells. Got: {numeric_tokens}")

    submitted = parse_int(numeric_tokens[0])
    total_val = parse_float(numeric_tokens[1])
    bfp_val = parse_float(numeric_tokens[2])
    approved = parse_int(numeric_tokens[3])
    reserve = parse_int(numeric_tokens[4])
    rejected = parse_int(numeric_tokens[5])

    # Ensure counts are int (not decimals)
    return {
        COL_SUBMITTED: int(submitted) if submitted is not None else None,
        COL_TOTAL: float(total_val) if total_val is not None else None,
        COL_BFP: float(bfp_val) if bfp_val is not None else None,
        COL_APPROVED: int(approved) if approved is not None else None,
        COL_RESERVE: int(reserve) if reserve is not None else None,
        COL_REJECTED: int(rejected) if rejected is not None else None,
    }


def load_existing(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _num_equal(a, b) -> bool:
    """
    Compare numbers robustly:
    - NaN/None treated as equal
    - floats with tolerance
    - ints exact
    """
    if (pd.isna(a) or a is None) and (pd.isna(b) or b is None):
        return True
    try:
        fa = float(a)
        fb = float(b)
        return abs(fa - fb) < 1e-6
    except Exception:
        return str(a) == str(b)


def append_if_changed(existing: pd.DataFrame, new_row: dict) -> pd.DataFrame:
    new_df = pd.DataFrame([new_row])

    if existing is None or existing.empty:
        return new_df

    # If columns mismatch, append (but keep going)
    for c in [COL_TIMESTAMP, COL_CODE, COL_NAME] + METRIC_COLS:
        if c not in existing.columns:
            return pd.concat([existing, new_df], ignore_index=True)

    last = existing.iloc[-1]

    # Compare only the 6 metric columns (NOT timestamp)
    for c in METRIC_COLS:
        if not _num_equal(last[c], new_row.get(c)):
            return pd.concat([existing, new_df], ignore_index=True)

    # No changes -> no append
    return existing


def main():
    print("Fetching export…")
    content = fetch_with_fallback(EXPORT_URL)

    print("Parsing export…")
    df = read_table_from_export(content)

    print("Finding target procedure row…")
    row = find_target_row(df)

    metrics = extract_metrics_from_row(row)

    snapshot = {
        COL_TIMESTAMP: now_utc_iso(),
        COL_CODE: TARGET_PROCEDURE_CODE,
        COL_NAME: TARGET_PROCEDURE_NAME,
        **metrics,
    }

    print("Snapshot:", snapshot)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    existing = load_existing(OUT_CSV)
    updated = append_if_changed(existing, snapshot)

    if len(updated) == len(existing):
        print("No changes; nothing to append.")
        return

    updated.to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV} (rows: {len(existing)} -> {len(updated)})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Always leave a clear error
        print(f"ERROR: {e}", file=sys.stderr)
        raise
