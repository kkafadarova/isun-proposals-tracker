import os
import re
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


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


# --- Final CSV column names (Bulgarian headers) ---
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


ANTI_BOT_MARKERS = [
    "/TSPD/",
    "APM_DO_NOT_TOUCH",
    "TSPD/08997",
    "window.gzwW",
    "distil_r_captcha",  # sometimes
]


def is_protected_page(html: str) -> bool:
    h = html or ""
    return any(m in h for m in ANTI_BOT_MARKERS)


def _clean(s: str) -> str:
    # normalize whitespace + non-breaking spaces
    return re.sub(r"\s+", " ", str(s).replace("\xa0", " ")).strip()


def parse_int(x):
    s = _clean(x)
    if s == "":
        return None
    s = re.sub(r"[^\d-]", "", s)
    return int(s) if s else None


def parse_float(x):
    s = _clean(x)
    if s == "":
        return None

    # keep digits, comma, dot, minus
    s = re.sub(r"[^\d\-,\.]", "", s)

    # "5,23" -> "5.23"
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")

    # "1,234.56" -> "1234.56"
    if "," in s and "." in s:
        s = s.replace(",", "")

    return float(s) if s else None


def fetch_with_requests(url: str, timeout: int = 60) -> bytes:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "bg-BG,bg;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.content


def fetch_with_playwright(url: str, timeout_ms: int = 60_000) -> bytes:
    """
    Headless browser fallback to bypass JS anti-bot pages.
    Requires: playwright + chromium installed in CI.
    """
    from playwright.sync_api import sync_playwright  # lazy import

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            locale="bg-BG",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

        # some anti-bot scripts set cookies after a short delay
        page.wait_for_timeout(3000)

        # wait for at least one table to exist (best effort)
        try:
            page.wait_for_selector("table", timeout=10_000)
        except Exception:
            pass

        html = page.content()
        browser.close()
        return html.encode("utf-8", errors="replace")


def fetch_export(url: str) -> bytes:
    # 1) try requests
    content = fetch_with_requests(url)
    html = content.decode("utf-8", errors="replace")

    if is_protected_page(html):
        (DEBUG_DIR / "last_response_requests.html").write_text(html, encoding="utf-8")
        print("Protected page detected (requests). Falling back to Playwright…")
        content = fetch_with_playwright(url)
        html2 = content.decode("utf-8", errors="replace")
        (DEBUG_DIR / "last_response_playwright.html").write_text(html2, encoding="utf-8")

        if is_protected_page(html2):
            raise RuntimeError(
                "ISUN returned a protected/anti-bot page even via Playwright. "
                "See debug/last_response_playwright.html"
            )
        return content

    return content


def parse_html_tables(content_bytes: bytes) -> list[pd.DataFrame]:
    html = content_bytes.decode("utf-8", errors="replace")
    if is_protected_page(html):
        raise RuntimeError("Protected/anti-bot HTML (cannot parse tables).")
    # pandas needs html5lib/lxml
    return pd.read_html(StringIO(html))


def find_row_in_tables(tables: list[pd.DataFrame]) -> pd.Series:
    needle_code = TARGET_PROCEDURE_CODE
    needle_name = TARGET_PROCEDURE_NAME

    for t in tables:
        tt = t.fillna("").astype(str)
        mask = tt.apply(
            lambda row: any((needle_code in cell) or (needle_name in cell) for cell in row),
            axis=1,
        )
        if mask.any():
            return t.loc[mask].iloc[0]

    raise RuntimeError("Target procedure row not found in any parsed table.")


def extract_metrics_from_row(row: pd.Series) -> dict:
    """
    The export row contains:
      code | name | submitted | total | bfp | approved | reserve | rejected
    But headers may vary, so we scan after the code cell and take the next 6 numeric cells.
    """
    cells = ["" if pd.isna(v) else str(v) for v in row.tolist()]

    code_idx = None
    for i, v in enumerate(cells):
        if TARGET_PROCEDURE_CODE in v:
            code_idx = i
            break

    if code_idx is None:
        raise RuntimeError("Could not locate procedure code inside the target row.")

    scan = cells[code_idx + 1 : code_idx + 1 + 30]

    numeric_tokens = []
    for v in scan:
        vv = _clean(v)
        if re.search(r"\d", vv):
            f = parse_float(vv)
            if f is not None:
                numeric_tokens.append(vv)
        if len(numeric_tokens) >= 6:
            break

    if len(numeric_tokens) < 6:
        raise RuntimeError(f"Could not extract 6 numeric cells. Got: {numeric_tokens}")

    submitted = parse_int(numeric_tokens[0])
    total_val = parse_float(numeric_tokens[1])
    bfp_val = parse_float(numeric_tokens[2])
    approved = parse_int(numeric_tokens[3])
    reserve = parse_int(numeric_tokens[4])
    rejected = parse_int(numeric_tokens[5])

    return {
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
        COL_NAME: TARGET_PROCEDURE_NAME,
        **metrics,
    }


def load_existing(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=[COL_TIMESTAMP, COL_CODE, COL_NAME, *METRIC_COLS])
    return pd.read_csv(path)


def _num_equal(a, b) -> bool:
    # treat None/NaN as equal
    if (a is None or (isinstance(a, float) and pd.isna(a))) and (b is None or (isinstance(b, float) and pd.isna(b))):
        return True
    if pd.isna(a) and pd.isna(b):
        return True
    try:
        fa = float(a)
        fb = float(b)
        return abs(fa - fb) < 1e-6
    except Exception:
        return str(a) == str(b)


def append_if_changed(existing: pd.DataFrame, new_row: dict) -> pd.DataFrame:
    """
    Requirement:
    1) DO NOT append if the last row has the same metric values.
    """
    new_df = pd.DataFrame([new_row])

    if existing is None or existing.empty:
        return new_df

    last = existing.iloc[-1]

    # if columns differ, be safe and append
    for c in METRIC_COLS:
        if c not in existing.columns:
            return pd.concat([existing, new_df], ignore_index=True)

    # compare ONLY metric cols (not timestamp)
    changed = any(not _num_equal(last[c], new_row.get(c)) for c in METRIC_COLS)
    if not changed:
        return existing

    return pd.concat([existing, new_df], ignore_index=True)


def main():
    print("Fetching export…")
    content = fetch_export(EXPORT_URL)

    print("Parsing tables…")
    tables = parse_html_tables(content)
    if not tables:
        raise RuntimeError("No tables found after fetch (unexpected).")

    print("Finding target procedure row…")
    row = find_row_in_tables(tables)

    snapshot = make_snapshot(row)
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
    main()
