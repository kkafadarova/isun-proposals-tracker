import os
import re
import time
from datetime import datetime, timezone
from io import StringIO

import pandas as pd
import requests


EXPORT_URL = os.getenv(
    "EXPORT_URL",
    "https://2020.eufunds.bg/bg/0/0/ProjectProposals/ExportToHtml?ProgrammeId=yIyRFEzMEDyPTP0ZcYrk5g%3D%3D&ShowRes=True",
)

TARGET_PROCEDURE_CODE = os.getenv("TARGET_PROCEDURE_CODE", "BG16RFPR002-1.010")
TARGET_PROCEDURE_NAME = os.getenv(
    "TARGET_PROCEDURE_NAME",
    "Зелени и цифрови партньорства за интелигентна трансформация",
)

OUT_CSV = os.getenv("OUT_CSV", "data/isun_bg16rfpr002-1.010_history.csv")


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


# ---------- helpers ----------
def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).replace("\xa0", " ")).strip()


def _clean_number_str(x) -> str:
    s = _collapse_ws(x)
    s = s.replace(" ", "")
    return s


def parse_int(x):
    s = _clean_number_str(x)
    if s == "":
        return None
    s = re.sub(r"[^\d-]", "", s)
    return int(s) if s else None


def parse_float(x):
    s = _clean_number_str(x)
    if s == "":
        return None
    s = re.sub(r"[^\d\-,\.]", "", s)

    # "5,23" -> 5.23
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")

    # "1,234.56" -> 1234.56
    if "," in s and "." in s:
        s = s.replace(",", "")

    return float(s) if s else None


def is_protected_page(html: str) -> bool:
    h = html.lower()
    # crude but practical signals for anti-bot / challenge
    return any(
        token in h
        for token in [
            "apm_do_not_touch",
            "/tspd/",
            "captcha",
            "challenge",
            "access denied",
            "forbidden",
        ]
    )


# ---------- fetching ----------
def fetch_bytes_requests(url: str) -> bytes:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "bg-BG,bg;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    }
    r = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
    r.raise_for_status()

    html = r.text
    if is_protected_page(html):
        os.makedirs("debug", exist_ok=True)
        with open("debug/last_response_requests.html", "w", encoding="utf-8") as f:
            f.write(html[:500000])
        raise RuntimeError("ISUN returned a protected/anti-bot page (requests).")

    return r.content


def fetch_bytes_playwright(url: str) -> bytes:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="bg-BG",
            viewport={"width": 1280, "height": 800},
        )
        page = context.new_page()

        # warm-up (cookies / potential JS checks)
        page.goto("https://2020.eufunds.bg/bg/0/0", wait_until="domcontentloaded", timeout=90000)
        page.wait_for_timeout(1500)

        resp = page.goto(url, wait_until="networkidle", timeout=90000)
        if resp is None:
            browser.close()
            raise RuntimeError("Playwright: no response from goto().")

        # IMPORTANT: take response body (not DOM)
        html = resp.text()

        browser.close()

    return html.encode("utf-8", errors="ignore")


def fetch_bytes_with_fallback(url: str, attempts: int = 3) -> bytes:
    last_err = None
    for i in range(1, attempts + 1):
        try:
            return fetch_bytes_requests(url)
        except Exception as e:
            last_err = e
            sleep_s = 2 * i
            print(f"[requests attempt {i}/{attempts}] failed: {e} -> sleeping {sleep_s}s")
            time.sleep(sleep_s)

    print("Protected page detected (requests). Falling back to Playwright…")
    return fetch_bytes_playwright(url)


# ---------- parsing ----------
def parse_html_tables(content_bytes: bytes) -> list[pd.DataFrame]:
    html = content_bytes.decode("utf-8", errors="replace")

    if "<table" not in html.lower():
        os.makedirs("debug", exist_ok=True)
        with open("debug/last_playwright.html", "w", encoding="utf-8") as f:
            f.write(html[:500000])
        raise RuntimeError("No <table> in HTML (saved debug/last_playwright.html).")

    # pandas may need bs4/html5lib depending on version
    return pd.read_html(StringIO(html))


def find_procedure_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """
    ISUN page has multiple tables; we want the one containing header like 'Номер на процедура'
    or at least containing our procedure code/name.
    """
    needle_code = TARGET_PROCEDURE_CODE
    needle_name = TARGET_PROCEDURE_NAME

    # Prefer table which has "Номер на процедура" header
    for t in tables:
        cols = [str(c).strip() for c in t.columns]
        if any("номер" in c.lower() and "процед" in c.lower() for c in cols):
            return t

    # Otherwise, pick first table that contains our target code/name anywhere
    for t in tables:
        tt = t.fillna("").astype(str)
        mask = tt.apply(
            lambda row: any((needle_code in cell) or (needle_name in cell) for cell in row),
            axis=1,
        )
        if mask.any():
            return t

    raise RuntimeError("Could not find a relevant procedure table in parsed tables.")


def find_target_row(df: pd.DataFrame) -> pd.Series:
    needle_code = TARGET_PROCEDURE_CODE
    needle_name = TARGET_PROCEDURE_NAME

    df_str = df.copy()
    df_str.columns = [str(c).strip() for c in df_str.columns]
    df_str = df_str.fillna("").astype(str)

    mask = df_str.apply(
        lambda row: any((needle_code in cell) or (needle_name in cell) for cell in row),
        axis=1,
    )

    hits = df.loc[mask]
    if hits.empty:
        raise RuntimeError(
            "Не намерих таргет ред.\n"
            f"Колони: {list(df.columns)}\n"
            f"Търсих: {needle_code} / {needle_name}"
        )

    return hits.iloc[0]


def extract_metrics_from_row(row: pd.Series) -> dict:
    """
    Expected row includes:
      code | name | submitted | total | bfp | approved | reserve | rejected
    We do not rely on exact column names. We locate procedure code cell,
    then scan forward and take first 6 numeric-looking cells.
    """
    cells = ["" if pd.isna(v) else str(v) for v in row.tolist()]

    code_idx = None
    for i, v in enumerate(cells):
        if TARGET_PROCEDURE_CODE in v:
            code_idx = i
            break

    if code_idx is None:
        # fallback: if code not present but name is present
        for i, v in enumerate(cells):
            if TARGET_PROCEDURE_NAME in v:
                code_idx = i
                break

    if code_idx is None:
        raise RuntimeError("Не намерих нито кода, нито името в таргет реда.")

    scan = cells[code_idx + 1 : code_idx + 1 + 30]

    numeric_tokens = []
    for v in scan:
        vv = v.strip()
        if re.search(r"\d", vv):
            f = parse_float(vv)
            if f is not None:
                numeric_tokens.append(vv)
        if len(numeric_tokens) >= 6:
            break

    if len(numeric_tokens) < 6:
        raise RuntimeError(f"Не успях да извлека 6 числови клетки. Намерих: {numeric_tokens}")

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


# ---------- storage ----------
def load_existing(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _num_equal(a, b) -> bool:
    if pd.isna(a) and (b is None or pd.isna(b)):
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

    last = existing.iloc[-1]

    # If columns missing (first run after schema change), append
    for c in METRIC_COLS:
        if c not in existing.columns:
            return pd.concat([existing, new_df], ignore_index=True)

    # Append only if at least one metric differs
    for c in METRIC_COLS:
        if not _num_equal(last[c], new_row.get(c)):
            return pd.concat([existing, new_df], ignore_index=True)

    return existing


def main():
    print("Fetching export…")
    content = fetch_bytes_with_fallback(EXPORT_URL, attempts=3)

    print("Parsing tables…")
    tables = parse_html_tables(content)

    table = find_procedure_table(tables)

    print("Finding target procedure row…")
    row = find_target_row(table)

    snapshot = make_snapshot(row)
    print("Snapshot:", snapshot)

    out_dir = os.path.dirname(OUT_CSV) or "."
    os.makedirs(out_dir, exist_ok=True)

    existing = load_existing(OUT_CSV)
    updated = append_if_changed(existing, snapshot)

    if existing is not None and len(updated) == len(existing):
        print("No changes; nothing to append.")
        return

    updated.to_csv(OUT_CSV, index=False)
    prev_len = 0 if existing is None else len(existing)
    print(f"Saved -> {OUT_CSV} (rows: {prev_len} -> {len(updated)})")


if __name__ == "__main__":
    main()
