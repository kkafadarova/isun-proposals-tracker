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

# If blocked by anti-bot in CI, exit(0) instead of failing the job
SKIP_ON_BLOCK = os.getenv("SKIP_ON_BLOCK", "1") == "1"

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
    return url.replace("ExportToHtml", "ExportToExcel").replace("ExportToXml", "ExportToExcel")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).replace("\xa0", " ")).strip()


def parse_int(x):
    s = _clean_ws(x)
    if s == "" or s.lower() == "nan":
        return None
    s = re.sub(r"[^\d-]", "", s)
    return int(s) if s else None


def parse_float(x):
    s = _clean_ws(x)
    if s == "" or s.lower() == "nan":
        return None

    s = re.sub(r"[^\d\-,\.]", "", s)

    # "5,23" -> 5.23
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")

    # "1,234.56" -> 1234.56
    if "," in s and "." in s:
        s = s.replace(",", "")

    return float(s) if s else None


def looks_like_xlsx(content: bytes) -> bool:
    return content[:2] == b"PK"


def is_probably_protected_html(content: bytes, content_type: str | None) -> bool:
    if content_type and "text/html" in content_type.lower():
        text = content[:8000].decode("utf-8", errors="ignore").lower()
        markers = ["apm_do_not_touch", "tspd", "incapsula", "captcha", "cloudflare", "attention required"]
        return any(m in text for m in markers)

    head = content[:20].lstrip()
    if head.startswith(b"<") or head.startswith(b"<!DOCTYPE"):
        text = content[:8000].decode("utf-8", errors="ignore").lower()
        markers = ["apm_do_not_touch", "tspd", "incapsula", "captcha", "cloudflare", "attention required"]
        return any(m in text for m in markers)

    return False


def save_debug(name: str, data: bytes):
    path = os.path.join(DEBUG_DIR, name)
    with open(path, "wb") as f:
        f.write(data)


def save_debug_text(name: str, text: str):
    path = os.path.join(DEBUG_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def normalize_snapshot(snapshot: dict) -> dict:
    snap = dict(snapshot)

    for k in [COL_TOTAL, COL_BFP]:
        v = snap.get(k)
        if v is not None and not pd.isna(v):
            snap[k] = round(float(v), 2)

    for k in [COL_SUBMITTED, COL_APPROVED, COL_RESERVE, COL_REJECTED]:
        v = snap.get(k)
        if v is not None and not pd.isna(v):
            snap[k] = int(v)

    return snap


# -----------------------
# Fetch (requests + Playwright fallback)
# -----------------------
def requests_fetch(url: str, timeout: int = 60) -> bytes:
    session = requests.Session()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,"
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        ),
        "Accept-Language": "bg-BG,bg;q=0.9,en;q=0.8",
        "Referer": "https://2020.eufunds.bg/bg/0/0/ProjectProposals",
    }

    r = session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    ct = r.headers.get("content-type", "")
    content = r.content

    save_debug("last_response.bin", content)
    save_debug_text(
        "last_response_headers.txt",
        f"URL: {r.url}\nSTATUS: {r.status_code}\nCONTENT-TYPE: {ct}\n\nHEADERS:\n{dict(r.headers)}\n",
    )

    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code} from ISUN")

    if is_probably_protected_html(content, ct):
        save_debug("last_response.html", content)
        raise RuntimeError("ISUN returned a protected/anti-bot page (requests).")

    return content


def playwright_fetch(url: str) -> bytes:
    """
    Playwright fallback: navigate and inspect main response (no download event).
    """
    from playwright.sync_api import sync_playwright

    base_url = "https://2020.eufunds.bg/bg/0/0/ProjectProposals"
    headless = os.getenv("PW_HEADLESS", "1") != "0"
    slowmo = int(os.getenv("PW_SLOWMO", "0") or "0")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, slow_mo=slowmo)
        context = browser.new_context(
            locale="bg-BG",
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        page.goto("https://2020.eufunds.bg/", wait_until="domcontentloaded", timeout=120000)
        page.wait_for_timeout(1500)
        page.goto(base_url, wait_until="domcontentloaded", timeout=120000)
        page.wait_for_timeout(1500)

        resp = page.goto(url, wait_until="domcontentloaded", timeout=120000)
        if resp is None:
            html = page.content()
            save_debug_text("pw_no_response.html", html)
            raise RuntimeError("Playwright: no main response (likely blocked).")

        status = resp.status
        ct = (resp.headers.get("content-type") or "").lower()

        save_debug_text(
            "pw_export_headers.txt",
            f"STATUS: {status}\nCONTENT-TYPE: {ct}\nURL: {resp.url}\nHEADERS: {resp.headers}\n",
        )

        # If excel content-type, take bytes
        if "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in ct:
            body = resp.body()
            save_debug("pw_export.bin", body)
        else:
            html = page.content()
            save_debug_text("pw_export_page.html", html)
            try:
                body = resp.body()
                save_debug("pw_export_response_body.bin", body)
            except Exception:
                body = b""

            blob = body if body else html.encode("utf-8", errors="ignore")
            if is_probably_protected_html(blob, ct or "text/html"):
                raise RuntimeError("Playwright: ISUN served anti-bot HTML (no XLSX).")

            raise RuntimeError(f"Playwright: unexpected content-type '{ct}', status={status}")

        context.close()
        browser.close()

        if not looks_like_xlsx(body):
            raise RuntimeError("Playwright: response is not XLSX (missing PK header).")

        return body


def fetch_with_fallback(url: str) -> bytes:
    excel_url = to_excel_url(url)
    print(f"Using export URL: {excel_url}")

    try:
        return requests_fetch(excel_url)
    except Exception as e:
        print(f"[requests] failed: {e}")
        print("Falling back to Playwright navigation fetch…")
        return playwright_fetch(excel_url)


# -----------------------
# Parsing
# -----------------------
def read_table_from_export(content: bytes) -> pd.DataFrame:
    if not looks_like_xlsx(content):
        raise RuntimeError("Export response is not XLSX (missing PK header).")

    df = pd.read_excel(BytesIO(content), sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_target_row(df: pd.DataFrame) -> pd.Series:
    df_str = df.fillna("").astype(str)

    mask = df_str.apply(
        lambda row: any(
            (TARGET_PROCEDURE_CODE in cell) or (TARGET_PROCEDURE_NAME in cell)
            for cell in row
        ),
        axis=1,
    )

    hits = df.loc[mask]
    if hits.empty:
        raise RuntimeError("Target row not found in export.")

    return hits.iloc[0]


def extract_metrics_from_row(row: pd.Series) -> dict:
    cells = ["" if pd.isna(v) else str(v) for v in row.tolist()]

    code_idx = None
    for i, v in enumerate(cells):
        if TARGET_PROCEDURE_CODE in v:
            code_idx = i
            break

    if code_idx is None:
        for i, v in enumerate(cells):
            if TARGET_PROCEDURE_NAME in v:
                code_idx = max(0, i - 1)
                break

    if code_idx is None:
        raise RuntimeError("Could not locate procedure cell.")

    scan = cells[code_idx + 1 : code_idx + 31]

    numeric_tokens = []
    for v in scan:
        vv = _clean_ws(v)
        if re.search(r"\d", vv):
            f = parse_float(vv)
            if f is not None:
                numeric_tokens.append(vv)
        if len(numeric_tokens) >= 6:
            break

    if len(numeric_tokens) < 6:
        raise RuntimeError(f"Could not extract 6 metric cells. Got: {numeric_tokens}")

    return {
        COL_SUBMITTED: parse_int(numeric_tokens[0]),
        COL_TOTAL: parse_float(numeric_tokens[1]),
        COL_BFP: parse_float(numeric_tokens[2]),
        COL_APPROVED: parse_int(numeric_tokens[3]),
        COL_RESERVE: parse_int(numeric_tokens[4]),
        COL_REJECTED: parse_int(numeric_tokens[5]),
    }


# -----------------------
# CSV history logic: append only if changed
# -----------------------
def load_existing(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _num_equal(a, b) -> bool:
    if (pd.isna(a) or a is None) and (pd.isna(b) or b is None):
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
    for c in METRIC_COLS:
        if c not in existing.columns:
            return pd.concat([existing, new_df], ignore_index=True)
        if not _num_equal(last[c], new_row.get(c)):
            return pd.concat([existing, new_df], ignore_index=True)

    return existing


# -----------------------
# Main
# -----------------------
def main():
    print("Fetching export…")
    try:
        content = fetch_with_fallback(EXPORT_URL)
    except Exception as e:
        msg = str(e).lower()
        if SKIP_ON_BLOCK and (
            "anti-bot" in msg or "protected" in msg or "captcha" in msg or "text/html" in msg
        ):
            print(f"SKIP: blocked by anti-bot protection: {e}")
            print("No update performed. Exiting with code 0.")
            return
        raise

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
    snapshot = normalize_snapshot(snapshot)

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
        print(f"ERROR: {e}", file=sys.stderr)
        raise
