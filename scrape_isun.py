import os
import re
from datetime import datetime, timezone
from io import StringIO

import pandas as pd
import requests


EXPORT_XML_URL = os.getenv(
    "EXPORT_XML_URL",
    "https://2020.eufunds.bg/bg/0/0/ProjectProposals/ExportToXml?ProgrammeId=yIyRFEzMEDyPTP0ZcYrk5g%3D%3D&ShowRes=True",
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


def _clean(s: str) -> str:
    return re.sub(r"\s+", "", str(s).replace("\xa0", " ")).strip()


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
    s = re.sub(r"[^\d\-,\.]", "", s)

    # "5,23" -> 5.23
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")

    # "1,234.56" -> 1234.56
    if "," in s and "." in s:
        s = s.replace(",", "")

    return float(s) if s else None


def fetch_bytes(url: str) -> bytes:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; github-actions; +https://github.com)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
    r.raise_for_status()
    return r.content


def parse_html_tables(content_bytes: bytes) -> list[pd.DataFrame]:
    html = content_bytes.decode("utf-8", errors="replace")
    return pd.read_html(StringIO(html))


def parse_export_to_table(content_bytes: bytes) -> pd.DataFrame:
    tables = parse_html_tables(content_bytes)

    needle_code = TARGET_PROCEDURE_CODE
    needle_name = TARGET_PROCEDURE_NAME

    hits = []
    for t in tables:
        tt = t.copy()
        tt.columns = [str(c).strip() for c in tt.columns]
        tt = tt.applymap(lambda x: "" if pd.isna(x) else str(x))

        mask = tt.apply(
            lambda row: any((needle_code in cell) or (needle_name in cell) for cell in row),
            axis=1,
        )

        if mask.any():
            hits.append(t[mask])

    if not hits:
        debug_cols = [list(map(str, t.columns)) for t in tables[:5]]
        raise RuntimeError(
            "Не намерих ред с таргет процедурата в HTML таблиците.\n"
            f"Търсих: {needle_code} / {needle_name}\n"
            f"Първите 5 таблици колони: {debug_cols}"
        )

    return pd.concat(hits, ignore_index=True)


def find_target_row(df: pd.DataFrame) -> pd.Series:
    needle_code = TARGET_PROCEDURE_CODE
    needle_name = TARGET_PROCEDURE_NAME

    df_str = df.copy()
    df_str.columns = [str(c).strip() for c in df_str.columns]
    df_str = df_str.applymap(lambda x: "" if pd.isna(x) else str(x))

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
    Очакван ред:
      код | име | submitted | total | bfp | approved | reserve | rejected
    Извличаме 6-те метрики устойчиво, без да разчитаме на header-и.
    """
    cells = ["" if pd.isna(v) else str(v) for v in row.tolist()]

    code_idx = None
    for i, v in enumerate(cells):
        if TARGET_PROCEDURE_CODE in v:
            code_idx = i
            break

    if code_idx is None:
        raise RuntimeError("Не намерих кода в таргет реда.")

    scan = cells[code_idx + 1 : code_idx + 1 + 25]

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


def load_existing(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _num_equal(a, b) -> bool:
    """
    Сравнява числа robust:
    - None/NaN се считат равни помежду си
    - int/float се сравняват като float с толеранс за float полетата
    """
    if pd.isna(a) and pd.isna(b):
        return True

    # ints may come as strings sometimes
    try:
        fa = float(a)
        fb = float(b)
    except Exception:
        return str(a) == str(b)

    return abs(fa - fb) < 1e-6


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


def main():
    print("Fetching export…")
    content = fetch_bytes(EXPORT_XML_URL)

    print("Parsing tables…")
    table = parse_export_to_table(content)

    print("Finding target procedure row…")
    row = find_target_row(table)

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
