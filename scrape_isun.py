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


def _clean(s: str) -> str:
    # NBSP + всички whitespace -> махаме
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

    # ако има само една запетая и няма точка -> приемаме запетаята за десетичен разделител
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")

    # ако има и "," и "." -> приемаме "." за десетична, махаме "," (хилядни)
    if "," in s and "." in s:
        s = s.replace(",", "")

    return float(s) if s else None


def fetch_bytes(url: str) -> bytes:
    # Реално endpoint-ът често връща HTML; Accept оставяме широк
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; github-actions; +https://github.com)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
    r.raise_for_status()
    return r.content


def parse_html_tables(content_bytes: bytes) -> list[pd.DataFrame]:
    html = content_bytes.decode("utf-8", errors="replace")
    # read_html върху literal string -> използваме StringIO
    return pd.read_html(StringIO(html))


def parse_export_to_table(content_bytes: bytes) -> pd.DataFrame:
    """
    Връща DataFrame с ред(ове), които съдържат таргет процедурата.
    Работи устойчиво при промени в header-и / различни таблици.
    """
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
    """
    Намира първия ред, в който има TARGET_PROCEDURE_CODE или TARGET_PROCEDURE_NAME.
    """
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
            "Не намерих таргет ред (търсене във всички клетки).\n"
            f"Колони: {list(df.columns)}\n"
            f"Търсих: {needle_code} / {needle_name}"
        )

    return hits.iloc[0]


def extract_metrics_from_row(row: pd.Series) -> dict:
    """
    Извлича 6-те стойности от реда без да разчита на имена на колони.
    Очакван ред в таблицата:
      код | име | submitted | total_value | bfp_value | approved | reserve | rejected
    """
    cells = ["" if pd.isna(v) else str(v) for v in row.tolist()]

    code_idx = None
    for i, v in enumerate(cells):
        if TARGET_PROCEDURE_CODE in v:
            code_idx = i
            break

    if code_idx is None:
        raise RuntimeError("Не намерих кода в таргет реда, за да извлека числата.")

    after = cells[code_idx + 1 :]
    scan = after[:25]  # буфер за всеки случай

    numeric = []
    for v in scan:
        vv = v.strip()
        # ако има цифра, пробваме да стане число
        if re.search(r"\d", vv):
            f = parse_float(vv)
            if f is not None:
                numeric.append(vv)
        if len(numeric) >= 6:
            break

    if len(numeric) < 6:
        raise RuntimeError(
            f"Не успях да извлека 6 числови клетки от реда. Намерих {len(numeric)}: {numeric}"
        )

    return {
        "submitted_count": parse_int(numeric[0]),
        "total_value_eur": parse_float(numeric[1]),
        "bfp_value_eur": parse_float(numeric[2]),
        "approved_count": parse_int(numeric[3]),
        "reserve_count": parse_int(numeric[4]),
        "rejected_count": parse_int(numeric[5]),
    }


def normalize_row(row: pd.Series) -> dict:
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    # код/име ги държим стабилни
    procedure_code = TARGET_PROCEDURE_CODE
    procedure_name = TARGET_PROCEDURE_NAME

    metrics = extract_metrics_from_row(row)

    return {
        "timestamp_utc": ts,
        "procedure_code": procedure_code,
        "procedure_name": procedure_name,
        **metrics,
    }


def load_existing(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def append_if_changed(existing: pd.DataFrame, new_row: dict) -> pd.DataFrame:
    new_df = pd.DataFrame([new_row])

    if existing is None or existing.empty:
        return new_df

    compare_cols = [c for c in new_df.columns if c != "timestamp_utc"]

    last = existing.iloc[-1]
    changed = False
    for c in compare_cols:
        if str(last.get(c, "")) != str(new_row.get(c, "")):
            changed = True
            break

    if not changed:
        return existing

    return pd.concat([existing, new_df], ignore_index=True)


def main():
    print("Fetching export…")
    content = fetch_bytes(EXPORT_XML_URL)

    print("Parsing tables…")
    table = parse_export_to_table(content)

    print("Finding target procedure row…")
    row = find_target_row(table)

    snapshot = normalize_row(row)
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
