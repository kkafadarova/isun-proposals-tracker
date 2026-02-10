import os
from datetime import datetime, timezone
import pandas as pd
import requests


EXPORT_XML_URL = os.getenv(
    "EXPORT_XML_URL",
    "https://2020.eufunds.bg/bg/0/0/ProjectProposals/ExportToXml?ProgrammeId=yIyRFEzMEDyPTP0ZcYrk5g%3D%3D&ShowRes=True"
)

TARGET_PROCEDURE_CODE = os.getenv("TARGET_PROCEDURE_CODE", "BG16RFPR002-1.010")
TARGET_PROCEDURE_NAME = os.getenv(
    "TARGET_PROCEDURE_NAME",
    "Зелени и цифрови партньорства за интелигентна трансформация"
)

OUT_CSV = os.getenv("OUT_CSV", "data/isun_bg16rfpr002-1.010_history.csv")


def fetch_xml(url: str) -> bytes:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; github-actions; +https://github.com)",
        "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
    r.raise_for_status()
    return r.content


import pandas as pd

def parse_xml_to_table(content_bytes: bytes) -> pd.DataFrame:
    """
    Despite the name, this now parses the ExportToHtml output (HTML tables).
    We keep the function name to minimize changes elsewhere.
    """
    html = content_bytes.decode("utf-8", errors="replace")

    # Quick guard: if for some reason we got an error page without tables
    if "<table" not in html.lower():
        raise RuntimeError("Expected HTML with <table>, got something else (possibly blocked).")

    # Read all tables from HTML
    tables = pd.read_html(html)

    # We want the table with the per-procedure rows.
    # In the page, there is usually a 'Общо' table and then 'Проектни предложения' table.
    # The second one typically contains 'Номер на процедура' column.
    for t in tables:
        cols = [str(c).strip() for c in t.columns]
        if "Номер на процедура" in cols:
            return t

    raise RuntimeError("Could not find the 'Номер на процедура' table in HTML export.")


def find_target_row(df: pd.DataFrame) -> pd.Series:
    code_cols = [c for c in df.columns if "процед" in c.lower() and "номер" in c.lower()] + \
               [c for c in df.columns if "procedure" in c.lower() and "number" in c.lower()] + \
               [c for c in df.columns if "Procedure" in c]

    name_cols = [c for c in df.columns if "име" in c.lower()] + \
               [c for c in df.columns if "name" in c.lower()]

    preferred_code = None
    for cand in ["Номер на процедура", "ProcedureNumber", "Procedure", "ProcedureCode"]:
        if cand in df.columns:
            preferred_code = cand
            break

    preferred_name = None
    for cand in ["Име на процедура", "ProcedureName", "Name"]:
        if cand in df.columns:
            preferred_name = cand
            break

    code_col = preferred_code or (code_cols[0] if code_cols else None)
    name_col = preferred_name or (name_cols[0] if name_cols else None)

    if code_col:
        hit = df[df[code_col].astype(str).str.strip() == TARGET_PROCEDURE_CODE]
        if len(hit) == 1:
            return hit.iloc[0]

    if name_col:
        hit = df[df[name_col].astype(str).str.contains(TARGET_PROCEDURE_NAME, regex=False)]
        if len(hit) == 1:
            return hit.iloc[0]

    raise RuntimeError(
        "Не намерих таргет ред в XML.\n"
        f"Колони: {list(df.columns)}\n"
        f"Търсих code={TARGET_PROCEDURE_CODE} и name contains '{TARGET_PROCEDURE_NAME}'."
    )


def normalize_metrics(row: pd.Series) -> dict:
    def pick(*names):
        for n in names:
            if n in row.index:
                return row[n]
        return None

    procedure_code = pick("Номер на процедура", "ProcedureNumber", "ProcedureCode", "Procedure")
    procedure_name = pick("Име на процедура", "ProcedureName", "Name")

    submitted_count = pick(
        "Брой подадени проектни предложения",
        "SubmittedCount",
        "CountSubmitted",
        "Submitted"
    )
    total_value = pick(
        "Обща стойност на подадените проектни предложения",
        "TotalSubmittedValue",
        "SubmittedTotalValue",
        "TotalValue"
    )
    bfp_eur = pick(
        "Стойност на подадените проектни предложения БФП (в евро)",
        "BfpEurValue",
        "BFP",
        "BfpValue"
    )
    approved = pick("Брой одобрени проектни предложения", "ApprovedCount", "Approved")
    reserve = pick("Брой проектни предложения в резервен списък", "ReserveCount", "Reserve")
    rejected = pick("Брой отхвърлени проектни предложения", "RejectedCount", "Rejected")

    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    return {
        "timestamp_utc": ts,
        "procedure_code": str(procedure_code).strip() if procedure_code is not None else TARGET_PROCEDURE_CODE,
        "procedure_name": str(procedure_name).strip() if procedure_name is not None else TARGET_PROCEDURE_NAME,
        "submitted_count": submitted_count,
        "total_value_eur": total_value,
        "bfp_value_eur": bfp_eur,
        "approved_count": approved,
        "reserve_count": reserve,
        "rejected_count": rejected,
    }


def load_existing(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)



def append_if_changed(existing: pd.DataFrame, new_row: dict) -> pd.DataFrame:
    new_df = pd.DataFrame([new_row])

    if existing.empty:
        return new_df

    compare_cols = [c for c in new_df.columns if c != "timestamp_utc"]

    last_row = existing.iloc[-1][compare_cols]
    current_row = new_df.iloc[0][compare_cols]

    if last_row.equals(current_row):
        return existing

    return pd.concat([existing, new_df], ignore_index=True)

def main():
    print("Fetching XML export…")
    xml_bytes = fetch_xml(EXPORT_XML_URL)

    print("Parsing XML…")
    table = parse_xml_to_table(xml_bytes)

    print("Finding target procedure row…")
    row = find_target_row(table)

    metrics = normalize_metrics(row)
    print("Metrics:", metrics)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    existing = load_existing(OUT_CSV)
    updated = append_if_changed(existing, metrics)

    if len(updated) == len(existing):
        return

    updated.to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV} (rows: {len(existing)} -> {len(updated)})")


if __name__ == "__main__":
    main()
