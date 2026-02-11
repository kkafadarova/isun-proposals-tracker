# ISUN Project Proposals Tracker

An hourly tracker for **ISUN 2020** public project proposals that monitors **one specific procedure** and stores **historical snapshots** (append-only) without overwriting older data.

Currently configured for:

**BG16RFPR002-1.010 — “Green and Digital Partnerships for Intelligent Transformation”**  
(“Зелени и цифрови партньорства за интелигентна трансформация”)

---

## What it does

- ⏱️ Runs **every hour** via GitHub Actions cron
- 🌐 Fetches the public export from ISUN 2020 (prefers **Excel export** for reliability)
- 🔎 Finds the row for the target procedure
- 📊 Extracts these metrics:
  - Number of submitted project proposals
  - Total value of submitted proposals (EUR)
  - EU grant value (BFP) of submitted proposals (EUR)
  - Number of approved proposals
  - Number of reserve-list proposals
  - Number of rejected proposals
- 🧠 Appends a **new CSV row only when the metric values change**
- 🗂️ Writes append-only history to a CSV file
- 🤖 Commits & pushes only if the CSV changed

---

## Output data

The history is stored in:

`data/isun_bg16rfpr002-1.010_history.csv`

### CSV columns

| timestamp_utc | Номер на процедура | Име на процедура | Брой подадени проектни предложения | Обща стойност на подадените проектни предложения | Стойност на подадените проектни предложения БФП (в евро) | Брой одобрени проектни предложения | Брой проектни предложения в резервен списък | Брой отхвърлени проектни предложения |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

- `timestamp_utc` is ISO 8601 (UTC)
- Count fields are stored as **integers**
- Value fields are stored as **floats (EUR)**
- No duplicate rows are appended when values do not change

---

## Data source

Public ISUN export (Excel):

`https://2020.eufunds.bg/bg/0/0/ProjectProposals/ExportToExcel?ProgrammeId=...&ShowRes=True`

> Note: ISUN may occasionally return an anti-bot/protected page. In such cases the workflow will fail and save the last response under `debug/` for inspection.

---

## GitHub Actions

Workflow file:

- `.github/workflows/hourly.yml`

Steps:

1. Setup Python
2. Install dependencies
3. Run `scrape_isun.py`
4. Commit & push **only if the CSV changed**

You can also run it manually:
**Actions → Track ISUN proposals (hourly) → Run workflow**
