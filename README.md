# ISUN Project Proposals Tracker

An automated (hourly) tracker for **ISUN 2020** project proposals that monitors **one specific procedure only** and keeps **historical snapshots** without overwriting older data.

Currently configured for:

**BG16RFPR002-1.010 –  
“Green and Digital Partnerships for Smart Transformation”**  
(“Зелени и цифрови партньорства за интелигентна трансформация”)

---

## What this project does

- ⏱️ Runs **automatically every hour** (GitHub Actions cron)
- 🌐 Fetches the public ISUN export (HTML)
- 🔎 Finds the row for the target procedure
- 📊 Extracts the following metrics:
  - Number of submitted project proposals
  - Total value of submitted proposals
  - Value of submitted proposals (grant/BFP) in EUR
  - Number of approved proposals
  - Number of proposals on the reserve list
  - Number of rejected proposals
- 🧠 **Appends a new row only when the metrics change**
- 🗂️ Stores an append-only CSV history file
- 🤖 Automatically commits & pushes when there is a change

---

## Output data

The history is stored in:

`data/isun_bg16rfpr002-1.010_history.csv`

### CSV schema

| timestamp_utc | Procedure code | Procedure name | Submitted | Total value | Grant/BFP value (EUR) | Approved | Reserve | Rejected |
| ------------- | -------------- | -------------- | --------: | ----------: | --------------------: | -------: | ------: | -------: |

- `timestamp_utc` is ISO 8601 (UTC)
- Count fields are **integers**
- Value fields are **floats (EUR)**
- No duplicate rows are written if nothing changed

---

## Data source

Public ISUN export:

`https://2020.eufunds.bg/bg/0/0/ProjectProposals/ExportToHtml?ProgrammeId=yIyRFEzMEDyPTP0ZcYrk5g%3D%3D&ShowRes=True`

_(We parse the HTML table from the export because the XML export is not stable.)_

---

## GitHub Actions

Workflow file:

- `.github/workflows/hourly.yml`

What it does:

1. Set up Python
2. Install dependencies
3. Run `scrape_isun.py`
4. Commit & push **only if the CSV changed**

Manual run:
Actions → Track ISUN proposals (hourly) → Run workflow
