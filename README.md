# ISUN Project Proposals Tracker

An automated **hourly** tracker for **ISUN 2020** project proposals that monitors **one specific procedure** and stores **historical snapshots** (append-only), without overwriting old data.

Currently configured for:

**BG16RFPR002-1.010 —  
“Green and Digital Partnerships for Smart Transformation”**  
(„Зелени и цифрови партньорства за интелигентна трансформация“)

---

## What this project does

- ⏱️ Runs **automatically every hour** (GitHub Actions cron)
- 🌐 Fetches the public export from ISUN 2020 (HTML export page)
- 🧩 Uses a **Playwright fallback** if ISUN returns an anti-bot JS page (TSPD)
- 🔎 Locates the row for the target procedure (by code or name)
- 📊 Extracts these metrics:
  - Number of submitted project proposals
  - Total value of submitted project proposals
  - EU grant amount (BFP) of submitted project proposals (EUR)
  - Number of approved proposals
  - Number of proposals on the reserve list
  - Number of rejected proposals
- 🧠 **Appends a new row only if the numbers changed**
- 🗂️ Stores history in a CSV file (append-only)
- 🤖 Auto-commit + push to the repo when a new snapshot is appended

---

## Output (CSV)

Data is stored in:

`data/isun_bg16rfpr002-1.010_history.csv`

### CSV columns

| timestamp_utc | Номер на процедура | Име на процедура | Брой подадени проектни предложения | Обща стойност на подадените проектни предложения | Стойност на подадените проектни предложения БФП (в евро) | Брой одобрени проектни предложения | Брой проектни предложения в резервен списък | Брой отхвърлени проектни предложения |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

Notes:
- `timestamp_utc` is ISO 8601 in UTC
- Count fields are integers
- Value fields are floats (EUR)
- No duplicates are added if the metric values remain unchanged

---

## Data source

ISUN public export (HTML):

`https://2020.eufunds.bg/bg/0/0/ProjectProposals/ExportToHtml?ProgrammeId=yIyRFEzMEDyPTP0ZcYrk5g%3D%3D&ShowRes=True`

---

## GitHub Actions

Workflow file:

`.github/workflows/hourly.yml`

What it does:
1. Sets up Python
2. Installs dependencies
3. Installs Playwright Chromium
4. Runs `scrape_isun.py`
5. Commits & pushes **only if the CSV changed**

You can also run it manually:
**Actions → Track ISUN proposals (hourly) → Run workflow**
