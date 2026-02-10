# ISUN Project Proposals Tracker

An automated **hourly tracker** for project proposals from **ISUN 2020**, focused on **a single specific procedure**, while preserving **historical snapshots** without overwriting past data.

Currently configured for:

**BG16RFPR002-1.010 –  
“Green and Digital Partnerships for Smart Transformation”**

---

## What This Project Does

- ⏱️ Runs **automatically every hour** (GitHub Actions cron)
- 🌐 Fetches the public HTML/XML export from ISUN 2020
- 🔎 Locates the row for the target procedure
- 📊 Extracts the following metrics:
  - Number of submitted project proposals
  - Total value of submitted project proposals
  - Value of submitted project proposals – EU grant (EUR)
  - Number of approved project proposals
  - Number of project proposals on the reserve list
  - Number of rejected project proposals
- 🧠 **Appends a new row only if the numbers have actually changed**
- 🗂️ Stores data in an append-only CSV history file
- 🤖 Automatically commits & pushes changes when data is updated

---

## Data Output

The data is stored in the following file:

