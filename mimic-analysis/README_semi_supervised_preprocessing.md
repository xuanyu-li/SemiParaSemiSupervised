# Semi-Supervised Preprocessing for MIMIC-III PEEP Analysis

## Purpose
This pipeline builds two datasets for partially linear / semi-supervised treatment-effect modeling:

- `supervised_patients.csv` (or equivalent): `(X, A, Y)` windows
- `unlabeled_windows.csv`: `(X, A)` windows with no outcome

where:

- `A` is treatment (`PEEP`)
- `Y` is short-horizon oxygenation change (`Delta P/F`)
- `X` contains baseline covariates

## End-to-End Workflow
Run the steps in this order:

1. Build a SQLite database from raw MIMIC-III CSV files.
2. Find ventilated adult patients.
3. Extract supervised windows.
4. Extract unlabeled windows (excluding supervised subjects).

---

## 1) Build the MIMIC-III SQLite Database
### Data Acess
To access the data, you would first need to create a Physionet account [MIMIC-III database](https://mimic.physionet.org/). Then after the completion of a training course in human subjects research,
and signing the data use agreement, you would have the acess to the full data. To build the database locally and create indexes for faster data preprocessing, you would need to 
put the download dataset in a folder.

Then use [build_mimic3_db.sh]

### What the script does
- Reads all `*.csv` / `*.csv.gz` in your MIMIC-III folder.
- Imports them into one SQLite file.
- Creates indexes for faster queries on key event tables.

### Before running
Edit these variables in the script:

- `DATA_DIR`: directory containing raw MIMIC-III CSV files
- `OUT_DIR`: output directory for the SQLite DB
- `OUTFILE`: final SQLite file path (default `mimic3.db`)

Current defaults in the script:
- `DATA_DIR="$HOME/mimic-iii-clinical-database-1.4"`
- `OUT_DIR="$HOME/autodl-fs"`
- `OUTFILE="$OUT_DIR/mimic3.db"`

### Run
```bash
bash build_mimic3_db.sh
```

### Expected result
A SQLite database is created at `OUTFILE`, with indexes already applied.

---

## 2) Extract Ventilated Adult Cohort

Use [find_subjects_with_stable_fio2_peep.py] to identify candidate subjects and cache reusable IDs.

Typical cached artifacts:
- `ventilated_patients.json`
- `subjects_with_stable_fio2_peep.json`

---

## 3) Build the Supervised Dataset `(X, A, Y)`

Use [extract_supervised_patients.py](/Users/mac/Desktop/causal_inference/longitudinal/semi-parametric-semi-supervised/code/mimic-analysis/extract_supervised_patients.py).

### Key window logic
- Choose PaO2 pair `(t1, t2)` with gap in a target range.
- Require FiO2 consistency around `t1` and `t2`.
- Require PEEP stability in the window.
- Require baseline oxygenation filter (e.g., `PF(t1) < 300`).
- Keep covariates observed before window start.

### Treatment and outcome
- Treatment: stable `PEEP`
- Outcome:
  - `PF = (PaO2 * 100) / FiO2`
  - `Y = Delta PF = PF(t2) - PF(t1)`

Typical output:
- `supervised_patients.csv` (or `stable_fio2_peep_best_windows.csv`)
- `supervised_patients.json`

---

## 4) Build the Unlabeled Dataset `(X, A)`

Use [extract_unlabeled_windows.py]

### Key logic
- Start from ventilated candidates.
- Exclude subjects in `supervised_patients.json`.
- Construct stable windows with same treatment/covariate structure.
- Do not include outcome `Y`.

Typical output:
- `unlabeled_windows.csv`

---

## Data Elements

## Core signals
- PaO2 (`labevents`)
- FiO2 (`labevents`, often `ITEMID 50816`)
- PEEP (`chartevents`)
- tidal volume (`chartevents`)

## Common covariates
- `age`, `weight`
- `meanbp`, `hr`, `rr`, `tempc`, `spo2`
- `tidal_volume`, `plateau_pressure`
- baseline oxygenation fields (e.g., `pf_ratio_t1`)

## Important implementation detail
In some SQLite imports, numeric lab/chart values can be text. Use explicit numeric casts (for example `CAST(valuenum AS REAL)`) when querying FiO2, PaO2, etc.

---

## Recommended Checks

1. Confirm row counts after each filter stage (FiO2 match, PEEP stability, covariate availability).
2. Compare covariate distributions between supervised and unlabeled sets.
3. Check missingness and outliers for each covariate.
4. Verify no subject leakage from supervised into unlabeled.

---

## Typical File Outputs

- `mimic3.db`
- `ventilated_patients.json`
- `subjects_with_stable_fio2_peep.json`
- `supervised_patients.csv`
- `unlabeled_windows.csv`
 
