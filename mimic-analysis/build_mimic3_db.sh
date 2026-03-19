#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$HOME/mimic-iii-clinical-database-1.4"
OUT_DIR="$HOME/autodl-fs"
OUTFILE="$OUT_DIR/mimic3.db"

mkdir -p "$OUT_DIR"

if [ -s "$OUTFILE" ]; then
    echo "File \"$OUTFILE\" already exists." >&2
    exit 111
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "Data directory \"$DATA_DIR\" does not exist." >&2
    exit 112
fi

echo "Creating SQLite database: $OUTFILE"
echo "Reading source files from: $DATA_DIR"

sqlite3 "$OUTFILE" <<'EOF'
PRAGMA journal_mode = OFF;
PRAGMA synchronous = OFF;
PRAGMA temp_store = MEMORY;
PRAGMA cache_size = -200000;
PRAGMA locking_mode = EXCLUSIVE;
PRAGMA foreign_keys = OFF;
EOF

shopt -s nullglob

cd "$DATA_DIR"

for FILE in *.csv *.csv.gz; do
    [ -f "$FILE" ] || continue

    TABLE_NAME=$(echo "${FILE%%.*}" | tr '[:upper:]' '[:lower:]')

    echo "Loading $FILE into table $TABLE_NAME"

    case "$FILE" in
        *.csv)
            sqlite3 "$OUTFILE" <<EOF
.mode csv
.import '$FILE' $TABLE_NAME
EOF
            ;;
        *.csv.gz)
            sqlite3 "$OUTFILE" <<EOF
.mode csv
.import '|gzip -dc "$FILE"' $TABLE_NAME
EOF
            ;;
        *)
            continue
            ;;
    esac

    echo "Finished loading $FILE"
done

echo "All files imported. Creating indexes..."

sqlite3 "$OUTFILE" <<'EOF'
CREATE INDEX IF NOT EXISTS idx_admissions_subject
    ON admissions(subject_id);

CREATE INDEX IF NOT EXISTS idx_admissions_hadm
    ON admissions(hadm_id);

CREATE INDEX IF NOT EXISTS idx_icustays_subject
    ON icustays(subject_id);

CREATE INDEX IF NOT EXISTS idx_icustays_hadm
    ON icustays(hadm_id);

CREATE INDEX IF NOT EXISTS idx_icustays_icustay
    ON icustays(icustay_id);

CREATE INDEX IF NOT EXISTS idx_patients_subject
    ON patients(subject_id);

CREATE INDEX IF NOT EXISTS idx_chartevents_subject_item
    ON chartevents(subject_id, itemid);

CREATE INDEX IF NOT EXISTS idx_chartevents_subj_item_time
    ON chartevents(subject_id, itemid, charttime);

CREATE INDEX IF NOT EXISTS idx_chartevents_icustay_item_time
    ON chartevents(icustay_id, itemid, charttime);

CREATE INDEX IF NOT EXISTS idx_labevents_subj_item_time
    ON labevents(subject_id, itemid, charttime);

CREATE INDEX IF NOT EXISTS idx_labevents_hadm_item_time
    ON labevents(hadm_id, itemid, charttime);

CREATE INDEX IF NOT EXISTS idx_inputevents_mv_icustay_item_time
    ON inputevents_mv(icustay_id, itemid, starttime);

CREATE INDEX IF NOT EXISTS idx_outputevents_icustay_item_time
    ON outputevents(icustay_id, itemid, charttime);

CREATE INDEX IF NOT EXISTS idx_procedureevents_mv_icustay_item_time
    ON procedureevents_mv(icustay_id, itemid, starttime);

CREATE INDEX IF NOT EXISTS idx_microbiology_hadm
    ON microbiologyevents(hadm_id);

CREATE INDEX IF NOT EXISTS idx_prescriptions_hadm
    ON prescriptions(hadm_id);

CREATE INDEX IF NOT EXISTS idx_d_items_itemid
    ON d_items(itemid);

CREATE INDEX IF NOT EXISTS idx_d_labitems_itemid
    ON d_labitems(itemid);

CREATE INDEX IF NOT EXISTS idx_d_icd_diag_itemid
    ON d_icd_diagnoses(icd9_code);

CREATE INDEX IF NOT EXISTS idx_d_icd_proc_itemid
    ON d_icd_procedures(icd9_code);
EOF

echo "Done. SQLite database created at: $OUTFILE"
