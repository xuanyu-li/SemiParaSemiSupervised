from __future__ import print_function

import sqlite3
from os.path import join


class DataAccess(object):
    DB_FILE_NAME = "mimic3.db"

    def __init__(self, data_dir="."):
        self.db = self.connect(data_dir)

    def connect(self, data_dir):
        db = sqlite3.connect(
            join(data_dir, DataAccess.DB_FILE_NAME),
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        return db

    def get_items_by_id_set(self, patient_id=None, id_set=set(), table_name="CHARTEVENTS",
                            get_subjects=False, limit=None, offset=None, value_case=None):
        if get_subjects:
            columns = "DISTINCT subject_id"
            subject_clause = ""
            order_clause = ""
        else:
            time_col = "charttime"
            columns = "{TIME_COL}, valuenum".format(TIME_COL=time_col) if value_case is None else \
                "{TIME_COL}, {VALUE_CASE}".format(TIME_COL=time_col, VALUE_CASE=value_case)
            subject_clause = "AND SUBJECT_ID = '{SUBJECT_ID}'".format(SUBJECT_ID=patient_id)
            order_clause = " ORDER BY {TIME_COL} ".format(TIME_COL=time_col)

        offset_clause = "" if offset is None else "OFFSET {OFFSET_NUM}".format(OFFSET_NUM=offset)
        limit_clause = "" if limit is None else "LIMIT {LIMIT_NUM}".format(LIMIT_NUM=limit)

        error_clause = ""
        if table_name.upper() == "CHARTEVENTS":
            error_clause = "{TABLE_NAME}.error IS NOT 1 AND ".format(TABLE_NAME=table_name)

        id_set = map(str, id_set)
        id_set_string = ", ".join(id_set)

        query = ("SELECT {COLUMNS} "
                 "FROM {TABLE_NAME} "
                 "WHERE {ERROR_CLAUSE} ITEMID IN ({ID_SET}) "
                 "{SUBJECT_CLAUSE} {ORDER_CLAUSE} {LIMIT_CLAUSE} {OFFSET_CLAUSE};") \
            .format(COLUMNS=columns,
                    TABLE_NAME=table_name,
                    ID_SET=id_set_string,
                    ERROR_CLAUSE=error_clause,
                    SUBJECT_CLAUSE=subject_clause,
                    ORDER_CLAUSE=order_clause,
                    LIMIT_CLAUSE=limit_clause,
                    OFFSET_CLAUSE=offset_clause)

        result = self.db.execute(query).fetchall()

        if value_case is not None:
            return list(filter(lambda x: x[1] is not None, result))
        else:
            return result

    def get_spo2_values(self, patient_id):
        return self.get_items_by_id_set(patient_id=patient_id, id_set={"646", "220277"})

    def get_fio2_values(self, patient_id):
        """
        Use FiO2 from labevents (ITEMID 50816).
        Explicitly cast valuenum to REAL because in this SQLite build it is stored as TEXT.
    
        Accept either:
          - fraction form in [0.20, 1.00], converted to percentage
          - percentage form in [20, 100]
    
        Exclude implausibly small fraction values like 0.04.
        """
        fio2_lab_item_ids = {"50816"}
        return self.get_items_by_id_set(
            patient_id=patient_id,
            id_set=fio2_lab_item_ids,
            table_name="labevents",
            value_case="""
            case
                when CAST(valuenum AS REAL) >= 0.20 and CAST(valuenum AS REAL) <= 1.00
                    then CAST(valuenum AS REAL) * 100
                when CAST(valuenum AS REAL) >= 20 and CAST(valuenum AS REAL) <= 100
                    then CAST(valuenum AS REAL)
                else null
            end as valuenum
            """
        )

    def get_peep_values(self, patient_id):
        return self.get_items_by_id_set(patient_id=patient_id, id_set={"60", "437", "505", "506", "686", "220339", "224700"})

    def get_hr_values(self, patient_id):
        return self.get_items_by_id_set(patient_id=patient_id, id_set={211, 220045})

    def get_meanbp_values(self, patient_id):
        return self.get_items_by_id_set(patient_id=patient_id, id_set={456, 52, 6702, 443, 220052, 220181, 225312})

    def get_rr_values(self, patient_id):
        return self.get_items_by_id_set(patient_id=patient_id, id_set={615, 618, 220210, 224690})

    def get_tempc_values(self, patient_id):
        return self.get_items_by_id_set(
            patient_id=patient_id,
            id_set={223762, 676, 223761, 678},
            value_case="case when itemid in (223761, 678) then (valuenum - 32) / 1.8 else valuenum end as valuenum"
        )

    def get_tidal_volume_values(self, patient_id):
        return self.get_items_by_id_set(patient_id=patient_id, id_set={639, 654, 681, 682, 683, 684, 224685, 224684, 224686})

    def get_plateau_pressure_values(self, patient_id):
        return self.get_items_by_id_set(patient_id=patient_id, id_set={543, 224696})

    def get_pao2_values(self, patient_id):
        return self.get_items_by_id_set(
            patient_id=patient_id,
            id_set={"50821"},
            table_name="labevents"
        )

    def get_age(self, patient_id):
        """
        Return approximate age in years at the patient's first admission.
    
        Parameters
        ----------
        patient_id : int or str
            SUBJECT_ID in MIMIC-III.
    
        Returns
        -------
        float or None
            Age in years at first admission, or None if unavailable.
        """
        row = self.db.execute(
            """
            SELECT
                MIN((julianday(a.admittime) - julianday(p.dob)) / 365.2425) AS first_admit_age
            FROM admissions AS a
            INNER JOIN patients AS p
                ON p.subject_id = a.subject_id
            WHERE a.subject_id = ?
            """,
            (patient_id,)
        ).fetchone()
    
        if row is None or row[0] is None:
            return None
    
        return float(row[0])

    def get_weight_values(self, patient_id):
        """
        Weight in kg from chartevents.
        Common MIMIC-III itemids:
        - 762: Admit Wt (kg)
        - 763: Daily Weight (kg)
        - 3723: Weight Kg
        - 3580: Present Weight (kg)
        - 226512: Admission Weight (Kg)
        - 224639: Daily Weight (Kg)
        """
        item_ids = {
            762, 763, 3723, 3580, 226512, 224639
        }
        return self.get_items_by_id_set(
            patient_id=patient_id,
            id_set=item_ids
        )
        
    def get_ventilated_patients(self,
                            use_cache=True,
                            cache_file="ventilated_patients.json",
                            refresh_cache=False):
        """
        Return a set of subject_ids for patients who:
          1. were age >= 14 and < 89 at their first admission, and
          2. have at least one non-error chartevents row with a ventilation-related ITEMID.
    
        Parameters
        ----------
        use_cache : bool
            Whether to use a local cache json file.
        cache_file : str
            Path to cache json file.
        refresh_cache : bool
            If True, ignore existing cache and rebuild it.
    
        Returns
        -------
        set[int]
            Set of matching subject_id values.
        """
        import os
        import json
    
        if use_cache and (not refresh_cache) and os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cached_rows = json.load(f)
            return {int(x) for x in cached_rows}
    
        ventilation_indicators_item_ids = (
            1, 60, 218, 221, 223, 437, 444, 445, 448, 449, 450, 501, 502, 503,
            505, 506, 543, 639, 654, 667, 668, 669, 670, 671, 672, 681, 682, 683,
            684, 686, 720, 1211, 1340, 1486, 1600, 1655, 2000, 3459, 5865, 5866,
            220339, 223848, 223849, 224419, 224685, 224684, 224686, 224687,
            224695, 224696, 224697, 224700, 224701, 224702, 224705, 224706,
            224707, 224709, 224738, 224746, 224747, 224750, 226873, 227187
        )
    
        itemid_placeholders = ",".join("?" for _ in ventilation_indicators_item_ids)
    
        query = f"""
            WITH first_admission_age AS (
                SELECT
                    a.subject_id,
                    MIN((julianday(a.admittime) - julianday(p.dob)) / 365.2425) AS first_admit_age
                FROM admissions AS a
                INNER JOIN patients AS p
                    ON p.subject_id = a.subject_id
                GROUP BY a.subject_id
            )
            SELECT DISTINCT ce.subject_id
            FROM chartevents AS ce
            INNER JOIN first_admission_age AS faa
                ON faa.subject_id = ce.subject_id
            WHERE faa.first_admit_age >= 14
              AND faa.first_admit_age < 89
              AND (ce.error IS NULL OR ce.error != 1)
              AND ce.itemid IN ({itemid_placeholders})
            ORDER BY ce.subject_id
        """
    
        cursor = self.db.execute(query, list(ventilation_indicators_item_ids))
        rows = cursor.fetchall()
        result = {int(row[0]) for row in rows}
    
        if use_cache:
            with open(cache_file, "w") as f:
                json.dump(sorted(result), f)
    
        return result
