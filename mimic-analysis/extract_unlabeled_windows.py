from __future__ import print_function

import json
import csv
import sys
import numpy as np
from datetime import datetime
from bisect import bisect_right, bisect_left
from data_access import DataAccess


INPUT_PATIENTS_JSON = "ventilated_patients.json"
OUTPUT_CSV = "unlabeled_windows.csv"

MIN_GAP_MINUTES = 20
MAX_GAP_MINUTES = 60

# fixed pseudo-horizon for unlabeled windows
UNLABELED_GAP_MINUTES = 30

SUPERVISED_PATIENTS_JSON = "supervised_patients.json"

def timestamp_string_to_datetime(timestamp):
    try:
        return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
    except Exception:
        return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")


def timestamp_from_string(timestamp_string):
    return int(timestamp_string_to_datetime(timestamp_string).strftime("%s")) * 1000

def select_best_unlabeled_window(windows):
    """
    Keep only one unlabeled window per patient.

    Priority:
    1. smallest lag_from_peep_change_min
    2. earliest t1 if tied
    """
    if len(windows) == 0:
        return None

    def sort_key(w):
        lag = w["lag_from_peep_change_min"]
        if lag is None:
            lag = float("inf")
        return (lag, w["t1"])

    windows = sorted(windows, key=sort_key)
    return windows[0]
    
def to_ts_value_array(raw_pairs):
    if raw_pairs is None or len(raw_pairs) == 0:
        return np.empty((0, 2), dtype=float)

    rows = []
    for t, v in raw_pairs:
        if v is None:
            continue
        try:
            rows.append((timestamp_from_string(t), float(v)))
        except Exception:
            continue

    if len(rows) == 0:
        return np.empty((0, 2), dtype=float)

    rows.sort(key=lambda x: x[0])
    return np.array(rows, dtype=float)


def is_close_float(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def get_last_value_at_or_before(arr, t):
    last_val = None
    for k in range(len(arr)):
        tk = int(arr[k, 0])
        if tk <= t:
            last_val = float(arr[k, 1])
        else:
            break
    return last_val

def get_last_value_before_t1(arr, t1, max_gap_hours=10):
    """
    Return the nearest value at or before t1.
    If none exists, return None.
    If the nearest value is more than max_gap_hours before t1, return None.
    """
    if arr is None or len(arr) == 0:
        return None

    max_gap_ms = max_gap_hours * 60 * 60 * 1000

    left_idx = None
    for k in range(len(arr)):
        tk = int(arr[k, 0])
        if tk <= t1:
            left_idx = k
        else:
            break

    if left_idx is None:
        return None

    left_t = int(arr[left_idx, 0])
    if (t1 - left_t) > max_gap_ms:
        return None

    return float(arr[left_idx, 1])
    
def get_first_value_at_or_after(arr, t):
    for k in range(len(arr)):
        tk = int(arr[k, 0])
        if tk >= t:
            return float(arr[k, 1])
    return None


def get_nearest_weight_for_window(weight_arr, t1, t2):
    if weight_arr is None or len(weight_arr) == 0:
        return None

    left_idx = None
    right_idx = None

    for k in range(len(weight_arr)):
        tk = int(weight_arr[k, 0])
        if tk <= t1:
            left_idx = k
        else:
            break

    for k in range(len(weight_arr)):
        tk = int(weight_arr[k, 0])
        if tk >= t2:
            right_idx = k
            break

    left_val = None
    right_val = None
    left_gap = None
    right_gap = None

    if left_idx is not None:
        left_t = int(weight_arr[left_idx, 0])
        left_val = float(weight_arr[left_idx, 1])
        left_gap = t1 - left_t

    if right_idx is not None:
        right_t = int(weight_arr[right_idx, 0])
        right_val = float(weight_arr[right_idx, 1])
        right_gap = right_t - t2

    if left_val is None and right_val is None:
        return None
    if left_val is None:
        return right_val
    if right_val is None:
        return left_val

    return left_val if left_gap <= right_gap else right_val


def get_last_peep_change_before_t1(peep_arr, t1, tol=1e-6):
    if peep_arr is None or len(peep_arr) < 2:
        return None

    hist = []
    for k in range(len(peep_arr)):
        tk = int(peep_arr[k, 0])
        if tk <= t1:
            hist.append((tk, float(peep_arr[k, 1])))
        else:
            break

    if len(hist) < 2:
        return None

    last_change_time = None
    prev_val = hist[0][1]

    for k in range(1, len(hist)):
        tk, cur_val = hist[k]
        if abs(cur_val - prev_val) > tol:
            last_change_time = tk
        prev_val = cur_val

    return last_change_time


def get_stable_peep_between_strict(signal_arr, t1, t2, tol=1e-6, max_gap_hours=12):
    if signal_arr is None or len(signal_arr) == 0:
        return None

    max_gap_ms = max_gap_hours * 60 * 60 * 1000

    left_idx = bisect_right(signal_arr[:, 0], t1) - 1
    if left_idx < 0:
        return None

    left_time = int(signal_arr[left_idx, 0])
    if (t1 - left_time) > max_gap_ms:
        return None

    right_idx = bisect_left(signal_arr[:, 0], t2)
    if right_idx >= len(signal_arr):
        return None

    right_time = int(signal_arr[right_idx, 0])
    if (right_time - t2) > max_gap_ms:
        return None

    window = signal_arr[left_idx:right_idx + 1]
    if len(window) == 0:
        return None

    vals = window[:, 1].astype(float)
    if np.max(vals) - np.min(vals) > tol:
        return None

    return float(vals[0])


def get_window_tidal_value(tidal_arr, t1, t2, tidal_window_hours=10, tol=100):
    """
    In [t1-10h, t2+10h], find the two nearest tidal observations to the window.
    If they differ by <= 100, return their mean; otherwise None.
    """
    if tidal_arr is None or len(tidal_arr) == 0:
        return None

    tidal_window_ms = tidal_window_hours * 60 * 60 * 1000
    left_bound = t1 - tidal_window_ms
    right_bound = t2 + tidal_window_ms

    candidates = []
    for k in range(len(tidal_arr)):
        ts = int(tidal_arr[k, 0])
        val = float(tidal_arr[k, 1])

        if ts < left_bound:
            continue
        if ts > right_bound:
            break

        if ts < t1:
            dist = t1 - ts
        elif ts > t2:
            dist = ts - t2
        else:
            dist = 0

        candidates.append((dist, ts, val))

    if len(candidates) < 2:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]))
    v1 = candidates[0][2]
    v2 = candidates[1][2]

    if abs(v1 - v2) > tol:
        return None

    return (v1 + v2) / 2.0


def exists_true_labeled_pair_from_t1(pao2_arr, fio2_arr, peep_arr, i,
                                     min_gap_minutes=20,
                                     max_gap_minutes=60):
    """
    Check whether pao2_arr[i] can form a true labeled pair with some later PaO2.
    If yes, we skip this t1 from the unlabeled set.
    """
    t1 = int(pao2_arr[i, 0])

    min_gap_ms = min_gap_minutes * 60 * 1000
    max_gap_ms = max_gap_minutes * 60 * 1000

    for j in range(i + 1, len(pao2_arr)):
        t2 = int(pao2_arr[j, 0])
        delta_ms = t2 - t1

        if delta_ms < min_gap_ms:
            continue
        if delta_ms > max_gap_ms:
            break

        fio2_before_t1 = get_last_value_at_or_before(fio2_arr, t1)
        fio2_after_t2 = get_first_value_at_or_after(fio2_arr, t2)

        if fio2_before_t1 is None or fio2_after_t2 is None:
            continue
        if not is_close_float(fio2_before_t1, fio2_after_t2, tol=1e-6):
            continue

        peep_val = get_stable_peep_between_strict(peep_arr, t1, t2, tol=1e-6)
        if peep_val is None:
            continue

        return True

    return False

def find_unlabeled_windows_for_patient(da, patient_id, pao2_arr, fio2_arr, peep_arr, tidal_arr,
                                       weight_arr,
                                       age_val,
                                       pseudo_gap_minutes=30):
    """
    Build unlabeled windows:
    - anchor on PaO2 at t1
    - define pseudo t2 = t1 + 30 min
    - require same FiO2/PEEP logic as labeled set
    - require tidal summary available
    - append covariates into each window dict
    """
    windows = []

    if len(pao2_arr) == 0 or len(fio2_arr) == 0 or len(peep_arr) == 0:
        return windows

    pseudo_gap_ms = pseudo_gap_minutes * 60 * 1000

    # collect covariate arrays once per patient
    meanbp_arr = to_ts_value_array(da.get_meanbp_values(patient_id))
    hr_arr = to_ts_value_array(da.get_hr_values(patient_id))
    rr_arr = to_ts_value_array(da.get_rr_values(patient_id))
    tempc_arr = to_ts_value_array(da.get_tempc_values(patient_id))
    spo2_arr = to_ts_value_array(da.get_spo2_values(patient_id))
    plateau_arr = to_ts_value_array(da.get_plateau_pressure_values(patient_id))

    for i in range(len(pao2_arr)):
        t1 = int(pao2_arr[i, 0])
        pao2_t1 = float(pao2_arr[i, 1])

        t2 = t1 + pseudo_gap_ms

        fio2_before_t1 = get_last_value_at_or_before(fio2_arr, t1)
        fio2_after_t2 = get_first_value_at_or_after(fio2_arr, t2)

        if fio2_before_t1 is None or fio2_after_t2 is None:
            continue
        if not is_close_float(fio2_before_t1, fio2_after_t2, tol=1e-6):
            continue
        if fio2_before_t1 <= 0:
            continue

        pf_ratio_t1 = (pao2_t1 * 100.0) / fio2_before_t1

        peep_val = get_stable_peep_between_strict(peep_arr, t1, t2, tol=1e-6)
        if peep_val is None:
            continue

        tidal_val = get_window_tidal_value(tidal_arr, t1, t2, tidal_window_hours=10, tol=100)
        if tidal_val is None:
            continue

        weight_val = get_nearest_weight_for_window(weight_arr, t1, t2)

        last_change_time = get_last_peep_change_before_t1(peep_arr, t1, tol=1e-6)
        lag_from_change_min = None if last_change_time is None else (t1 - last_change_time) / (60.0 * 1000.0)

        # find covariate values before t1
        meanbp_val = get_last_value_before_t1(meanbp_arr, t1)
        hr_val = get_last_value_before_t1(hr_arr, t1)
        rr_val = get_last_value_before_t1(rr_arr, t1)
        tempc_val = get_last_value_before_t1(tempc_arr, t1)
        spo2_val = get_last_value_before_t1(spo2_arr, t1)
        plateau_val = get_last_value_before_t1(plateau_arr, t1)

        # if core covariates are missing, skip this window
        if any(v is None for v in [meanbp_val, hr_val, rr_val, tempc_val, spo2_val, tidal_val]):
            continue

        windows.append({
            "t1": t1,
            "t2": t2,
            "subjectid": patient_id,
            "treatment": peep_val,
            "pf_ratio_t1": pf_ratio_t1,
            "fio2": fio2_before_t1,
            "age": age_val,
            "weight": weight_val,
            "lag_from_peep_change_min": lag_from_change_min,

            # covariates
            "meanbp": meanbp_val,
            "hr": hr_val,
            "rr": rr_val,
            "tempc": tempc_val,
            "spo2": spo2_val,
            "tidal_volume": tidal_val,
            "plateau_pressure": plateau_val,
        })

    return windows


def main():
    da = DataAccess(".")

    with open(INPUT_PATIENTS_JSON, "r") as f:
        patients = json.load(f)

    with open(SUPERVISED_PATIENTS_JSON, "r") as f:
        supervised_patients = json.load(f)

    patients = sorted(int(x) for x in patients)
    supervised_patients = {int(x) for x in supervised_patients}

    # exclude patients already used in supervised data
    patients = [p for p in patients if p not in supervised_patients]

    print(
        "[INFO] After excluding supervised patients, remaining unlabeled candidate patients = {}".format(
            len(patients)
        ),
        file=sys.stderr
    )

    written = 0

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subjectid",
            "t1",
            "t2",
            "treatment",
            "pf_ratio_t1",
            "fio2",
            "tidal_volume",
            "age",
            "weight",
            "meanbp",
            "hr",
            "rr",
            "tempc",
            "spo2",
            "plateau_pressure",
            "lag_from_peep_change_min",
        ])

        for idx, patient_id in enumerate(patients, start=1):
            try:
                pao2_arr = to_ts_value_array(da.get_pao2_values(patient_id))
                # add a rule that if pao2 number is less than 10, skip
                if len(pao2_arr) < 10:
                    continue
                fio2_arr = to_ts_value_array(da.get_fio2_values(patient_id))
                peep_arr = to_ts_value_array(da.get_peep_values(patient_id))
                tidal_arr = to_ts_value_array(da.get_tidal_volume_values(patient_id))
                weight_arr = to_ts_value_array(da.get_weight_values(patient_id))
                age_val = da.get_age(patient_id)

                windows = find_unlabeled_windows_for_patient(
                    da = da,
                    patient_id = patient_id,
                    pao2_arr=pao2_arr,
                    fio2_arr=fio2_arr,
                    peep_arr=peep_arr,
                    tidal_arr=tidal_arr,
                    weight_arr=weight_arr,
                    age_val=age_val,
                    pseudo_gap_minutes=UNLABELED_GAP_MINUTES,
                )
                
                    
                best_window = select_best_unlabeled_window(windows)
                
                if best_window is None:
                    print(
                        "[INFO] patient {}/{} skipped: subject_id={} (no eligible unlabeled window)".format(
                            idx, len(patients), patient_id
                        ),
                        file=sys.stderr
                    )
                    continue

                
                    
                writer.writerow([
                    patient_id,
                    best_window["t1"],
                    best_window["t2"],
                    best_window["treatment"],
                    best_window["pf_ratio_t1"],
                    best_window["fio2"],
                    best_window["tidal_volume"],
                    best_window["age"],
                    best_window["weight"],
                    best_window["meanbp"],
                    best_window["hr"],
                    best_window["rr"],
                    best_window["tempc"],
                    best_window["spo2"],
                    best_window["plateau_pressure"],
                    best_window["lag_from_peep_change_min"],
                ])
                written += 1
                
                print(
                    "[INFO] patient {}/{} written: subject_id={}, total_candidate_windows={}, total_written={}".format(
                        idx, len(patients), patient_id, len(windows), written
                    ),
                    file=sys.stderr
                )

            except Exception as e:
                print(
                    "[WARN] patient {}/{} failed: subject_id={}, error={}".format(
                        idx, len(patients), patient_id, str(e)
                    ),
                    file=sys.stderr
                )

    print(
        "[INFO] Done. input_patients={}, written_rows={}, output_csv={}".format(
            len(patients), written, OUTPUT_CSV
        ),
        file=sys.stderr
    )


if __name__ == "__main__":
    main()