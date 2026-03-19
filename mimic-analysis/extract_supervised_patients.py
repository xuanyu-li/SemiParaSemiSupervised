from __future__ import print_function

import json
import csv
import sys
import numpy as np
from datetime import datetime
from bisect import bisect_right, bisect_left
from data_access import DataAccess

INPUT_PATIENTS_JSON = "subjects_with_stable_fio2_peep.json"
OUTPUT_CSV = "supervised_patients.csv"

MIN_GAP_MINUTES = 20
MAX_GAP_MINUTES = 60


def timestamp_string_to_datetime(timestamp):
    try:
        return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
    except Exception:
        return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")


def timestamp_from_string(timestamp_string):
    return int(timestamp_string_to_datetime(timestamp_string).strftime("%s")) * 1000


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
    """
    Return the value from the latest record with timestamp <= t.
    arr: sorted by time ascending, each row = [timestamp_ms, value]
    """
    last_val = None
    for k in range(len(arr)):
        tk = int(arr[k, 0])
        if tk <= t:
            last_val = float(arr[k, 1])
        else:
            break
    return last_val


def get_first_value_at_or_after(arr, t):
    """
    Return the value from the earliest record with timestamp >= t.
    arr: sorted by time ascending, each row = [timestamp_ms, value]
    """
    for k in range(len(arr)):
        tk = int(arr[k, 0])
        if tk >= t:
            return float(arr[k, 1])
    return None


def get_stable_peep_between_strict(signal_arr, t1, t2, tol=1e-6, max_gap_hours=12):
    """
    Strict rule:
    - find the nearest PEEP at or before t1
    - find the nearest PEEP at or after t2
    - require both boundary PEEP records to be within max_gap_hours of t1/t2
    - take all PEEP records from that left boundary to that right boundary
    - only if all values are equal, return that stable value
    - otherwise return None

    Parameters
    ----------
    signal_arr : np.ndarray
        Nx2 array sorted by timestamp ascending, each row = [timestamp_ms, value]
    t1 : int
        Start timestamp in ms
    t2 : int
        End timestamp in ms
    tol : float
        Tolerance for equality of PEEP values
    max_gap_hours : float
        Maximum allowed distance from t1/t2 to the boundary PEEP records
    """
    if signal_arr is None or len(signal_arr) == 0:
        return None

    max_gap_ms = max_gap_hours * 60 * 60 * 1000

    # nearest PEEP at or before t1
    left_idx = bisect_right(signal_arr[:, 0], t1) - 1
    if left_idx < 0:
        return None

    left_time = int(signal_arr[left_idx, 0])
    if (t1 - left_time) > max_gap_ms:
        return None

    # nearest PEEP at or after t2
    right_idx = bisect_left(signal_arr[:, 0], t2)
    if right_idx >= len(signal_arr):
        return None

    right_time = int(signal_arr[right_idx, 0])
    if (right_time - t2) > max_gap_ms:
        return None

    # all records from left boundary to right boundary, inclusive
    window = signal_arr[left_idx:right_idx + 1]
    if len(window) == 0:
        return None

    vals = window[:, 1].astype(float)

    if np.max(vals) - np.min(vals) > tol:
        return None

    return float(vals[0])
 

def get_weight_before_t1(arr, t1 ):
    """
    Return the nearest value at or before t1.
    If none exists, return None.
    """
    if arr is None or len(arr) == 0:
        return None

    left_idx = None
    for k in range(len(arr)):
        tk = int(arr[k, 0])
        if tk <= t1:
            left_idx = k
        else:
            break

    if left_idx is None:
        return None

    return float(arr[left_idx, 1])

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


def get_last_peep_change_before_t1(peep_arr, t1,tol=1e-6):
    """
    Find the most recent timestamp <= t1 at which PEEP changed.

    peep_arr: Nx2 numpy array sorted by timestamp ascending
              each row = [timestamp_ms, value]

    Returns
    -------
    int or None
        Timestamp (ms) of the last detected PEEP change before or at t1.
        Returns None if no change point can be identified.
    """
    if peep_arr is None or len(peep_arr) < 2:
        return None

    # keep only observations at or before t1
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

def get_window_tidal_value(tidal_arr, t1, t2, tol=100):
        """
        In [t1-10h, t2+10h], find the two nearest tidal observations to the window.
        Distance to window is:
        - 0 if observation is inside [t1, t2]
        - t1 - ts if ts < t1
        - ts - t2 if ts > t2

        If fewer than 2 observations exist, return None.
        If the two nearest values differ by more than tol, return None.
        Otherwise return their mean.
        """
        tidal_window_ms = 10 * 60 * 60 * 1000
        if tidal_arr is None or len(tidal_arr) == 0:
            return None

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
    
def find_all_eligible_windows(pao2_arr, fio2_arr, peep_arr, tidal_arr,
                              min_gap_minutes=20,
                              max_gap_minutes=40,
                             tidal_tol = 100):
    """
    Return all eligible windows for one patient.

    Each window contains:
    - t1, t2
    - treatment = stable peep
    - outcome = ((PaO2(t2) - PaO2(t1)) * 100) / FiO2
    - gap_ms
    - pf_ratio_t1 < 300 restriction
    - last_peep_change_before_t1
    - lag_from_peep_change_ms
    """
    windows = []

    if len(pao2_arr) < 2 or len(fio2_arr) == 0 or len(peep_arr) == 0:
        return windows

    min_gap_ms = min_gap_minutes * 60 * 1000
    max_gap_ms = max_gap_minutes * 60 * 1000
    
    n = len(pao2_arr)
    for i in range(n):
        t1 = int(pao2_arr[i, 0])
        pao2_t1 = float(pao2_arr[i, 1])

        for j in range(i + 1, n):
            t2 = int(pao2_arr[j, 0])
            pao2_t2 = float(pao2_arr[j, 1])

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

            if fio2_before_t1 <= 0:
                continue

            # standard P/F ratio
            pf_ratio_t1 = (pao2_t1 * 100.0) / fio2_before_t1
            # if pf_ratio_t1 >= 300:
            #     continue

            peep_val = get_stable_peep_between_strict(peep_arr, t1, t2, tol=1e-6)
            if peep_val is None:
                continue

            tidal_val = get_window_tidal_value(tidal_arr, t1, t2, tol=tidal_tol)
            if tidal_val is None:
                continue

            
            outcome = ((pao2_t2 - pao2_t1) * 100.0) / fio2_before_t1

            last_change_time = get_last_peep_change_before_t1(peep_arr, t1, tol=1e-6)
            lag_from_change_ms = None if last_change_time is None else (t1 - last_change_time)

            windows.append({
                "t1": t1,
                "t2": t2,
                "gap_ms": delta_ms,
                "treatment": peep_val,
                "outcome": outcome,
                "fio2": fio2_before_t1,
                "pao2_t1": pao2_t1,
                "pao2_t2": pao2_t2,
                "fio2_t1": fio2_before_t1,
                "fio2_t2": fio2_after_t2,
                "pf_ratio_t1": pf_ratio_t1,
                "last_peep_change_before_t1": last_change_time,
                "lag_from_peep_change_ms": lag_from_change_ms,
                "tidal_volume": tidal_val,
            })


    return windows


def select_best_window(windows):
    """
    Select the best window by:
    1. smallest lag from the most recent PEEP change before t1
    2. if tied, shortest gap_ms

    Windows with no detectable prior PEEP change are ranked last.
    """
    if len(windows) == 0:
        return None

    def sort_key(w):
        lag = w["lag_from_peep_change_ms"]
        if lag is None:
            lag = float("inf")
        return (lag, w["gap_ms"])

    windows = sorted(windows, key=sort_key)
    return windows[0]

def main():
    da = DataAccess(".")

    with open(INPUT_PATIENTS_JSON, "r") as f:
        patients = json.load(f)

    patients = sorted(int(x) for x in patients)

    written = 0

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subjectid",
            "treatment",
            "outcome",
            "pf_ratio_t1",
            "pf_ratio_t2",
            "age",
            "weight",
            "meanbp",
            "hr",
            "rr",
            "tempc",
            "tidal_volume",
            "plateau_pressure",
            "time_gap"
        ])

        for idx, patient_id in enumerate(patients, start=1):
            try:
                pao2_arr = to_ts_value_array(da.get_pao2_values(patient_id))
                # add a rule that if pao2 number is less than 50, skip
                if len(pao2_arr) < 50:
                    continue
                fio2_arr = to_ts_value_array(da.get_fio2_values(patient_id))
                peep_arr = to_ts_value_array(da.get_peep_values(patient_id))
                tidal_arr = to_ts_value_array(da.get_tidal_volume_values(patient_id))
                # new succinct covariates
                meanbp_arr = to_ts_value_array(da.get_meanbp_values(patient_id))
                hr_arr = to_ts_value_array(da.get_hr_values(patient_id))
                rr_arr = to_ts_value_array(da.get_rr_values(patient_id))
                tempc_arr = to_ts_value_array(da.get_tempc_values(patient_id))
                spo2_arr = to_ts_value_array(da.get_spo2_values(patient_id))
                plateau_arr = to_ts_value_array(da.get_plateau_pressure_values(patient_id))
                    
                windows = find_all_eligible_windows(
                    pao2_arr,
                    fio2_arr,
                    peep_arr,
                    tidal_arr,
                    min_gap_minutes=MIN_GAP_MINUTES,
                    max_gap_minutes=MAX_GAP_MINUTES
                )

                best_window = select_best_window(windows)

                # keep only plausible PEEP values in (4.5, 30)
                if best_window["treatment"] is None or not (4.5 < best_window["treatment"] < 30):
                    print(
                        "[INFO] patient {}/{} skipped: subject_id={}".format(
                            idx, len(patients), patient_id
                        ),
                        file=sys.stderr
                    )
                    continue

                age_val = da.get_age(patient_id)

                weight_arr = to_ts_value_array(da.get_weight_values(patient_id))
                weight_val = get_weight_before_t1(
                    weight_arr,
                    best_window["t1"]
                )

                # FiO2 pair used for this window:
                # - before/at t1
                # - after/at t2
                fio2_t1 = get_last_value_at_or_before(fio2_arr, best_window["t1"])
                fio2_t2 = get_first_value_at_or_after(fio2_arr, best_window["t2"])

                # PaO2 values from the selected window
                pao2_t1 = best_window["pao2_t1"]
                pao2_t2 = best_window["pao2_t2"]

                # P/F ratios
                pf_ratio_t1 = None if fio2_t1 is None or fio2_t1 <= 0 else (pao2_t1 * 100.0) / fio2_t1
                pf_ratio_t2 = None if fio2_t2 is None or fio2_t2 <= 0 else (pao2_t2 * 100.0) / fio2_t2

                # collect covariates: nearest values for the selected window
                meanbp_val = get_last_value_before_t1(meanbp_arr, best_window["t1"] )
                hr_val = get_last_value_before_t1(hr_arr, best_window["t1"] )
                rr_val = get_last_value_before_t1(rr_arr, best_window["t1"] )
                tempc_val = get_last_value_before_t1(tempc_arr, best_window["t1"] )
                tidal_val = get_window_tidal_value(tidal_arr, best_window["t1"], best_window["t2"] )
                plateau_val = get_last_value_before_t1(plateau_arr, best_window["t1"] )

                # if any covariate is missing, skip this patient/window
                if any(v is None for v in [meanbp_val, hr_val, rr_val, tempc_val, tidal_val]):
                    continue
                    
                writer.writerow([
                    patient_id,
                    best_window["treatment"],
                    best_window["outcome"],
                    pf_ratio_t1,
                    pf_ratio_t2,
                    age_val,
                    weight_val,
                    meanbp_val,
                    hr_val,
                    rr_val,
                    tempc_val,
                    tidal_val,
                    plateau_val,
                    best_window["gap_ms"] / (60.0 * 1000.0)
                ])
                written += 1
                
                print(
                    "[INFO] patient {}/{} written: subject_id={}, gap_min={:.2f}, treatment={}, outcome={}, pf_t1={}, pf_t2={}, total_written={}".format(
                        idx,
                        len(patients),
                        patient_id,
                        best_window["gap_ms"] / (60.0 * 1000.0),
                        best_window["treatment"],
                        best_window["outcome"],
                        pf_ratio_t1,
                        pf_ratio_t2,
                        written
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