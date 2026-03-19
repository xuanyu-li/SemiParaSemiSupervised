from __future__ import print_function

import json
import sys
import numpy as np
from datetime import datetime
from bisect import bisect_right, bisect_left
from data_access import DataAccess


OUTPUT_PATIENTS_JSON = "subjects_with_stable_fio2_peep.json"

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


def get_values_in_interval(arr, left_t, right_t):
    """
    Return rows with timestamps in (left_t, right_t].
    arr: sorted by time ascending, each row = [timestamp_ms, value]
    """
    if arr is None or len(arr) == 0:
        return np.empty((0, 2), dtype=float)

    rows = []
    for k in range(len(arr)):
        tk = int(arr[k, 0])
        if tk <= left_t:
            continue
        if tk > right_t:
            break
        rows.append((arr[k, 0], arr[k, 1]))

    if len(rows) == 0:
        return np.empty((0, 2), dtype=float)

    return np.array(rows, dtype=float)


def get_stable_peep_between_strict(signal_arr, t1, t2, tol=1e-6):
    """
    New strict rule:
    - find the nearest PEEP at or before t1
    - find the nearest PEEP at or after t2
    - take all PEEP records from that left boundary to that right boundary
    - only if all values are equal, return that stable value
    - otherwise return None

    signal_arr: Nx2 numpy array sorted by timestamp ascending
                each row = [timestamp_ms, value]
    """
    if signal_arr is None or len(signal_arr) == 0:
        return None

    # index of last point at or before t1
    left_idx = bisect_right(signal_arr[:, 0], t1) - 1
    if left_idx < 0:
        return None

    # index of first point at or after t2
    right_idx = bisect_left(signal_arr[:, 0], t2)
    if right_idx >= len(signal_arr):
        return None

    # all records from left boundary to right boundary, inclusive
    window = signal_arr[left_idx:right_idx + 1]
    if len(window) == 0:
        return None

    vals = window[:, 1].astype(float)

    if np.max(vals) - np.min(vals) > tol:
        return None

    return float(vals[0])


def patient_has_eligible_pair(data_access, patient_id,
                              min_gap_minutes=20,
                              max_gap_minutes=60):
    """
    A patient is eligible if there exists at least one PaO2 pair (t1, t2) such that:
    - t2 - t1 is between 20 and 60 minutes
    - FiO2 just before/at t1 equals FiO2 just after/at t2
    - PEEP is constant under the strict rule between t1 and t2
    """
    pao2_arr = to_ts_value_array(data_access.get_pao2_values(patient_id))
    fio2_arr = to_ts_value_array(data_access.get_fio2_values(patient_id))
    peep_arr = to_ts_value_array(data_access.get_peep_values(patient_id))

    if len(pao2_arr) < 2 or len(fio2_arr) == 0 or len(peep_arr) == 0:
        return False

    min_gap_ms = min_gap_minutes * 60 * 1000
    max_gap_ms = max_gap_minutes * 60 * 1000

    n = len(pao2_arr)
    for i in range(n):
        t1 = int(pao2_arr[i, 0])

        for j in range(i + 1, n):
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


def main():
    da = DataAccess(".")

    patients = list(da.get_ventilated_patients(use_cache=True))

    patients = sorted(int(x) for x in patients)

    matched_subjects = []

    for idx, patient_id in enumerate(patients, start=1):
        try:
            ok = patient_has_eligible_pair(
                da,
                patient_id,
                min_gap_minutes=MIN_GAP_MINUTES,
                max_gap_minutes=MAX_GAP_MINUTES
            )

            if ok:
                matched_subjects.append(int(patient_id))
                print(
                    "[INFO] patient {}/{} matched: subject_id={}, total_matched={}".format(
                        idx, len(patients), patient_id, len(matched_subjects)
                    ),
                    file=sys.stderr
                )
            else:
                print(
                    "[INFO] patient {}/{} skipped: subject_id={}".format(
                        idx, len(patients), patient_id
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

    with open(OUTPUT_PATIENTS_JSON, "w") as f:
        json.dump(matched_subjects, f)

    print(
        "[INFO] Done. input_patients={}, matched_patients={}, output_file={}".format(
            len(patients), len(matched_subjects), OUTPUT_PATIENTS_JSON
        ),
        file=sys.stderr
    )


if __name__ == "__main__":
    main()