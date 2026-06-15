from __future__ import annotations

import random
import pandas as pd

from icu_pretrain.constants import SPLIT_NAMES
from icu_pretrain.data.eicu_event_builder import SplitRecord


def assign_patient_splits(
    outcomes: pd.DataFrame,
    patient_df: pd.DataFrame,
    seed: int = 42,
) -> list[SplitRecord]:
    if not isinstance(outcomes, pd.DataFrame) or not isinstance(patient_df, pd.DataFrame):
        raise ValueError("outcomes and patient_df must be pandas DataFrames")

    required_outcomes = {"patientunitstayid", "mortality"}
    if not required_outcomes.issubset(outcomes.columns):
        raise ValueError("outcomes is missing required columns")

    required_patient = {"patientunitstayid", "uniquepid", "hospitalid"}
    if not required_patient.issubset(patient_df.columns):
        raise ValueError("patient_df is missing required columns")

    merged = pd.merge(
        outcomes[["patientunitstayid", "mortality"]],
        patient_df[["patientunitstayid", "uniquepid", "hospitalid"]],
        on="patientunitstayid",
        how="inner"
    )

    patient_groups = {}
    for _, row in merged.iterrows():
        pid = row["uniquepid"]
        stay_id = row["patientunitstayid"]
        hid = row["hospitalid"]
        mort = int(row["mortality"])
        if pid not in patient_groups:
            patient_groups[pid] = {
                "stays": [],
                "mortality_sum": 0,
            }
        patient_groups[pid]["stays"].append((stay_id, hid, mort))
        patient_groups[pid]["mortality_sum"] += mort

    alive_only_patients = []
    expired_patients = []
    for pid, group in patient_groups.items():
        if group["mortality_sum"] > 0:
            expired_patients.append(pid)
        else:
            alive_only_patients.append(pid)

    if len(alive_only_patients) < 3 or len(expired_patients) < 3:
        raise ValueError(
            "cohort size is insufficient: at least 3 patients with only alive stays "
            "and 3 patients with expired stays are required to populate all splits."
        )

    split_assignments = {}
    split_stays = {"train": 0, "validation": 0, "test": 0}
    grand_total_stays = len(merged)
    rng = random.Random(seed)

    strata = [expired_patients, alive_only_patients]
    for stratum in strata:
        sorted_stratum = sorted(stratum)
        rng.shuffle(sorted_stratum)

        split_assignments[sorted_stratum[0]] = "train"
        split_stays["train"] += len(patient_groups[sorted_stratum[0]]["stays"])

        split_assignments[sorted_stratum[1]] = "validation"
        split_stays["validation"] += len(patient_groups[sorted_stratum[1]]["stays"])

        split_assignments[sorted_stratum[2]] = "test"
        split_stays["test"] += len(patient_groups[sorted_stratum[2]]["stays"])

        for pid in sorted_stratum[3:]:
            patient_stay_count = len(patient_groups[pid]["stays"])
            train_deficit = 0.70 * grand_total_stays - split_stays["train"]
            val_deficit = 0.15 * grand_total_stays - split_stays["validation"]
            test_deficit = 0.15 * grand_total_stays - split_stays["test"]

            deficits = [
                ("train", train_deficit),
                ("validation", val_deficit),
                ("test", test_deficit),
            ]
            best_split = max(deficits, key=lambda x: x[1])[0]

            split_assignments[pid] = best_split
            split_stays[best_split] += patient_stay_count

    split_records = []
    for pid, split_name in split_assignments.items():
        for stay_id, hid, mort in patient_groups[pid]["stays"]:
            split_records.append(
                SplitRecord(
                    patientunitstayid=stay_id,
                    uniquepid=pid,
                    hospitalid=hid,
                    split_name=split_name,
                )
            )

    if len(split_records) != len(outcomes):
        raise ValueError("Split assignment count does not match outcomes count")

    assigned_stay_ids = {r.patientunitstayid for r in split_records}
    expected_stay_ids = set(outcomes["patientunitstayid"])
    if assigned_stay_ids != expected_stay_ids:
        raise ValueError("Split stay IDs do not match eligible stay IDs")

    split_outcome_counts = {
        "train": {0: 0, 1: 0},
        "validation": {0: 0, 1: 0},
        "test": {0: 0, 1: 0},
    }
    stay_mortality = dict(zip(outcomes["patientunitstayid"], outcomes["mortality"]))
    for r in split_records:
        m = stay_mortality[r.patientunitstayid]
        split_outcome_counts[r.split_name][m] += 1

    for split, counts in split_outcome_counts.items():
        if counts[0] == 0 or counts[1] == 0:
            raise ValueError(
                f"cohort size is insufficient: split '{split}' is missing one or both outcome classes "
                f"(Alive count: {counts[0]}, Expired count: {counts[1]})."
            )

    return split_records
