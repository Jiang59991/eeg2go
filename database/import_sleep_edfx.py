import os
import re
import json
import sqlite3
from typing import List, Tuple

import mne
import numpy as np

from logging_config import logger

# Database path (align with other modules)
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "eeg2go.db"))


SLEEP_STAGE_MAP = {
    # EDF annotations usually like: 'Sleep stage W', 'Sleep stage 1', ...
    "W": "W",
    "0": "W",  # in case of "Sleep stage 0"
    "1": "N1",
    "2": "N2",
    "3": "N3",
    "4": "N3",  # merge N4 into N3
    "R": "REM",
}

DROP_LABELS = {"Movement time", "?", "Movement", "UNKNOWN", "Unscored"}


def _map_sleep_desc_to_aasm(desc: str) -> str | None:
    d = (desc or "").strip()
    # common patterns
    if d.lower().startswith("sleep stage"):
        token = d.split(" ")[-1].strip()
    else:
        token = d

    token_up = token.upper()
    if token_up in DROP_LABELS:
        return None

    if token_up in SLEEP_STAGE_MAP:
        return SLEEP_STAGE_MAP[token_up]

    # Sometimes annotations are just single letters/numbers
    if token in SLEEP_STAGE_MAP:
        return SLEEP_STAGE_MAP[token]

    # Unknown → drop
    return None


def _find_pair_hypnogram(psg_path: str) -> str | None:
    base_dir = os.path.dirname(psg_path)
    base_name = os.path.basename(psg_path)

    # e.g., SC4001E0-PSG.edf -> SC4001E0-Hypnogram.edf
    candidate = base_name.replace("-PSG.edf", "-Hypnogram.edf")
    cand_path = os.path.join(base_dir, candidate)
    if os.path.exists(cand_path):
        return cand_path

    # fallback: search by pattern
    for fname in os.listdir(base_dir):
        if fname.endswith(".edf") and "Hypnogram" in fname:
            return os.path.join(base_dir, fname)
    return None


def _extract_subject_id_from_fname(fname: str) -> str:
    # SC4001E0-PSG.edf -> SC4001
    m = re.match(r"^(SC|ST)(\d{4})", os.path.basename(fname))
    if m:
        return m.group(1) + m.group(2)
    # fallback: strip suffix
    return os.path.basename(fname).split("-")[0]


def import_sleep_edfx(conn: sqlite3.Connection, base_dir: str, dataset_name: str = "SleepEDFx_v1") -> int:
    c = conn.cursor()

    # dataset
    c.execute("SELECT id FROM datasets WHERE name = ?", (dataset_name,))
    row = c.fetchone()
    if row is None:
        c.execute(
            "INSERT INTO datasets (name, description, source_type, path) VALUES (?, ?, ?, ?)",
            (dataset_name, "Sleep-EDFx (expanded) v1.0.0", "edf", base_dir),
        )
        dataset_id = c.lastrowid
    else:
        dataset_id = row[0]

    subdirs = ["sleep-cassette", "sleep-telemetry"]

    imported = 0
    for sd in subdirs:
        d = os.path.join(base_dir, sd)
        if not os.path.exists(d):
            logger.warning(f"Subdir not found: {d}, skipping.")
            continue

        # iterate PSG files
        for fname in sorted(os.listdir(d)):
            if not fname.endswith("-PSG.edf"):
                continue

            psg_path = os.path.join(d, fname)
            subj_id = _extract_subject_id_from_fname(fname)

            # subject row
            c.execute("SELECT subject_id FROM subjects WHERE subject_id = ? AND dataset_id = ?", (subj_id, dataset_id))
            if not c.fetchone():
                c.execute("INSERT INTO subjects (subject_id, dataset_id) VALUES (?, ?)", (subj_id, dataset_id))

            # read meta from PSG
            try:
                raw = mne.io.read_raw_edf(psg_path, preload=False, verbose='ERROR')
                sfreq = float(raw.info['sfreq'])
                channels = int(len(raw.info['ch_names']))
                duration = float(raw.n_times) / sfreq
            except Exception as e:
                logger.error(f"SKIPPING (cannot read PSG): {fname}: {e}")
                continue

            # insert recording row
            c.execute(
                """INSERT INTO recordings
                (dataset_id, subject_id, filename, path, duration, channels, sampling_rate,
                 original_reference, recording_type, eeg_ground, placement_scheme, manufacturer,
                 powerline_frequency, software_filters, has_events, event_types)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    dataset_id,
                    subj_id,
                    fname,
                    psg_path,
                    duration,
                    channels,
                    sfreq,
                    "n/a",
                    "continuous",
                    "n/a",
                    "10-20",
                    "n/a",
                    "n/a",
                    "n/a",
                    False,
                    None,
                ),
            )
            recording_id = c.lastrowid

            # pair hypnogram and parse annotations
            hyp_path = _find_pair_hypnogram(psg_path)
            has_events = False
            event_types = None
            stage_values: List[str] = []
            if hyp_path and os.path.exists(hyp_path):
                try:
                    # Prefer direct annotations API (Sleep-EDFx hypnogram EDF+ stores stage epochs as annotations)
                    ann = mne.read_annotations(hyp_path)
                    # Clear previous sleep_stage events for idempotent re-runs
                    c.execute("DELETE FROM recording_events WHERE recording_id=? AND event_type='sleep_stage'", (recording_id,))
                    if ann is None or len(ann) == 0:
                        logger.warning(f"No annotations found in hypnogram EDF: {os.path.basename(hyp_path)}")
                    else:
                        for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
                            stage = _map_sleep_desc_to_aasm(desc)
                            if stage is None:
                                continue
                            dur = float(duration) if duration is not None else 0.0
                            if dur <= 0:
                                dur = 30.0
                            c.execute(
                                """INSERT INTO recording_events (recording_id, event_type, onset, duration, value)
                                VALUES (?, ?, ?, ?, ?)""",
                                (recording_id, "sleep_stage", float(onset), float(dur), stage),
                            )
                            stage_values.append(stage)
                        if len(stage_values) > 0:
                            has_events = True
                            event_types = json.dumps(sorted(list(set(stage_values))))
                except Exception as e:
                    logger.warning(f"Failed to read hypnogram for {fname}: {e}")

            # update recording row with events flag
            c.execute(
                "UPDATE recordings SET has_events = ?, event_types = ? WHERE id = ?",
                (has_events, event_types, recording_id),
            )

            imported += 1
            logger.info(f"Imported {fname} (rec_id={recording_id}), events={has_events}")

    conn.commit()
    logger.info(f"Sleep-EDFx import completed: {imported} recordings into dataset '{dataset_name}' (id={dataset_id}).")
    return dataset_id


def main():
    # Default base directory for user environment
    BASE_DIR = os.environ.get("SLEEP_EDFX_DIR", "/rds/general/user/zj724/ephemeral/sleep_edfs")
    DATASET_NAME = os.environ.get("SLEEP_EDFX_DATASET", "SleepEDFx_v1")

    if not os.path.exists(BASE_DIR):
        raise FileNotFoundError(f"Base dir not found: {BASE_DIR}")

    conn = sqlite3.connect(DB_PATH)
    try:
        import_sleep_edfx(conn, BASE_DIR, DATASET_NAME)
    finally:
        conn.close()


if __name__ == "__main__":
    main()


