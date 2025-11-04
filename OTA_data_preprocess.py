import os
from pathlib import Path
import pandas as pd
import re
from typing import Dict, List, Tuple
import csv
import tarfile
import numpy as np
import cv2


FNAME_RE = re.compile(r"^frame_(\d{10,})_(\d+)\.jpg$")

SEVERITY_BUCKETS = {
    "0": [0, 1],
    "1": [3, 4, 6, 8, 15, 17, 27],
    "2": [2, 5, 7, 10, 12, 16, 24, 26, 30, 31],
    "3": [9, 11, 14, 18, 20, 21, 28, 29],
    "4": [13, 19, 22, 23, 25, 32],
}

def build_label_to_severity() -> Dict[int, int]:
    label_to_sev = {}
    for sev_str, labels in SEVERITY_BUCKETS.items():
        sev = int(sev_str)
        for lab in labels:
            label_to_sev[int(lab)] = sev
    return label_to_sev

LABEL_TO_SEVERITY = build_label_to_severity()

def parse_fname(name: str):
    """
    Returns (timestamp_ms:int, frame_id:int) or None if not match.
    """
    m = FNAME_RE.match(name)
    if not m:
        return None
    ts = int(m.group(1))
    fid = int(m.group(2))
    return ts, fid

def estimate_fps(timestamps: List[int], max_fps: int = 60) -> float:
    if len(timestamps) < 2:
        return 15.0  # fallback
    diffs = np.diff(sorted(timestamps))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 15.0
    med_dt_ms = float(np.median(diffs))
    fps = 1000.0 / med_dt_ms
    fps = float(np.clip(fps, 1.0, float(max_fps)))
    return fps

def list_members_in_range(tar_path: Path, start_ts: int, end_ts: int) -> List[Tuple[int,int,str]]:
    """
    Returns list of (timestamp, frame_id, member_name) for frames within [start_ts, end_ts].
    Does not load image bytes.
    """
    out = []
    # Use "r|" stream mode to keep memory lower for large tars; but we need to iterate anyway.
    # We prefer default "r" to allow getmembers() which is fast for local files.
    with tarfile.open(tar_path, "r") as tf:
        for m in tf:
            if not m.isfile():
                continue
            # member names might include subfolders; use basename
            name = Path(m.name).name
            parsed = parse_fname(name)
            if parsed is None:
                continue
            ts, fid = parsed
            if start_ts <= ts <= end_ts:
                out.append((ts, fid, m.name))
    return out

def find_shards(frames_dir: Path) -> List[Path]:
    shards = sorted(frames_dir.glob("frames-*.tar"))
    if not shards:
        raise FileNotFoundError(f"No .tar shards found in {frames_dir}")
    return shards

def write_clip_from_members(shard_paths: List[Path],
                            members_index: Dict[Path, List[Tuple[int,int,str]]],
                            out_path: Path,
                            fps: float) -> int:
    """
    Stream frames from tar members to a VideoWriter.
    Returns number of frames written.
    """
    # First pass: open first image to get size
    first_img = None
    first_src = None
    for tar_p in shard_paths:
        entries = members_index.get(tar_p, [])
        if not entries:
            continue
        entries_sorted = sorted(entries, key=lambda x: (x[0], x[1]))
        with tarfile.open(tar_p, "r") as tf:
            m = tf.getmember(entries_sorted[0][2])
            f = tf.extractfile(m)
            if f is None:
                continue
            data = f.read()
            import numpy as np
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                first_img = img
                first_src = (tar_p, m.name)
                break
    if first_img is None:
        return 0
    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter at {out_path}")

    # We want global ordering across shards; build a global list.
    global_entries = []
    for tar_p, entries in members_index.items():
        for ts, fid, name in entries:
            global_entries.append((ts, fid, tar_p, name))
    global_entries.sort(key=lambda x: (x[0], x[1]))

    frames_written = 0
    for ts, fid, tar_p, name in global_entries:
        with tarfile.open(tar_p, "r") as tf:
            try:
                m = tf.getmember(name)
            except KeyError:
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            data = f.read()
            import numpy as np
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            # Resize to match first image if necessary
            if (img.shape[1], img.shape[0]) != (w, h):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            vw.write(img)
            frames_written += 1

    vw.release()
    return frames_written

def process_folder(folder_path):
    intersection_name = Path(folder_path).parts[-2]
    labels_csv = Path(folder_path) / "anomaly-labels.csv"
    frames_dir = Path(folder_path) / "frames_shards"
    outdir = Path(folder_path) / "videos"

    outdir.mkdir(parents=True, exist_ok=True)

    shards = find_shards(frames_dir)

    rows = []

    with open(labels_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"object_id", "start_timestamp", "end_timestamp", "label"}
        if not required_cols.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain columns {required_cols}, got {reader.fieldnames}")
        for r in reader:
            try:
                start_ts = int(r["start_timestamp"])
                end_ts = int(r["end_timestamp"])
                label = int(r["label"])
            except Exception as e:
                print(f"[WARN] Skipping row due to parse error: {r} ({e})")
                continue
            if end_ts < start_ts:
                start_ts, end_ts = end_ts, start_ts
            rows.append((start_ts, end_ts, label, r.get("object__id", "")))

    # Process each event
    for idx, (start_ts, end_ts, label, obj_id) in enumerate(rows, 1):
        severity = LABEL_TO_SEVERITY.get(label, -1)
        out_path = outdir / f"{intersection_name}_av_{idx}_{label}_{severity}.mp4"
        print(f"[{idx}/{len(rows)}] Event label={label}, severity={severity}, ts=[{start_ts},{end_ts}] -> {out_path}")

        # Build per-shard member lists in range
        members_index: Dict[Path, List[Tuple[int,int,str]]] = {}
        all_ts: List[int] = []
        for tar_p in shards:
            entries = list_members_in_range(tar_p, start_ts, end_ts)
            if entries:
                members_index[tar_p] = entries
                all_ts.extend([e[0] for e in entries])

        if not all_ts:
            print(f"  [WARN] No frames found in range for this event. Skipping.")
            continue

        fps = estimate_fps(all_ts, max_fps=30)
        print(f"  Frames: {sum(len(v) for v in members_index.values())} | Estimated FPS: {fps:.2f}")

        written = write_clip_from_members(shards, members_index, out_path, fps=fps)
        if written == 0:
            print(f"  [WARN] Wrote 0 frames for event (unexpected).")
        else:
            print(f"  [OK] Wrote {written} frames -> {out_path}")


if __name__ == "__main__":
    folders = ["./OTA/RangelineS116thSt/testdata","./OTA/RangelineSMedicalDr/testdata"]
    # "./OTA/MononElmStreetNB/testdata",
    for folder in folders:
        if os.path.exists(folder):
            process_folder(folder)
        else:
            print(f"Folder {folder} does not exist.")