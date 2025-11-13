#!/usr/bin/env python3
import os
import json
import random
from pathlib import Path
import re
from typing import Dict, List, Tuple, Any
import csv
import tarfile
import numpy as np
import cv2

# ---------- Config derived from dataset spec ----------
FRAMES_PER_SHARD = 10_000      # frames-0000xx.tar has 10,000 frames
STEP_MS = 100                  # 10 fps => one frame every 100 ms
SHARD_DURATION_MS = FRAMES_PER_SHARD * STEP_MS  # 1,000,000 ms per shard

FNAME_RE = re.compile(r"^frame_(\d{10,})_(\d+)\.jpg$")

SEVERITY_BUCKETS = {
    "0": [0, 1],
    "1": [3, 4, 6, 8, 15, 17, 27],
    "2": [2, 5, 7, 10, 12, 16, 24, 26, 30, 31],
    "3": [9, 11, 14, 18, 20, 21, 28, 29],
    "4": [13, 19, 22, 23, 25, 32],
}

def random_ts_add(start_ts:int, end_ts:int, global_min_ts:int, global_max_ts:int, max_add_ms:int=20):
    if end_ts - start_ts < 2000:
        add_start = random.randint(10, max_add_ms) * 100 # in ms
        add_end = random.randint(10, max_add_ms) * 100 # in ms
    else:
        add_start = random.randint(5, max_add_ms) * 100 # in ms
        add_end = random.randint(5, max_add_ms) * 100 # in ms
        
    start_ts = max(start_ts-add_start, global_min_ts) # Add 10 frames margin for event
    end_ts = min(end_ts+add_end, global_max_ts) # Add 10 frames margin for event
    return start_ts, end_ts

def record_obj_select(event_idx:int, event_obj_id:str, anomaly_event_obj_dict:Dict[int,Dict[str,Any]], full_obj_dict:Dict[str, List[Dict[str, Any]]],video_st,video_et) -> Dict[int,Dict[str,Any]]:
    if event_idx not in anomaly_event_obj_dict:
        anomaly_event_obj_dict[event_idx] = {}

    if event_obj_id in full_obj_dict:
        timestamps = [obj_data["timestamp"] for obj_data in full_obj_dict[event_obj_id]]
        if timestamps:
            min_timestamp = min(timestamps)
            max_timestamp = max(timestamps)
            
            # Find bounding boxes for min and max timestamps
            start_bbox = None
            end_bbox = None
            for obj_data in full_obj_dict[event_obj_id]:
                if obj_data["timestamp"] == min_timestamp and start_bbox is None:
                    start_bbox = obj_data["boundingbox"]
                if obj_data["timestamp"] == max_timestamp:
                    end_bbox = obj_data["boundingbox"]

            video_length = (video_et - video_st) / 1000.0  # in seconds


            if video_st <= min_timestamp and max_timestamp <= video_et:
                anomaly_start_ts = (min_timestamp - video_st) / 1000.0  # in seconds
                anomaly_end_ts = (max_timestamp - video_st) / 1000.0  # in seconds
            else:
                anomaly_start_ts = 0.0
                anomaly_end_ts = video_length
                
            anomaly_event_obj_dict[event_idx][event_obj_id] = {
            "video_length": video_length,
            "start_time": anomaly_start_ts,
            "end_time": anomaly_end_ts,
            "start_bbox": start_bbox,
            "end_bbox": end_bbox
            }
    return anomaly_event_obj_dict

def read_obj_detections_json(json_path: str) -> Dict[str, List[Dict[str, Any]]]:
    obj_dict: Dict[str, List[Dict[str, Any]]] = {}
    with open(json_path, "r") as f:
        data = json.load(f)
    
        for item in data:
            ts = item.get("timestamp", [])
            frame_index = item.get("frame_index", [])
            frame_key = item.get("frame_key", [])
            detections = item.get("detections", [])

            if detections is not []:
                for det in detections:
                    obj_id = det.get("object_id", "")
                    boundingbox = det.get("boundingbox", [])
                    if obj_id not in obj_dict:
                        obj_dict[obj_id] = []
                    obj_dict[obj_id].append({
                        "timestamp": ts,
                        "frame_index": frame_index,
                        "frame_key": frame_key,
                        "boundingbox": boundingbox
                    })
    return obj_dict

def save_anomaly_obj_dict(anomaly_event_obj_dict: Dict[int,Dict[str,Any]], out_json_path: str):
    with open(out_json_path, "w") as f:
        json.dump(anomaly_event_obj_dict, f, indent=4)

def build_label_to_severity() -> Dict[int, int]:
    label_to_sev = {}
    for sev_str, labels in SEVERITY_BUCKETS.items():
        sev = int(sev_str)
        for lab in labels:
            label_to_sev[int(lab)] = sev
    return label_to_sev

LABEL_TO_SEVERITY = build_label_to_severity()

def parse_fname(name: str):
    m = FNAME_RE.match(name)
    if not m:
        return None
    ts = int(m.group(1))
    fid = int(m.group(2))
    return ts, fid

def estimate_fps(timestamps: List[int], max_fps: int = 60) -> float:
    if len(timestamps) < 2:
        return 15.0
    diffs = np.diff(sorted(timestamps))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 15.0
    med_dt_ms = float(np.median(diffs))
    fps = 1000.0 / med_dt_ms
    fps = float(np.clip(fps, 1.0, float(max_fps)))
    return fps

def sorted_shards(frames_dir: Path) -> List[Path]:
    shards = sorted(frames_dir.glob("frames-*.tar"))
    if not shards:
        raise FileNotFoundError(f"No .tar shards found in {frames_dir}")
    return shards

def shard_min_max_timestamps(shard_paths: List[Path]) -> Dict[int, Tuple[int, int]]:
    """Return {shard_index: (min_ts, max_ts)} for each shard.
    This scans filenames inside each tar and records min/max timestamp.
    """
    out: Dict[int, Tuple[int, int]] = {}
    for tar_path in shard_paths:
        ts_min = None
        ts_max = None
        with tarfile.open(tar_path, "r") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                name = Path(m.name).name
                parsed = parse_fname(name)
                if parsed is None:
                    continue
                ts1, _ = parsed
                if ts_min is None or ts1 < ts_min:
                    ts_min = ts1
                if ts_max is None or ts1 > ts_max:
                    ts_max = ts1
        if ts_min is None or ts_max is None:
            raise RuntimeError(f"No frame_* files parsed in {tar_path}")
        idx = int(tar_path.stem.split("-")[-1])  # frames-000001.tar -> 1
        out[idx] = (ts_min, ts_max)
    # Keep order by index
    return dict(sorted(out.items()))

def pick_shards_for_range(shards: List[Path], ts_dict: Dict[int, Tuple[int,int]], start_ts: int, end_ts: int) -> List[Tuple[int, Path]]:
    """Return list of (index, shard_path) overlapping [start_ts, end_ts]."""
    items = list(ts_dict.items())  # [(idx, (min,max)), ...] sorted already by construction
    if not items:
        return []
    first_idx = items[0][0]
    last_idx  = items[-1][0]

    start_idx = first_idx
    for idx, (mn, mx) in items:
        if mx >= start_ts:
            start_idx = idx
            break

    end_idx = last_idx
    for idx, (mn, mx) in reversed(items):
        if mn <= end_ts:
            end_idx = idx
            break

    if end_idx < start_idx:
        return []
    return [(i, shards[i]) for i in range(start_idx, end_idx + 1)]

def list_members_in_range_from_selected_shards(selected: List[Tuple[int, Path]], start_ts: int, end_ts: int) -> Tuple[Dict[Path, List[Tuple[int,int,str]]], List[int]]:
    members_index: Dict[Path, List[Tuple[int,int,str]]] = {}
    all_ts: List[int] = []
    for idx, tar_p in selected:
        with tarfile.open(tar_p, "r") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                name = Path(m.name).name
                parsed = parse_fname(name)
                if parsed is None:
                    continue
                ts, fid = parsed
                if start_ts <= ts <= end_ts:
                    members_index.setdefault(tar_p, []).append((ts, fid, m.name))
                    all_ts.append(ts)
    return members_index, all_ts

def write_clip_from_members(shard_paths: List[Path],
                            members_index: Dict[Path, List[Tuple[int,int,str]]],
                            out_path: Path,
                            fps: float) -> int:
    first_img = None
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
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                first_img = img
                break
    if first_img is None:
        return 0
    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter at {out_path}")

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
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if (img.shape[1], img.shape[0]) != (w, h):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            vw.write(img)
            frames_written += 1

    vw.release()
    return frames_written

def read_labels_csv(labels_csv: Path):
    rows = []
    with open(labels_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        header_map = {k: k for k in reader.fieldnames or []}
        def pick(*cands):
            for c in cands:
                if c in header_map:
                    return c
            return None
        col_obj = pick("object__id", "object_id")
        col_st  = pick("start__timestamp", "start_timestamp")
        col_et  = pick("end__timestamp", "end_timestamp")
        col_lab = pick("label",)
        required = [col_st, col_et, col_lab]
        if any(c is None for c in required):
            raise ValueError(f"CSV must contain start/end timestamps and label. Got headers: {reader.fieldnames}")
        for r in reader:
            try:
                start_ts = int(r[col_st])
                end_ts = int(r[col_et]) 
                label   = int(r[col_lab])
            except Exception as e:
                print(f"[WARN] Skipping row due to parse error: {r} ({e})")
                continue
            if end_ts < start_ts:
                start_ts, end_ts = end_ts, start_ts
            start_ts = max(0, start_ts)
            obj_id = r.get(col_obj, "")
            rows.append((start_ts, end_ts, label, obj_id))
    return rows

def process_folder(folder_path):
    # Path Initalization
    folder_path = Path(folder_path)
    intersection_name = folder_path.parts[-2]
    labels_csv = Path(folder_path) / "anomaly-labels.csv"
    object_det_json = Path(folder_path) / "object_detections.json"
    anomaly_event_obj_dict_path = Path(folder_path) / "anomaly_event_obj_dict.json"
    frames_dir = Path(folder_path) / "frames_shards"
    outdir = Path(folder_path) / "videos"
    outdir.mkdir(parents=True, exist_ok=True)
    anomaly_event_obj_dict = {}

    # Read Cache Data
    full_obj_dict = read_obj_detections_json(object_det_json)
    shards = sorted_shards(frames_dir)

    # Precompute shard to get global timestamp ranges
    ts_dict = shard_min_max_timestamps(shards)
    all_mins = [ts_range[0] for ts_range in ts_dict.values()]
    all_maxs = [ts_range[1] for ts_range in ts_dict.values()]
    global_min_ts = min(all_mins) if all_mins else 0
    global_max_ts = max(all_maxs) if all_maxs else float('inf')

    # Read event, each row is one event
    rows = read_labels_csv(labels_csv)

    for idx, (start_ts, end_ts, label, obj_id) in enumerate(rows, 1):
        
        start_ts, end_ts = random_ts_add(start_ts,end_ts, global_min_ts,global_max_ts)
        anomaly_event_obj_dict = record_obj_select(idx,obj_id,anomaly_event_obj_dict,full_obj_dict,start_ts,end_ts)

        severity = LABEL_TO_SEVERITY.get(label, -1)
        out_path = outdir / f"{intersection_name}_av_{idx}_{label}_{severity}.mp4"
        print(f"[{idx}/{len(rows)}] Event label={label}, severity={severity}, ts=[{start_ts},{end_ts}] -> {out_path.name}")

        selected = pick_shards_for_range(shards, ts_dict, start_ts, end_ts)
        if not selected:
            print("  [WARN] No shard candidates for this event. Skipping.")
            continue

        members_index, all_ts = list_members_in_range_from_selected_shards(selected, start_ts, end_ts)
        if not all_ts:
            print("  [WARN] No frames found in range for this event. Skipping.")
            continue

        fps = estimate_fps(all_ts, max_fps=30)
        print(f"  Shards: {len(selected)} | Frames: {sum(len(v) for v in members_index.values())} | Estimated FPS: {fps:.2f}")

        written = write_clip_from_members([p for _, p in selected], members_index, out_path, fps=fps)
        if written == 0:
            print("  [WARN] Wrote 0 frames for event (unexpected).")
        else:
            print(f"  [OK] Wrote {written} frames -> {out_path}")

    # Save anomaly event object dictionary
    save_anomaly_obj_dict(anomaly_event_obj_dict, anomaly_event_obj_dict_path)

if __name__ == "__main__":
    folders = ["./OTA/MononElmStreetNB/testdata"]
    for folder in folders:
        if os.path.exists(folder):
            process_folder(folder)
        else:
            print(f"Folder {folder} does not exist.")
