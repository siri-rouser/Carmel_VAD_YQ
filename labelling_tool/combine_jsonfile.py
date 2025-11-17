#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, List, Dict, Union

def load_any_json(path: Path) -> List[Dict[str, Any]]:
    """
    Load a file that might be:
      - JSONL (newline-delimited JSON objects)
      - A JSON array
      - A single JSON object
    Returns a list of dicts.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise ValueError(f"Unsupported JSON root type in {path}: {type(data)}")
    except json.JSONDecodeError:
        # Fallback to JSONL
        items = []
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON on line {lineno} of {path}: {e}") from e
                if not isinstance(obj, dict):
                    raise ValueError(f"Line {lineno} in {path} is not a JSON object.")
                items.append(obj)
        return items

def dedupe(items: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    if not key:
        return items
    seen = set()
    result = []
    for it in items:
        k = it.get(key)
        if k is None:
            # If key missing, keep it (or you could skip)
            result.append(it)
            continue
        if k in seen:
            continue
        seen.add(k)
        result.append(it)
    return result

def save_json_array(path: Path, items: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def save_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Merge multiple GPT inference JSON/JSONL files into one.")
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[
            "./OTA/MononElmStreetNB/testdata_selected/processed_gpt_inference_results.json",
            "./OTA/RangelineSMedicalDr/testdata_selected/processed_gpt_inference_results.json",
            "./OTA/RangelineS116thSt/testdata_selected/processed_gpt_inference_results.json",
        ],
        help="Input files (JSON or JSONL). Defaults to your three paths."
    )
    parser.add_argument(
        "-o", "--output",
        default="combined_processed_gpt_inference_results_testset.json",
        help="Output file path. Defaults to combined_processed_gpt_inference_results.json"
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Write output as JSONL instead of a JSON array."
    )
    args = parser.parse_args()

    all_items: List[Dict[str, Any]] = []
    for p in args.inputs:
        path = Path(p)
        if not path.exists():
            print(f"[WARN] Skipping missing file: {path}")
            continue
        loaded = load_any_json(path)
        all_items.extend(loaded)
        print(f"[INFO] Loaded {len(loaded)} records from {path}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.jsonl:
        save_jsonl(out_path, all_items)
        print(f"[INFO] Wrote {len(all_items)} records to JSONL: {out_path}")
    else:
        save_json_array(out_path, all_items)
        print(f"[INFO] Wrote {len(all_items)} records to JSON array: {out_path}")

if __name__ == "__main__":
    main()
