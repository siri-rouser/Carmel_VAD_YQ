import os
from pathlib import Path
import re
import json
from typing import Dict, Any, Optional, List
from API_inference_GPT_bak import call_openai_for_video
from utils.QA_pair_database import QA_pair_database

FILENAME_PATTERN = re.compile(
    r"^no(?P<idx>\d+)_cat(?P<label>\d+)_sev(?P<severity>\d+)_downsize\.mp4$"
)

language_db = QA_pair_database()

def make_qwen_samples_for_video(
    video_path: Path,
    parsed: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Convert one parsed response JSON to multiple Qwen3-VL items (one question per sample).
    """
    samples: List[Dict[str, Any]] = []

    # Pull fields (with light normalization)
    anomaly_desc = safe_get(parsed, "event_description")
    severity = safe_get(parsed, "anomaly_relevance")
    type_label = safe_get(parsed, "anomaly_type_label")
    type_code = safe_get(parsed, "anomaly_type_code")
    analysis = safe_get(parsed, "event_analysis")


    # --- Q1: existence & description ---
    if anomaly_desc:
        description_question = language_db.question_selection("description")
        samples.append({
            "conversations": [
                {"from": "human", "value": f"<video>\n{description_question}"},
                {"from": "gpt", "value": anomaly_desc}
            ],
            "video": video_path.as_posix()
        })

    # --- Q2: severity ---
    if severity:
        severity_question = language_db.question_selection("severity")
        samples.append({
            "conversations": [
                {"from": "human", "value": f"<video>\n{severity_question}"},
                {"from": "gpt", "value": severity}
            ],
            "video": video_path.as_posix()
        })

    # --- Q3: category ---
    if type_label:
        category = language_db.question_selection("category")
        samples.append({
            "conversations": [
                {"from": "human", "value": f"<video>\n{category}"},
                {"from": "gpt", "value": type_label}
            ],
            "video": video_path.as_posix()
        })

    # --- Q4: cause / basis ---
    if analysis:
        analysis_question = language_db.question_selection("analysis")
        samples.append({
            "conversations": [
                {"from": "human", "value": f"<video>\n{analysis_question}"},
                {"from": "gpt", "value": analysis}
            ],
            "video": video_path.as_posix()
        })

    # If model said "no anomaly", you might get none of the above fields—emit a fallback minimal sample:
    if not samples:
        fallback_ans = "No anomaly detected."
        samples.append({
            "conversations": [
                {"from": "human", "value": "<video>\nIs there any anomaly in this video? If yes, describe the event."},
                {"from": "gpt", "value": fallback_ans}
            ],
            "video": video_path.as_posix()
        })

    return samples

def safe_get(d: Dict[str, Any], key: str) -> Optional[Any]:
    v = d.get(key, None)
    if isinstance(v, str):
        return v.strip()
    return v


def extract_first_json_obj(text: str) -> str:
    """
    Extracts the first {...} JSON object block from a messy string (handles code fences / logs).
    Uses a simple brace counter to avoid over-greedy regex pitfalls.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No '{' found in response text.")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    raise ValueError("Braces appear unbalanced; could not isolate a JSON object.")

def parse_json_from_text(text: str) -> Dict[str, Any]:
    """
    Best-effort JSON parse: direct -> fenced/extra text -> helpful error.
    """
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting the first balanced JSON block
    cleaned = extract_first_json_obj(text)
    return json.loads(cleaned)


def process_folder(folder_path):
    folder_path = Path(folder_path)
    json_output_path = folder_path / "gpt_inference_results.json"
    single_conversation_output_path = folder_path / "conversations"
    os.makedirs(single_conversation_output_path, exist_ok=True)

    count = 0
    with open(json_output_path, "a", encoding="utf-8") as fout:
        for file_path in folder_path.iterdir():
            if file_path.is_file() and file_path.suffix == ".mp4":
                match = FILENAME_PATTERN.match(file_path.name)
                if match:
                    label = match.group("label")
                    severity = match.group("severity")
                    if severity == "-1" or severity == "0":
                        print(f"[SKIP] File {file_path.name} has severity -1, skipping.")
                        continue
                    text_response = call_openai_for_video(file_path, severity, label, model="gpt-5-2025-08-07", timeout_s=180)

                try:
                    parsed = parse_json_from_text(text_response)
                except Exception as e:
                    print(f"[ERROR] JSON parse failed for {file_path.name}: {e}")
                    # Optional: save the raw text to inspect later
                    raw_dump = (folder / f"{file_path.stem}_raw_response.txt")
                    try:
                        raw_dump.write_text(text_response, encoding="utf-8")
                    except Exception:
                        pass
                    continue

                # Save each conversation turn as a separate JSON file
 
                convo_file_path = single_conversation_output_path / f"{file_path.stem}.json"
                with open(convo_file_path, "w", encoding="utf-8") as convo_fout:
                    json.dump(parsed, convo_fout, indent=2, ensure_ascii=False)

                # Convert to Qwen-style training samples
                samples = make_qwen_samples_for_video(file_path, parsed)
                for sample in samples:
                    fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count += 1
                print(f"[{count}] Processed {file_path.name}, generated {len(samples)} samples.")


if __name__ == "__main__":
    folders = ["./carmel_data/MedicalDrive-Rangeline-midres","./carmel_data/RangelineCityCenterSB-midres"]

    for folder in folders:
        if os.path.exists(folder):
            process_folder(folder)
        else:
            print(f"Folder {folder} does not exist.")