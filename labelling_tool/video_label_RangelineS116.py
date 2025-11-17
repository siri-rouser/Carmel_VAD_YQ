import os
from pathlib import Path
import re
import json
from typing import Dict, Any, Optional, List

from click import prompt
from quickstart_API_inference_GPT import call_openai_for_video_fewshot
from utils.QA_pair_database import QA_pair_database
from string import Template
from utils.few_shot_example import FewShotExamples

FILENAME_PATTERN = re.compile(r"^(?P<intersection>.+)_av_(?P<idx>\d+)_(?P<label>[^_]+)_(?P<severity>[^_]+)\.mp4$")

def make_prompt(file_path: Path, prompt_tmp: Template) -> str:
    match = FILENAME_PATTERN.match(file_path.name)
    if match:
        idx = match.group("idx")
        label = match.group("label")
        severity = match.group("severity")
        if severity == "-1" or severity == "0":
            print(f"[SKIP] File {file_path.name} has severity -1, skipping.")
            return None

        anomaly_event_path = file_path.parent.parent / "anomaly_event_obj_dict.json"
        with open(anomaly_event_path, "r", encoding="utf-8") as fin:
            anomaly_event_dict = json.load(fin)
        event_info = anomaly_event_dict.get(str(idx), {})
        sha_id = list(event_info.keys())[0]
        if event_info:
            start_bbox = event_info[sha_id].get("start_bbox", [])
            end_bbox = event_info[sha_id].get("end_bbox", [])
            start_time = event_info[sha_id].get("start_time", 0)
            end_time = event_info[sha_id].get("end_time", 1)
            video_length = event_info[sha_id].get("video_length", 0)

            prompt = prompt_tmp.substitute(
                severity=severity,
                cat=label,
                start_bbox=",".join([f"{x:.4f}" for x in start_bbox]),
                end_bbox=",".join([f"{x:.4f}" for x in end_bbox]),
                event_start=f"{start_time:.3f}",
                event_end=f"{end_time:.3f}",
                video_length=f"{video_length:.3f}"
            )
            return prompt, video_length
    return None

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

    QA_base = QA_pair_database()


    # --- Q1: existence & description ---
    if anomaly_desc:
        description_question = QA_base.question_selection("description")
        samples.append({
            "conversations": [
                {"from": "human", "value": f"<video>\n{description_question}"},
                {"from": "gpt", "value": anomaly_desc}
            ],
            "video": video_path.as_posix()
        })

    # --- Q2: severity ---
    if severity:
        severity_question = QA_base.question_selection("severity")
        samples.append({
            "conversations": [
                {"from": "human", "value": f"<video>\n{severity_question}"},
                {"from": "gpt", "value": severity}
            ],
            "video": video_path.as_posix()
        })

    # --- Q3: category ---
    if type_label:
        category = QA_base.question_selection("category")
        samples.append({
            "conversations": [
                {"from": "human", "value": f"<video>\n{category}"},
                {"from": "gpt", "value": type_label}
            ],
            "video": video_path.as_posix()
        })

    # --- Q4: cause / basis ---
    if analysis:
        analysis_question = QA_base.question_selection("analysis")
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
    # Initalization
    model = "gpt-5-2025-08-07"
    prompt_tmp = Template("""
    Review the raw video and the Context. Follow the description and analysis style in examples, return one JSON object that matches the schema pre-defined. Use only what is visible in the frames and what is given in Context. Keep present tense. Use short sentences. No extra text outside JSON.

    Context:                  
        • Human labeled anomaly relevance is ${severity}.
        • Human labeled anomaly type is ${cat}.
        • Subject start box in frame is [${start_bbox}].
        • Subject end box in frame is [${end_bbox}].
        • Event start from ${event_start} seconds and end at ${event_end} seconds, video length is ${video_length} seconds.
        • The road layout is a double-lane roundabout.
    """)

    few_shot_example = FewShotExamples()
    examples = few_shot_example.call_RangelineS116thSt_example()

    count = 0
    folder_path = Path(folder_path)
    json_output_path = folder_path.parent / "gpt_inference_results.json"
    raw_json_output_path = folder_path.parent / "gpt_raw_responses.json"
    with open(json_output_path, "w", encoding="utf-8") as fout:
        files = sorted([f for f in folder_path.iterdir() if f.is_file() and f.suffix == ".mp4"])
        for file_path in files:
            if file_path.is_file() and file_path.suffix == ".mp4":
                prompt, video_length = make_prompt(file_path, prompt_tmp)
                if prompt is None:
                    continue
                if video_length > 60:
                    continue

                target_max_frames = min(int(video_length * 3), 96)
                text_response = call_openai_for_video_fewshot(prompt, file_path, examples=examples, model=model, target_max_frames=target_max_frames, timeout_s=180)

                parsed = parse_json_from_text(text_response)

                with open(raw_json_output_path, "a", encoding="utf-8") as fraw:
                    raw_record = {
                        "video_file": file_path.name,
                        "response": text_response
                    }
                    fraw.write(json.dumps(raw_record, ensure_ascii=False) + "\n")

                # Convert to Qwen-style training samples
                samples = make_qwen_samples_for_video(file_path, parsed)
                for sample in samples:
                    fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                print(f"[{count}] Processed {file_path.name}, generated {len(samples)} samples.")

if __name__ == "__main__":
    folders = ["./OTA/RangelineS116thSt/testdata_selected/videos"]

    for folder in folders:
        if os.path.exists(folder):
            process_folder(folder)
        else:
            print(f"Folder {folder} does not exist.")