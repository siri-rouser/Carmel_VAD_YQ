"""
This script is to convert raw GPT inference results to a format suitable for further processing.
"""
import json
from pathlib import Path
from typing import List, Dict, Any
from utils.QA_pair_database import QA_pair_database
from labelling_tool.video_label_Monon import parse_json_from_text
from labelling_tool.video_label_Monon import make_qwen_samples_for_video


def make_sample(
    video_path: Path,
    anomaly_label: str,
    anomaly_code: int,
    anomaly_rel: str,
    event_desc: str,
    event_analysis: str
) -> List[Dict[str, Any]]:
    
    QA_base = QA_pair_database()
    samples: List[Dict[str, Any]] = []

    # --- Q1: existence & description ---
    if event_desc:
        description_question = QA_base.question_selection("description")
        samples.append({
            "conversations": [
                {"from": "human", "value": f"<video>\n{description_question}"},
                {"from": "gpt", "value": event_desc}
            ],
            "video": video_path.as_posix()
        })

    # --- Q2: severity ---
    if anomaly_rel:
        severity_question = QA_base.question_selection("severity")
        samples.append({
            "conversations": [
                {"from": "human", "value": f"<video>\n{severity_question}"},
                {"from": "gpt", "value": anomaly_rel}
            ],
            "video": video_path.as_posix()
        })

    # --- Q3: category ---
    if anomaly_label:
        category = QA_base.question_selection("category_new")
        samples.append({
            "conversations": [
                {"from": "human", "value": f"<video>\n{category}"},
                {"from": "gpt", "value": anomaly_label}
            ],
            "video": video_path.as_posix()
        })

    # --- Q4: cause / basis ---
    if event_analysis:
        analysis_question = QA_base.question_selection("analysis")
        samples.append({
            "conversations": [
                {"from": "human", "value": f"<video>\n{analysis_question}"},
                {"from": "gpt", "value": event_analysis}
            ],
            "video": video_path.as_posix()
        })
    
    return samples


def process_file(file_path):
    in_path = Path(file_path)
    out_path = in_path.parent / "gpt_inference_results.json"
    count = 0
    print(f"Processing file: {in_path} -> {out_path}")

    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                print("Skipping empty line")
                continue  # skip empty lines

            # 1) Parse the outer JSON line
            item = json.loads(line)

            video_path = item.get("video_file", "")
            response_str = item.get("response", "")

            parsed = parse_json_from_text(response_str)

            # Convert to Qwen-style training samples
            samples = make_qwen_samples_for_video(video_path, parsed)
            for sample in samples:
                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
        
        print(f"[{count}] Processed {file_path}, generated {len(samples)} samples.")

if __name__ == "__main__":
    json_files = ["./OTA/RangelineS116thSt/testdata_selected/gpt_raw_responses.json","./OTA/MononElmStreetNB/testdata_selected/gpt_raw_responses.json","./OTA/RangelineSMedicalDr/testdata_selected/gpt_raw_responses.json"]
    # json_files = ["./carmel_data/MedicalDrive-Rangeline-midres/gpt_inference_results.json","./carmel_data/RangelineCityCenterSB-midres/gpt_inference_results.json"]
    for file_path in json_files:
        process_file(file_path)