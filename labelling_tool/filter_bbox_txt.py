"""
This script filters out entries related to severity questions from JSON files.
"""

from pathlib import Path
import json
from utils.QA_pair_database import QA_pair_database

QA_pair_database = QA_pair_database()

def filter_sev(item):
    video_path = Path(item["video"])
    file_name = video_path.name
    question = item["conversations"][0]["value"].split("\n")[1]
    gpt_answer = item["conversations"][1]["value"]
    question_type = QA_pair_database.question_type_query(question)

    if ""
        
    return item

def process_file(file_path):
    in_path = Path(file_path)
    out_path = in_path.parent / f"processed_gpt_inference_results_final.json"
    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        gpt_data = [json.loads(line) for line in f_in if line.strip()]
        for item in gpt_data:
            item = filter_sev(item)
            if item is not None:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    json_files = ["./OTA/MononElmStreetNB/testdata_selected/processed_gpt_inference_results.json", "./OTA/RangelineS116thSt/testdata_selected/processed_gpt_inference_results.json", "./OTA/RangelineSMedicalDr/testdata_selected/processed_gpt_inference_results.json"]

    for file_path in json_files:
        process_file(file_path)