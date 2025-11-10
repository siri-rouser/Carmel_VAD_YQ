"""
Purpose of this file is to update gpt_inference_results.json with refined_conversations files.
"""
import json
import os
import logging
from pathlib import Path

import re
from QA_pair_database import QA_pair_database

logging.basicConfig(level=logging.INFO)

REFINED_CONV_PATTERN = re.compile(
    r"^refined_no(?P<idx>\d+)_cat(?P<label>\d+)_sev(?P<severity>\d+)_downsize\.json$"
)
CONV_PATTERN= re.compile(
    r"^no(?P<idx>\d+)_cat(?P<label>\d+)_sev(?P<severity>\d+)_downsize\.json$"
)
VIDEO_PATTERN = re.compile(
    r"^no(?P<idx>\d+)_cat(?P<label>\d+)_sev(?P<severity>\d+)_downsize\.mp4$"
)

language_db = QA_pair_database()

def process_folder(folder_path):
    gpt_inference_results = folder_path / "gpt_inference_results.json"
    refined_conversations_dir = folder_path / "refined_conversations"
    conversation_dir = folder_path / "conversations"

    # Step 1: Update gpt_inference_results.json using conversation files
    for conv_file in conversation_dir.glob("*.json"):
        with open(conv_file, "r") as f:
            conv_data = json.load(f)
            anomaly_type_label, severity, type_label, event_analysis, event_description = conv_data.get("anomaly_type_label", ""), conv_data.get("anomaly_relevance", ""), conv_data.get("anomaly_type_label", ""), conv_data.get("event_analysis", ""), conv_data.get("event_description", "")
            match = CONV_PATTERN.match(conv_file.name)
            if match:
                idx = match.group("idx")
                with open(gpt_inference_results, "r") as gpt_file:
                    gpt_data = [json.loads(line) for line in gpt_file if line.strip()]
                    
                    for item in gpt_data:
                        video_path = Path(item["video"])
                        file_name = video_path.name
                        match2 = VIDEO_PATTERN.match(file_name)
                        if match2 and match2.group("idx") == idx:
                            question = item["conversations"][0]["value"].split("\n")[1]
                            question_type = language_db.question_type_query(question)

                            if question_type == "description":
                                item["conversations"][1]["value"] = event_description
                                logging.info(f"Updated description for video no{idx}_")
                            elif question_type == "severity":
                                item["conversations"][1]["value"] = severity
                                logging.info(f"Updated severity for video no{idx}_")
                            elif question_type == "category":
                                item["conversations"][1]["value"] = anomaly_type_label
                                logging.info(f"Updated category for video no{idx}_")
                            elif question_type == "analysis":
                                item["conversations"][1]["value"] = event_analysis
                                logging.info(f"Updated analysis for video no{idx}_")
                            else:
                                logging.warning(f"Question:{question} | Unknown question type for video no{idx}_")
                    
                # write back as jsonl (one JSON object per line)
                gpt_file.close()
                with open(gpt_inference_results, "w", encoding="utf-8") as gpt_out:
                    for item in gpt_data:
                        gpt_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    # # Step 2: Update refined data
    for refined_file in refined_conversations_dir.glob("*.json"):
        with open(refined_file, "r") as f:
            refined_data = json.load(f)
            anomaly_type_label, severity, type_label, event_analysis, event_description = refined_data.get("anomaly_type_label", ""), refined_data.get("anomaly_relevance", ""), refined_data.get("anomaly_type_label", ""), refined_data.get("event_analysis", ""), refined_data.get("event_description", "")
            match = REFINED_CONV_PATTERN.match(refined_file.name)
            if match:
                idx = match.group("idx")

                with open(gpt_inference_results, "r") as gpt_file:
                    gpt_data = [json.loads(line) for line in gpt_file if line.strip()]
                    for item in gpt_data:
                        video_path = Path(item["video"])
                        file_name = video_path.name
                        match2 = VIDEO_PATTERN.match(file_name)
                        if match2 and match2.group("idx") == idx:
                            question = item["conversations"][0]["value"].split("\n")[1]
                            question_type = language_db.question_type_query(question)

                            if question_type == "description":
                                item["conversations"][1]["value"] = event_description
                                logging.info(f"Updated description for video no{idx}_")
                            elif question_type == "severity":
                                item["conversations"][1]["value"] = severity
                                logging.info(f"Updated severity for video no{idx}_")
                            elif question_type == "category":
                                item["conversations"][1]["value"] = anomaly_type_label
                                logging.info(f"Updated category for video no{idx}_")
                            elif question_type == "analysis":
                                item["conversations"][1]["value"] = event_analysis
                                logging.info(f"Updated analysis for video no{idx}_")
                            else:
                                logging.warning(f"Question:{question} | Unknown question type for video no{idx}_")
                    
                # write back as jsonl (one JSON object per line)
                gpt_file.close()
                with open(gpt_inference_results, "w", encoding="utf-8") as gpt_out:
                    for item in gpt_data:
                        gpt_out.write(json.dumps(item, ensure_ascii=False) + "\n")

            # Process the refined data as needed

if __name__ == "__main__":
    folders = [Path("./carmel_data/MedicalDrive-Rangeline-midres"), Path("./carmel_data/RangelineCityCenterSB-midres")]
    for folder in folders:
        if folder.exists():
            process_folder(folder)
        else:
            print(f"Folder {folder} does not exist.")