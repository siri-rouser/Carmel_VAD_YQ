"""
Aim of this module is to select description questions from the dataset for labelling.
"""
import argparse
import json
from pathlib import Path
from collections import Counter
from utils.QA_pair_database import QA_pair_database
QAB = QA_pair_database()

def cat_selector(args):
    input_path = args.input_path
    out_path = args.out_path

    with open(input_path, "r", encoding="utf-8") as f_in:
        data = json.load(f_in)
        output_data = []
        for item in data:
            # Extract the category from the last GPT response
            question = item["conversations"][0]["value"].split("\n")[1]
            if QAB.question_type_query(question) == "description":
                if args.add_sys:
                    item["conversations"][0]["value"] = item["conversations"][0]["value"].split("\n")[0] + QAB.four_category_context + item["conversations"][0]["value"].split("\n")[1]
                output_data.append(item)
    
    with open(out_path, "w", encoding="utf-8") as f_out:
        json.dump(output_data, f_out, ensure_ascii=False, indent=2)
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Get category statistics from InternVL formatted JSONL file."
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="Path to the JSONL file.",
    )

    parser.add_argument(
        "--out_path",
        type=Path,
        required=True,
        help="Path to the JSONL file.",
    )
    parser.add_argument(
        "--add_sys",
        action="store_true",
        help="Add system prompt flag.",
    )
    args = parser.parse_args()

    cat_selector(args)