"""
Aim of this module is to get the statistics results of 4 categories:
    1: "speed_trajectory_irregularities"
    2: "direction_space_violations"
    3: "conflict_near_collision"
    4: "stopped_obstruction_right_of_way"
"""
import argparse
import json
from pathlib import Path
from collections import Counter
from utils.QA_pair_database import QA_pair_database
QAB = QA_pair_database()

def cat_count(args):
    input_path = args.input_path

    category_counter = Counter()

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            # Extract the category from the last GPT response
            question = item["conversations"][0]["value"].split("\n")[1]
            if QAB.question_type_query(question) == "category" or QAB.question_type_query(question) == "category_new":
                gpt_answer = item["conversations"][-1]["value"]
                if gpt_answer is not None:
                    category_counter[gpt_answer] += 1
                category_counter[gpt_answer] += 1
                
    print("Category Statistics:")
    total = sum(category_counter.values())
    for category, count in category_counter.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{category}: {count} ({percentage:.2f}%)")

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
    args = parser.parse_args()

    cat_count(args)