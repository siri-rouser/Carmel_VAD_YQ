from pathlib import Path
import json
from utils.QA_pair_database import QA_pair_database

FINE_TO_MACRO_ID = {
    # 1) speed_trajectory_irregularities
    11: 1,
    12: 1,
    13: 1,
    14: 1,
    28: 1,
    32: 1,

    # 2) direction_space_violations
    20: 2,
    21: 2,
    22: 2,
    23: 2,
    29: 2,

    # 3) conflict_near_collision
    18: 3,
    19: 3,
    30: 3,
    31: 3,

    # 4) stopped_obstruction_right_of_way
    5: 4,
    25: 4,
    26: 4,
    27: 4,
}

MACRO_ID_TO_NAME = {
    1: "speed_trajectory_irregularities",
    2: "direction_space_violations",
    3: "conflict_near_collision",
    4: "stopped_obstruction_right_of_way",
}

def map_32_to_4(cat_no_str: str):
    """
    cat_no_str: string like "23"
    returns (macro_id:int or None, macro_name:str or None)
    """
    try:
        fine_id = int(cat_no_str)
    except (TypeError, ValueError):
        return None, None

    macro_id = FINE_TO_MACRO_ID.get(fine_id)
    if macro_id is None:
        return None, None
    return macro_id, MACRO_ID_TO_NAME[macro_id]

QA_pair_database = QA_pair_database()

def cat_conv(item):
    question = item["conversations"][0]["value"].split("\n")[1]
    gpt_answer = item["conversations"][1]["value"]
    question_type = QA_pair_database.question_type_query(question)

    if question_type == "category":
        cat_no = QA_pair_database.category_to_index(gpt_answer)
        macro_id, macro_name = map_32_to_4(cat_no)

        if macro_id is not None:
            item["conversations"][1]["value"] = macro_name
        
    return item

def process_file(file_path):
    in_path = Path(file_path)
    out_path = in_path.with_name(in_path.stem + "_4cat" + in_path.suffix)
    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        gpt_data = [json.loads(line) for line in f_in if line.strip()]
        for item in gpt_data:
            item = cat_conv(item)

            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # json_files = ["./OTA/RangelineS116thSt/testdata_selected/gpt_inference_results.json","./OTA/MononElmStreetNB/testdata_selected/gpt_inference_results.json", "./OTA/RangelineSMedicalDr/testdata_selected/gpt_inference_results.json"]
    json_files = ["./carmel_data/MedicalDrive-Rangeline-midres/gpt_inference_results.json","./carmel_data/RangelineCityCenterSB-midres/gpt_inference_results.json"]
    for file_path in json_files:
        process_file(file_path)
        print(f"Processed file: {file_path}")