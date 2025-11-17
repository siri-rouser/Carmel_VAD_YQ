#!/usr/bin/env python3
"""
Convert Carmel VAD rewritten annotations into InternVL video chat format.

Input format (carmel_vad_selected_rewritten.json):
[
  {
    "conversations": [
      {"from": "human", "value": "<video>\\n...desc question..."},
      {"from": "gpt",   "value": "...desc answer..."}
    ],
    "video": "OTA/.../MononElmStreetNB_av_165_28_3.mp4"
  },
  {
    "conversations": [
      {"from": "human", "value": "<video>\\n...category question..."},
      {"from": "gpt",   "value": "speed_trajectory_irregularities"}
    ],
    "video": "OTA/.../MononElmStreetNB_av_165_28_3.mp4"
  },
  ...
]

Output format (JSONL for InternVL video data):
{"id": 0, "video": "...mp4", "conversations": [ ...multi-round dialogue... ]}
{"id": 1, "video": "...mp4", "conversations": [ ... ]}
...
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path


def strip_video_tag(text: str) -> str:
    """
    Remove the first '<video>' placeholder (and an immediate newline, if any)
    from a human turn. Used for *later* rounds so that <video> appears only once
    per sample, as recommended by InternVL.
    """
    if "<video>" not in text:
        return text

    # Only strip the *first* occurrence
    idx = text.find("<video>")
    before = text[:idx]
    after = text[idx + len("<video>") :]

    # Optionally remove a single leading newline after <video>
    if after.startswith("\n"):
        after = after[1:]

    return before + after


def convert_to_internvl_video(
    input_path: Path,
    output_path: Path,
    keep_video_tag_only_first_turn: bool = True,
) -> None:
    # Load the original JSON (list of entries)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Group by video while preserving order of first appearance
    grouped = OrderedDict()  # video_path -> {"video": ..., "conversations": [...]}

    for item in data:
        video_path = item["video"]
        convs = item["conversations"]

        if video_path not in grouped:
            grouped[video_path] = {
                "video": video_path,
                "conversations": []
            }

        group = grouped[video_path]
        # If this is the first time we see this video, we keep its human
        # turn exactly as-is (including <video>).
        # For subsequent human turns, we optionally strip the <video> tag.
        first_turn_for_this_video = len(group["conversations"]) == 0

        for turn in convs:
            turn_copy = dict(turn)
            if (
                keep_video_tag_only_first_turn
                and turn_copy.get("from") == "human"
                and not first_turn_for_this_video
            ):
                turn_copy["value"] = strip_video_tag(turn_copy["value"])

            group["conversations"].append(turn_copy)
            # After we add the very first turn, subsequent ones are no longer "first"
            first_turn_for_this_video = False

    # Assign IDs and write JSONL
    with output_path.open("w", encoding="utf-8") as f_out:
        for idx, (video_path, entry) in enumerate(grouped.items()):
            sample = {
                "id": idx,
                "video": entry["video"],
                "conversations": entry["conversations"],
            }
            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Converted {len(data)} original entries "
          f"into {len(grouped)} video samples.")
    print(f"Output written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Carmel VAD annotations to InternVL video chat format."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("carmel_vad_selected_rewritten.json"),
        help="Path to the input JSON file (list of entries).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("carmel_vad_internvl_video_chat.jsonl"),
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--keep-video-tag-only-first",
        action="store_true",
        help=(
            "Ensure <video> appears only in the first human turn per video "
            "by stripping it from later turns."
        ),
    )

    args = parser.parse_args()

    convert_to_internvl_video(
        input_path=args.input,
        output_path=args.output,
        keep_video_tag_only_first_turn=args.keep_video_tag_only_first,
    )


if __name__ == "__main__":
    main()
