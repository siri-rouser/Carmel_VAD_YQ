import os, io, cv2, json, time, base64, re
from datetime import datetime
import numpy as np
from PIL import Image
from openai import OpenAI
from openai import RateLimitError, APIStatusError, InternalServerError
from utils.few_shot_example import FewShotExamples
from utils.prompt_base import dev_instruction, assitant_instruction
from string import Template
import tiktoken

# ---- helpers ----
def parse_json_from_text(text: str) -> dict:
    """Best-effort: direct JSON -> fallback to first {...} block."""
    # 1) direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2) extract the first JSON object block (handles extra logs / code fences)
    m = re.search(r'\{.*\}', text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in response text.")
    return json.loads(m.group(0))

def resize_keep_aspect(pil, max_side=512):
    w, h = pil.size
    s = max(w, h)
    if s <= max_side:
        return pil
    scale = max_side / s
    return pil.resize((int(w*scale), int(h*scale)), Image.BILINEAR)

def sample_frames_as_base64(video_path, max_frames=12, max_side=512, jpeg_q=80):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame[:, :, ::-1])  # BGR->RGB
    cap.release()
    if not frames:
        raise RuntimeError("No frames read from video.")

    idxs = np.linspace(0, len(frames)-1, min(max_frames, len(frames)), dtype=int)
    b64_list = []
    for i in idxs:
        pil = Image.fromarray(frames[i])
        pil = resize_keep_aspect(pil, max_side=max_side)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=jpeg_q, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        b64_list.append(f"data:image/jpeg;base64,{b64}")
    return b64_list

def build_messages(prompt, b64_images, instruction_text):
    dev = {
        "role": "developer",
        "content": instruction_text.strip()
    }
    user_content = [{"type": "input_text", "text": prompt}]
    # Responses API: image_url must be a STRING; no extra fields like "detail"
    for data_url in b64_images:
        user_content.append({"type": "input_image", "image_url": data_url})
    user = {"role": "user", "content": user_content}
    return [dev, user]

def build_fewshot_messages(
    instruction_text: str,
    examples: list,
    target_prompt: str,
    target_images_b64: list
):
    """
    Build a few-shot message list:
      [ developer(instruction),
        assistant(assistant-level instruction),
        user(example#1 frames+prompt), assistant(example#1 ideal JSON),
        ...,
        user(target frames+prompt)
      ]
    Each example is a dict:
      {
        "prompt": "short example instruction to the model",
        "video_images_b64": [data_url, ...],   # precomputed
        "answer_json": {...}                   # ideal structured JSON (dict) or str
      }
    """
    messages = [{
        "role": "developer",
        "content": instruction_text.strip()
    }]

    # --- NEW: global assistant message with extra guidance ---
    assistant_text = assitant_instruction().strip()
    if assistant_text:
        messages.append({
            "role": "assistant",
            "content": [{"type": "output_text", "text": assistant_text}]
        })
    # --------------------------------------------------------

    # add exemplars
    for ex in examples:
        # user turn with frames + a terse prompt
        user_content = [{"type": "input_text", "text": ex["prompt"]}]
        for data_url in ex["video_images_b64"]:
            user_content.append({"type": "input_image", "image_url": data_url})
        messages.append({"role": "user", "content": user_content})

        # assistant turn with the ideal gold JSON
        if isinstance(ex["answer_json"], dict):
            gold = json.dumps(ex["answer_json"], ensure_ascii=False)
        else:
            gold = str(ex["answer_json"]).strip()

        messages.append({
            "role": "assistant",
            "content": [{"type": "output_text", "text": gold}]
        })

    # final target query
    final_user_content = [{"type": "input_text", "text": target_prompt}]
    for data_url in target_images_b64:
        final_user_content.append({"type": "input_image", "image_url": data_url})

    messages.append({"role": "user", "content": final_user_content})
    return messages

def call_openai_for_video_fewshot(
    target_prompt: str,
    target_video_path: str,
    examples: list, 
    model="gpt-4.1-mini",
    timeout_s=120,
    target_max_frames=64,
    target_max_side=384
):
    """
    Build few-shot messages and call the Responses API.
    The 'examples' list contains raw video paths and answer_json; we will sample frames here.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY")

    client = OpenAI(api_key=api_key, timeout=timeout_s)

    # Sample frames for each example
    processed_examples = []
    for ex in examples:
        processed_examples.append({
            "prompt": ex.get("prompt", "Please analyze the video and output the JSON per schema."),
            "video_images_b64": ex["video_images_b64"],
            "answer_json": ex["answer_json"]
        })

    # Sample frames for target video
    target_imgs = sample_frames_as_base64(
        target_video_path,
        max_frames=target_max_frames,
        max_side=target_max_side,
        jpeg_q=80
    )

    # Reuse your existing (great) instruction block
    instruction = dev_instruction()

    messages = build_fewshot_messages(
        instruction_text=instruction,
        examples=processed_examples,
        target_prompt=target_prompt,
        target_images_b64=target_imgs
    )

    result = client.responses.create(
        model=model,
        input=messages,
        text={"verbosity": "low"},
    )
    return result.output_text


def call_openai_for_video(prompt, video_path, model="gpt-4.1-mini", timeout_s=60):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY")

    # Set a client timeout so calls can't hang forever
    client = OpenAI(api_key=api_key, timeout=timeout_s)

    images = sample_frames_as_base64(video_path, max_frames=16, max_side=384, jpeg_q=80)

    instruction = """
        # Role: You are a vision annotator that writes neutral, factual, chronological descriptions of traffic anomalies.

            Inputs available
        
                1. Video frames
                
                2. Human labeled context that includes
                    a. Anomaly relevance degree. Value in this case is <moderate>
                    b. Anomaly type. Value in this case is <illegal turn>

        # Task: Write a precise, factual, chronological description of the anomaly event using only what is observable in the frames and the given context. Be concise. You must include

            1. Road layout and surrounding environment. Example: an intersection with busy traffic, a snowy night roundabout, a straight road in an urban area

            2. Road users involved. Example: a white sedan, a red truck, a firefighter

            3. The event sequence in clear steps. Example: road user 1 did action 1. Then road user 1 interacted with road user 2 by doing action 2

            4. A brief analysis explaining why the behavior is anomalous. Example: late braking, violation of traffic rules, creation of a dangerous situation

        # Rules

            1. Use only observable evidence from the frames and the given context.

            2. Keep the timeline strictly chronological.

            3. Use unambiguous terms and consistent nouns.

            4. Limit the combined event description and analysis to fewer than 100 words

            5. choose anomaly relevance from {critical | moderate | low}

            6. Choose anomaly type from the catalog below. Output both the numeric code and the Anomaly type catalog.
                -1 detection or tracking mistake
                0 false positive
                1 change of lane
                2 late turn
                3 cutting inside turns
                4 driving on the centerline
                5 yielding to emergency vehicles
                6 brief wait at an open intersection
                7 long wait at an empty intersection
                8 too far onto the main road while waiting
                9 stopping at an unusual point
                10 slowing at an unusual point
                11 fast driving that appears reckless
                12 slow driving with apparent uncertainty
                13 unusual movement pattern
                14 brief reverse movement
                15 unusual approach toward waiting or slow cars
                16 traffic tie up
                17 almost cut another traffic agent off
                18 cut another traffic agent off clearly
                19 almost collision
                20 into oncoming lane while turning
                21 illegal turn
                22 short wrong way in roundabout then exit
                23 wrong way driver
                24 more than one full turn in a roundabout
                25 broken down vehicle on street
                26 stop mid street to let a person cross
                27 stop at a crosswalk to let a person cross
                28 slight departure from the roadway
                29 on or parking on sidewalk
                30 strong sudden braking
                31 swerve to avoid or maneuver around a vehicle
                32 risky behaviour that does not fit another category

        # Output Format
            Return a single JSON object that follows this schema:
            {
            "anomaly_relevance": "<critical or moderate or low>",
            "anomaly_type_code": <integer from the catalog>,
            "anomaly_type_label": "<label from the catalog>",
            "event_description": "<chronological description. fewer than 100 words total with analysis>",
            "event_analysis": "<brief reason this is anomalous. counted toward the same 100 word limit>"
            }

        # Examples
            {
            "anomaly_relevance": "moderate",
            "anomaly_type_code": 23,
            "anomaly_type_label": "wrong way driver",
            "event_description": "On a busy roundabout, a white SUV exits through the entry lane and proceeds against traffic.",
            "event_analysis": "The driver travels opposite the permitted direction, which violates traffic rules and creates a collision risk."
            }

            {
            "anomaly_relevance": "moderate",
            "anomaly_type_code": 18,
            "anomaly_type_label": "cut another traffic agent off clearly",
            "event_description": "On a snowy roundabout, a silver sedan enters without yielding and crosses ahead of a blue hatchback. The hatchback brakes hard to avoid a collision.",
            "event_analysis": "The silver sedan braked too late or partially lost control and entered the roundabout without yielding, forcing the other driver to brake suddenly."
            }

            {
            "anomaly_relevance": "moderate",
            "anomaly_type_code": 9,
            "anomaly_type_label": "stopping at an unusual point",
            "event_description": "At a four-way intersection, a red SUV stops within the entry lane while adjacent lanes continue moving.",
            "event_analysis": "Stopping at a non-designated position inside the approach lane disrupts traffic flow and increases rear-end risk."
            }

            {
            "anomaly_relevance": "low",
            "anomaly_type_code": 24,
            "anomaly_type_label": "more than one full turn in a roundabout",
            "event_description": "At a sunny single-lane roundabout, a white sedan circles once, continues for half a lap, then exits.",
            "event_analysis": "Extra circling can confuse following drivers; no conflicts are observed."
            }

            {
            "anomaly_relevance": "critical",
            "anomaly_type_code": 32,
            "anomaly_type_label": "risky behaviour that does not fit to another category",
            "event_description": "During night at an urban multiway intersection, a white sedan enters at visibly high speed, continues straight, and collides with another vehicle that is crossing the junction. Debris is visible and both vehicles come to a stop within the intersection.",
            "event_analysis": "The driver of white sedan appears to have lost control or is driving recklessly, leading to a serious collision in a high-risk area."
            }
    """

    messages = build_messages(prompt, images, instruction)

    # stream so you see partial tokens immediately
    result = client.responses.create(
        model=model,
        input=messages,
        text={ "verbosity": "low" },
    ) 
    return result.output_text


def save_output_to_json(prompt, video_path, response, model_used, output_file):
    data = {
        "timestamp": datetime.now().isoformat(),
        "video_path": str(video_path),
        "prompt": prompt,
        "model": model_used,
        "response": response,
        "metadata": {
            "video_exists": os.path.exists(video_path),
            "video_size_mb": round(os.path.getsize(video_path)/(1024*1024), 2) if os.path.exists(video_path) else None
        }
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nOutput saved to: {output_file}")

if __name__ == "__main__":
    
    few_shot_example = FewShotExamples()
    examples = few_shot_example.call_MononElmStreetNB_example()
    
    video_path = "./OTA/RangelineSMedicalDr/testdata/videos/RangelineSMedicalDr_av_258_18_3.mp4"

    start_bbox = [
                0.1328,
                0.3005,
                0.1536,
                0.3338
            ]

    end_bbox = [
                0.8917,
                0.4819,
                0.9847,
                0.5655
            ]
    
    event_time = [6,27.248]
    
    cat = 18
    severity = 3
    video_length = 28.446

    prompt_tmp = Template("""
    Review the raw video and the Context. Follow the description and analysis style in examples, return one JSON object that matches the schema pre-defined. Use only what is visible in the frames and what is given in Context. Keep present tense. Use short sentences. No extra text outside JSON.

    Context:                  
        • Human labeled anomaly relevance is ${severity}.
        • Human labeled anomaly type is ${cat}.
        • Subject start box in frame is [${start_bbox}].
        • Subject end box in frame is [${end_bbox}].
        • Event start from ${event_start} seconds and end at ${event_end} seconds, video length is ${video_length} seconds.
    """)

    prompt = prompt_tmp.substitute(
        severity=severity,
        cat=cat,
        video_length=video_length,
        start_bbox=", ".join([f"{x:.4f}" for x in start_bbox]),
        end_bbox=", ".join([f"{x:.4f}" for x in end_bbox]),
        event_start=event_time[0],
        event_end=event_time[1]
    )
                      
    # Pick a model you’re entitled to; try this first:
    model = "gpt-5-2025-08-07"

    response_text = call_openai_for_video_fewshot(prompt, video_path, examples=examples, model=model, timeout_s=180)
    out_path = f"anomaly_detection_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    print("Response from model:")
    print(response_text)
    print("-" * 50)

    # save_output_to_json(prompt, video_path, response_text, model, out_path)
    # Normalize to a Python dict, then save as real JSON
    if isinstance(response_text, dict):
        data = response_text
    else:
        data = parse_json_from_text(response_text)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved structured JSON to: {out_path}")