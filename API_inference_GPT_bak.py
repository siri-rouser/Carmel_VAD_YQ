import os, io, cv2, json, time, base64, re
from datetime import datetime
from typing import Dict
import numpy as np
from string import Template
from PIL import Image
from openai import OpenAI
from openai import RateLimitError, APIStatusError, InternalServerError

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

def call_openai_for_video(video_path, severity, anomaly_cat, model="gpt-5-2025-08-07", timeout_s=180):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY")

    # Set a client timeout so calls can't hang forever
    client = OpenAI(api_key=api_key, timeout=timeout_s)

    images = sample_frames_as_base64(video_path, max_frames=16, max_side=512, jpeg_q=80)

    prompt = """
    Please review the video and answer the three items below in English.

        1. Identify the anomaly in the video

        2. Provide a detailed description of the anomaly from start to end

        3. Provide a brief analysis that explains your basis for judging the behavior as an anomaly
    """

    instruction_tmpl = Template(r"""
        # Role: You are a vision annotator that writes neutral, factual, chronological descriptions of traffic anomalies.

            Inputs available
        
                1. Video frames
                
                2. Human labeled context that includes
                    a. Anomaly relevance degree. Value in this case is $severity
                    b. Anomaly category. Value in this case is $anomaly_cat

        # Task: Write a precise, factual, chronological description of the anomaly event using only what is observable in the frames and the given context. Traffic anomalies are observable violations of traffic rules or unsafe behaviors that create safety risk. Rely only on what is visible in the video. Name the road layout, the road users, the interaction, and why the behavior is risky. Be concise. You must include

            1. Road layout and surrounding environment. Example: an intersection with busy traffic, a snowy night roundabout, a straight road in an urban area

            2. Road users involved. Example: a white sedan, a red truck, a firefighter

            3. The event sequence in clear steps. Example: road user 1 did action 1. Then road user 1 interacted with road user 2 by doing action 2

            4. A brief analysis explaining why the behavior is anomalous. Example: late braking, violation of traffic rules, creation of a dangerous situation

        # Rules

            1. Use only observable evidence from the frames and the given context.

            2. Keep the timeline strictly chronological.

            3. Use unambiguous terms and consistent nouns.

            4. Limit the combined event description and analysis to fewer than 100 words

            5. choose anomaly relevance from 0, 1, 2, 3, 4. 0 means low relevance, 4 means critical relevance.

            6. Choose anomaly category from the catalog below. Output both the numeric code and the Anomaly category catalog.
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
            "anomaly_relevance": "<0-4, higher means more critical anomaly>",
            "anomaly_type_code": "<integer from the catalog>",
            "anomaly_type_label": "<label from the catalog>",
            "event_description": "<chronological description. fewer than 50 words>",
            "event_analysis": "<brief reason this is anomalous. fewer than 50 words>"
            }

        # Examples
            {
            "anomaly_relevance": "4",
            "anomaly_type_code": 23,
            "anomaly_type_label": "wrong way driver",
            "event_description": "On a busy roundabout, a white SUV exits through the entry lane and proceeds against traffic.",
            "event_analysis": "The driver travels opposite the permitted direction, which violates traffic rules and creates a collision risk."
            }

            {
            "anomaly_relevance": "3",
            "anomaly_type_code": 18,
            "anomaly_type_label": "cut another traffic agent off clearly",
            "event_description": "On a snowy roundabout, a silver sedan enters without yielding and crosses ahead of a blue hatchback. The hatchback brakes hard to avoid a collision.",
            "event_analysis": "The silver sedan braked too late or partially lost control and entered the roundabout without yielding, forcing the other driver to brake suddenly."
            }

            {
            "anomaly_relevance": "3",
            "anomaly_type_code": 9,
            "anomaly_type_label": "stopping at an unusual point",
            "event_description": "At a four-way intersection, a red SUV stops within the entry lane while adjacent lanes continue moving.",
            "event_analysis": "Stopping at a non-designated position inside the approach lane disrupts traffic flow and increases rear-end risk."
            }

            {
            "anomaly_relevance": "2",
            "anomaly_type_code": 24,
            "anomaly_type_label": "more than one full turn in a roundabout",
            "event_description": "At a sunny single-lane roundabout, a white sedan circles once, continues for half a lap, then exits.",
            "event_analysis": "Extra circling can confuse following drivers; no conflicts are observed."
            }

            {
            "anomaly_relevance": "4",
            "anomaly_type_code": 32,
            "anomaly_type_label": "risky behaviour that does not fit to another category",
            "event_description": "During night at an urban multiway intersection, a white sedan enters at visibly high speed, continues straight, and collides with another vehicle that is crossing the junction. Debris is visible and both vehicles come to a stop within the intersection.",
            "event_analysis": "The driver of white sedan appears to have lost control or is driving recklessly, leading to a serious collision in a high-risk area."
            }
    """)

    instruction = instruction_tmpl.substitute(severity=severity, anomaly_cat=anomaly_cat)
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
