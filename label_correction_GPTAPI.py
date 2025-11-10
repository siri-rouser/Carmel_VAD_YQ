import os, io, cv2, json, time, base64, re
from datetime import datetime
from string import Template
import numpy as np
from PIL import Image
from pathlib import Path
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

def read_previous_response(conversation_path):
    with open(conversation_path, "r", encoding="utf-8") as fin:
        return fin.read()

def build_messages(prompt, b64_images):
    user_content = [{"type": "input_text", "text": prompt}]
    # Responses API: image_url must be a STRING; no extra fields like "detail"
    for data_url in b64_images:
        user_content.append({"type": "input_image", "image_url": data_url})
    user = {"role": "user", "content": user_content}
    return [user]

def call_openai_for_video(prompt, video_path, model="gpt-4.1-mini", timeout_s=60):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY")

    # Set a client timeout so calls can't hang forever
    client = OpenAI(api_key=api_key, timeout=timeout_s)

    images = sample_frames_as_base64(video_path, max_frames=10, max_side=384, jpeg_q=80)
    messages = build_messages(prompt, images)

    # stream so you see partial tokens immediately
    result = client.responses.create(
        model=model,
        input=messages,
    ) 
    return result.output_text

if __name__ == "__main__":

    model = "gpt-5-2025-08-07"
    conversation_directory = Path("./carmel_data/RangelineCityCenterSB-midres/conversations")
    refined_conversation_directory = conversation_directory.parent / "refined_conversations"
    os.makedirs(refined_conversation_directory, exist_ok=True)
    pre_conv = Path("./carmel_data/RangelineCityCenterSB-midres/conversations/no22_cat23_sev4_downsize.json")
    video_path = "./carmel_data/RangelineCityCenterSB-midres/no22_cat23_sev4_downsize.mp4"
    response_text = read_previous_response(pre_conv) # Type: <class 'str'>

    prompt_temp = Template(r"""
    # Role:
    You are an expert language refinement model specialized in describing and analyzing traffic anomalies in video data. Your task is to revise and improve the previous response by strictly incorporating all provided contextual clarifications. Accuracy and factual consistency take highest priority.

    # Input Data:
    ## Previous Response:
    ${prev_resp}
    ## Videos for Context:
    (Images extracted from the video are provided separately; refer to them as needed.)

    ## Video Context Clarifications:
    1. The roundabout in this scene is a **double-lane roundabout**, not a single-lane one.
    2. A silver sedan starts in the center of roundabout but facing opposite to traffic flow; it keeps moving(clockwise) and exit through a entry lane of the roundabout, which is illegal and dangerous.

    # Output Requirements:
    Return a **single JSON object** strictly following this schema:
    {
    "anomaly_relevance": "<integer 0–4, where 0 means no anomaly and 4 means highly relevant>",
    "anomaly_type_code": <integer from the catalog>,
    "anomaly_type_label": "<label from the catalog>",
    "event_description": "<concise, chronological description of the event—fewer than 100 words total, incorporating all contextual clarifications>",
    "event_analysis": "<brief explanation of why this behavior is anomalous—part of the 100-word total limit>"
    }

    # Output Format Enforcement:
    Return **only the JSON object**, with no extra text, comments, or explanations.
    """)

    prompt = prompt_temp.substitute(prev_resp=response_text)

    refined_text = call_openai_for_video(prompt, video_path, model=model, timeout_s=180)

    if isinstance(refined_text, dict):
        data = refined_text 
    else:
        data = parse_json_from_text(refined_text)

    with open(refined_conversation_directory / f"refined_{pre_conv.name}", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved structured JSON to: {refined_conversation_directory / f'refined_{pre_conv.name}'}")