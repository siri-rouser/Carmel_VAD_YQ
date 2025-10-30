import os, io, cv2, json, time, base64, re
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
from PIL import Image

# ---- Google Gemini SDK ----
# pip install --upgrade google-generativeai
from google import genai
from google.genai import types
from google.api_core import exceptions as gapi_exceptions


# ---- helpers (largely mirrored from your original) ----
def parse_json_from_text(text: str) -> dict:
    """Best-effort: direct JSON -> fallback to first {...} block."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
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

def sample_frames_for_gemini(video_path: str, max_frames=16, max_side=384, jpeg_q=80) -> List[Dict[str, Any]]:
    """
    Samples frames from a video and returns a list of PIL Image objects.
    """
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
    image_list = []
    for i in idxs:
        pil = Image.fromarray(frames[i])
        pil = resize_keep_aspect(pil, max_side=max_side)
        # Instead of raw bytes, we'll return the PIL object for later upload/use
        image_list.append(pil)
    return image_list

def build_instruction_block() -> str:
    # Instructions remain the same, just keeping it clean
    return """
# Role
You are a vision annotator that writes neutral, factual, chronological descriptions of traffic anomalies.

# Inputs available
1) Video frames (sampled images)
2) Human labeled context that includes:
   a. Anomaly relevance degree. Value in this case is <moderate>
   b. Anomaly type. Value in this case is <illegal turn>

# Task
Write a precise, factual, chronological description of the anomaly event using only observable evidence in the frames and the given context. Be concise. Include:
1) Road layout / environment
2) Road users involved
3) Clear event sequence
4) Brief analysis of why it is anomalous

# Rules
1) Use only observable evidence + given context.
2) Keep the timeline strictly chronological.
3) Use unambiguous terms and consistent nouns.
4) Limit the combined description+analysis to < 100 words.
5) Choose anomaly relevance from {critical|moderate|low}.
6) Choose anomaly type from the catalog:
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
Return a single JSON object:
{
  "anomaly_relevance": "<critical|moderate|low>",
  "anomaly_type_code": <integer>,
  "anomaly_type_label": "<label>",
  "event_description": "<chronological description, <100 words incl. analysis>",
  "event_analysis": "<brief reason (counts toward same 100-word limit)>"
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

def call_gemini_for_video(prompt: str, video_path: str, model_name: str = "gemini-2.5-flash", timeout_s: int = 120) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY in your environment before running.")

    client = genai.Client()

    video_bytes = open(video_path, 'rb').read()

    # Prepare parts: system/instruction, user prompt, images
    instruction = build_instruction_block().strip()


    generation_config = {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_output_tokens": 1024,
        # Requesting JSON output is a strong signal for structured output
        "response_mime_type": "application/json"
    }

    # The parts now include the instruction, prompt, and the *file references*
    parts = [instruction, prompt, types.Part(inline_data=types.Blob(data=video_bytes, mime_type='video/mp4'))]

    resp_text = None

    # Fix: Pass 'timeout' directly instead of using 'request_options'
    resp = client.models.generate_content(
        model=model_name,
        contents=parts,
        config=generation_config
    )
    resp_text = resp.text

    # Retrieve text
    if not resp_text:
        raise RuntimeError("Empty response from Gemini.")
    return resp_text

def save_output_to_json(prompt, video_path, response, model_used, output_file):
    # This function is unchanged and handles saving the result
    data = {
        "timestamp": datetime.now().isoformat(),
        "video_path": str(video_path),
        "prompt": prompt,
        "model": model_used,
        "response_raw": response,
        "metadata": {
            "video_exists": os.path.exists(video_path),
            "video_size_mb": round(os.path.getsize(video_path)/(1024*1024), 2) if os.path.exists(video_path) else None
        }
    }
    # Try to normalize to JSON object
    try:
        normalized = parse_json_from_text(response)
        data["response"] = normalized
    except Exception:
        data["response"] = None

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nOutput saved to: {output_file}")

if __name__ == "__main__":
    prompt = """
    Please review the video and answer the three items below in English.
    1) Identify the anomaly in the video
    2) Provide a detailed description of the anomaly from start to end
    3) Provide a brief analysis that explains your basis for judging the behavior as an anomaly
    """.strip()

    video_path = "./carmel_data/MedicalDrive-Rangeline/WrongTurnIntoTraffic.mp4" 

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'.")
        print("Please change the 'video_path' variable to a valid path for a video file.")
    else:
        # Changed default to a faster model for general testing
        model = "gemini-2.5-pro"

        try:
            print(f"Calling Gemini API with model: {model} for video: {video_path}")
            response_text = call_gemini_for_video(prompt, video_path, model_name=model, timeout_s=180)
            print("-" * 50)
            print("Response from model:")
            print(response_text)
            print("-" * 50)

            # Save normalized JSON
            out_struct_path = f"anomaly_gemini_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            try:
                parsed = parse_json_from_text(response_text)
                with open(out_struct_path, "w", encoding="utf-8") as f:
                    json.dump(parsed, f, indent=2, ensure_ascii=False)
                print(f"Saved structured JSON to: {out_struct_path}")
            except Exception as e:
                # Fallback: save wrapper with raw + metadata
                out_fallback = f"anomaly_gemini_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                save_output_to_json(prompt, video_path, response_text, model, out_fallback)
                print(f"Failed to parse JSON strictly ({e}). Saved raw response & metadata.")
        except Exception as e:
            print(f"An error occurred during the API call or processing: {e}")