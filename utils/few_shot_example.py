import base64
import numpy as np
import cv2
from PIL import Image
import io

class FewShotExamples:
    def __init__(self):
        pass

    def resize_keep_aspect(self,pil, max_side=512):
        w, h = pil.size
        s = max(w, h)
        if s <= max_side:
            return pil
        scale = max_side / s
        return pil.resize((int(w*scale), int(h*scale)), Image.BILINEAR)

    def sample_frames_as_base64(self,video_path, max_frames=12, max_side=512, jpeg_q=80):
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
            pil = self.resize_keep_aspect(pil, max_side=max_side)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=jpeg_q, optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            b64_list.append(f"data:image/jpeg;base64,{b64}")
        return b64_list

    def call_RangelineSMedicalDr_example(self):
        prompts = ["""
                Review the video and return one JSON object that follows the schema below. Use only what is visible in the frames and what is given in Context. Keep present tense. Use short sentences. No extra text outside JSON.

                Context:
                    • Human labeled anomaly relevance is 3.
                    • Human labeled anomaly type is 18.
                    • Subject start box in frame is [0.1328, 0.3005, 0.1536, 0.3338].
                    • Subject end box in frame is [0.8917, 0.4819, 0.9847, 0.5655].
                    • Event [start_time, end_time] relative to video length (in seconds) is [2.3,27.248], video length is 28.446 seconds.
                   """,
                   """
                Review the video and return one JSON object that follows the schema below. Use only what is visible in the frames and what is given in Context. Keep present tense. Use short sentences. No extra text outside JSON.

                Context:
                    • Human labeled anomaly relevance is 4.
                    • Human labeled anomaly type is 22.
                    • Subject start box in frame is [0.2098, 0.6443, 0.3715, 0.8771].
                    • Subject end box in frame is [0.1458, 0.2456, 0.1841, 0.2871].
                    • Event [start_time, end_time] relative to video length (in seconds) is [0, 11.917], video length is 11.917 seconds.
                    """,
                   """
                Review the video and return one JSON object that follows the schema below. Use only what is visible in the frames and what is given in Context. Keep present tense. Use short sentences. No extra text outside JSON.

                Context:
                    • Human labeled anomaly relevance is 2.
                    • Human labeled anomaly type is 30.
                    • Subject start box in frame is [0.489, 0.5623, 0.6118, 0.7273].
                    • Subject end box in frame is [0.6689, 0.1949, 0.6897, 0.22881].
                    • Event [start_time, end_time] relative to video length (in seconds) is [0, 10.712], video length is 10.712 seconds.
    """]


        video_path = ["./OTA/RangelineSMedicalDr/testdata/videos/RangelineSMedicalDr_av_258_18_3.mp4", "./OTA/RangelineSMedicalDr/testdata/videos/RangelineSMedicalDr_av_279_22_4.mp4","./OTA/RangelineSMedicalDr/testdata/videos/RangelineSMedicalDr_av_273_30_2.mp4"]

        # Define a term: In xx, <road user-A> <action-A> xxxx. If it affects <road user-B> <action-B> xxxx.
        examples = [
            {
                "prompt": "Example:" + prompts[0],
                "video_images_b64": self.sample_frames_as_base64(video_path[0],max_frames=32, max_side=384),
                "answer_json": {
                    "anomaly_relevance": "3",
                    "anomaly_type_code": 18,
                    "anomaly_type_label": "approaching waiting/slow cars at unusual points (high traffic / behind slow vehicle -> car drives slower/waits)",
                    "event_description": "In daylight at a two-lane roundabout, a white car enters from the left, pauses at the yield line, and then merges into the roundabout without leaving a safe distance, causing a blue sedan to slow down. The white car continues and exits at the right-hand exit.",
                    "event_analysis": "The white car fails to yield and forces the blue car to brake abruptly. This clear cut-off disrupts flow and raises collision risk."
                }
            },
            {
                "prompt": "Example:" + prompts[1],
                "video_images_b64": self.sample_frames_as_base64(video_path[1],max_frames=24, max_side=384),
                "answer_json": {
                    "anomaly_relevance": "4",
                    "anomaly_type_code": 22,
                    "anomaly_type_label": "short wrong way driving (wrong direction in roundabout but leaving the scene correctly)",
                    "event_description": "At night in a two-lane roundabout, a black SUV enters from the bottom-left, then instead of circulating counter-clockwise, it turns left across the circulating lanes and exits toward the upper-left exit lane.",
                    "event_analysis": "The SUV briefly drives the wrong way by cutting directly to the exit instead of circulating, violating roundabout rules and creating a short conflict."
                }
            },
            {
                "prompt": "Example:" + prompts[2],
                "video_images_b64": self.sample_frames_as_base64(video_path[2],max_frames=24, max_side=384),
                "answer_json": {
                    "anomaly_relevance": "2",
                    "anomaly_type_code": 30,
                    "anomaly_type_label": "Strong, sudden braking",
                    "event_description": "In a two-lane roundabout, a silver sedan enters from the lower-middle of the frame, brakes sharply before the entry, slightly crosses the yield line, then continues normally into the roundabout.",
                    "event_analysis": "The sedan makes a sudden, strong brake while partially crossing the yield line, likely due to late reaction to traffic inside the roundabout."
                }
            }
        ]

        return examples

    def call_MononElmStreetNB_example(self):
        prompts = ["""
                Review the video and return one JSON object that follows the schema below. Use only what is visible in the frames and what is given in Context. Keep present tense. Use short sentences. No extra text outside JSON.

                Context:
                    • Human labeled anomaly relevance is 2.
                    • Human labeled anomaly type is 26.
                    • Subject start box in frame is [0.2894, 0.555, 0.5835, 0.9321].
                    • Subject end box in frame is [0.0165, 0.2584, 0.1114, 0.3664].
                    • Event [start_time, end_time] relative to video length (in seconds) is [0, 19.133], video length is 19.133 seconds.
                   """,
                   """
                Review the video and return one JSON object that follows the schema below. Use only what is visible in the frames and what is given in Context. Keep present tense. Use short sentences. No extra text outside JSON.

                Context:
                    • Human labeled anomaly relevance is 3.
                    • Human labeled anomaly type is 21.
                    • Subject start box in frame is [0.8156, 0.4345, 0.9651, 0.6175].
                    • Subject end box in frame is [0.7749, 0.5933, 0.984, 0.8742].
                    • Event [start_time, end_time] relative to video length (in seconds) is [0.997, 9.915], video length is 10.813 seconds.
                    """
    ]
        
        video_path = ["./OTA/MononElmStreetNB/testdata/videos/MononElmStreetNB_av_239_26_2.mp4", "./OTA/MononElmStreetNB/testdata/videos/MononElmStreetNB_av_303_21_3.mp4"]

        # Define a term: In xx, <road user-A> <action-A> xxxx. If it affects <road user-B> <action-B> xxxx.
        examples = [
        {
            "prompt": "Example:" + prompts[0],
            "video_images_b64": self.sample_frames_as_base64(video_path[0],max_frames=24, max_side=384),
            "answer_json": {
                "anomaly_relevance": "2",
                "anomaly_type_code": 26,
                "anomaly_type_label": "stopping in the middle of the street to let someone cross the street",
                "event_description": "In daylight at a curbless shared street, a black sedan enter from bottom of the frame, slow down and stop in the middle of the street, then resumes and exits via the top-left of the frame.",
                "event_analysis": "The vehicle pauses in the travel lane to yield to a pedestrian."
            }
        },
        {
            "prompt": "Example:" + prompts[1],
            "video_images_b64": self.sample_frames_as_base64(video_path[1],max_frames=24, max_side=384),
            "answer_json": {
                "anomaly_relevance": "3",
                "anomaly_type_code": 21,
                "anomaly_type_label": "illegal turns",
                "event_description": "In daylight on a curbless shared street, a dark sedan enters from the right, makes a mid-block U-turn, and exits at the bottom-right.",
                "event_analysis": "A mid-block U-turn on a shared street is an illegal maneuver that can confuse other users and obstruct oncoming or following traffic."
            }
        }
    ]

        return examples

    def call_RangelineS116thSt_example(self):
        prompts = ["""
                Review the video and return one JSON object that follows the schema below. Use only what is visible in the frames and what is given in Context. Keep present tense. Use short sentences. No extra text outside JSON.

                Context:
                    • Human labeled anomaly relevance is 3.
                    • Human labeled anomaly type is 18.
                    • Subject start box in frame is [0.0177, 0.2094, 0.0475, 0.2557].
                    • Subject end box in frame is [0.939, 0.384, 0.9899, 0.4422].
                    • Event [start_time, end_time] relative to video length (in seconds) is [0, 14.224], video length is 14.224 seconds.
                   """,
                   """
                Review the video and return one JSON object that follows the schema below. Use only what is visible in the frames and what is given in Context. Keep present tense. Use short sentences. No extra text outside JSON.

                Context:
                    • Human labeled anomaly relevance is 4.
                    • Human labeled anomaly type is 23.
                    • Subject start box in frame is [0.9124, 0.3456, 0.9677, 0.3997].
                    • Subject end box in frame is [0.0143, 0.4224, 0.1211, 0.573].
                    • Event [start_time, end_time] relative to video length (in seconds) is [0.2, 10.62], video length is 11.616 seconds.
                    """,
                    """
                Review the video and return one JSON object that follows the schema below. Use only what is visible in the frames and what is given in Context. Keep present tense. Use short sentences. No extra text outside JSON.

                Context:
                    • Human labeled anomaly relevance is 3.
                    • Human labeled anomaly type is 21.
                    • Subject start box in frame is [0.463, 0.0902, 0.4881, 0.1146].
                    • Subject end box in frame is [0.9163, 0.3281, 0.9829, 0.3918].
                    • Event [start_time, end_time] relative to video length (in seconds) is [0, 14.518], video length is 14.518 seconds.
                    """]
        
        video_path = ["./OTA/RangelineS116thSt/testdata/videos/RangelineS116thSt_av_53_18_3.mp4", "./OTA/RangelineS116thSt/testdata/videos/RangelineS116thSt_av_276_23_4.mp4", "./OTA/RangelineS116thSt/testdata/videos/RangelineS116thSt_av_315_21_3.mp4"]

        # Define a term: In xx, <road user-A> <action-A> xxxx. If it affects <road user-B> <action-B> xxxx.
        examples = [
        {
            "prompt": "Example:" + prompts[0],
            "video_images_b64": self.sample_frames_as_base64(video_path[0],max_frames=24, max_side=384),
            "answer_json": {
                "anomaly_relevance": "3",
                "anomaly_type_code": 18,
                "anomaly_type_label": "cut another traffic agent off clearly/strongly (very fast driving and/or the other agent has to break/stop)",
                "event_description": "In a rainy two-lane roundabout, a black car enters from the bottom, begins braking before the yield line, but comes to a stop with its front slightly over the line. A white car already in the roundabout must brake and change lanes to avoid a collision.",
                "event_analysis": "The black car fails to yield properly and encroaches past the yield line, forcing the white car to brake hard and shift lanes. This clear cut-off disrupts traffic flow and elevates collision risk."
            }
        },
        {
            "prompt": "Example:" + prompts[1],
            "video_images_b64": self.sample_frames_as_base64(video_path[1],max_frames=24, max_side=384),
            "answer_json": {
                "anomaly_relevance": "4",
                "anomaly_type_code": 23,
                "anomaly_type_label": "wrong way driver (wrong side of the road)",
                "event_description": "At night in a two-lane roundabout, a black car enters from an exit lane on the right, travels clockwise against the correct flow, then leaves via an entry lane on the left.",
                "event_analysis": "This is sustained wrong-way driving with entry and exit through improper lanes, creating head-on conflict risk and clearly violating roundabout rules."
            }
        },
        {
            "prompt": "Example:" + prompts[1],
            "video_images_b64": self.sample_frames_as_base64(video_path[1],max_frames=24, max_side=384),
            "answer_json": {
                "anomaly_relevance": "3",
                "anomaly_type_code": 21,
                "anomaly_type_label": "illegal turns",
                "event_description": "In a two-lane roundabout, the subject vehicle enters from the top approach in the outer lane, then make a illegal lane change inside the roundabout on the left of the frame and exits via the right-hand exit.",
                "event_analysis": "Vehicles aiming for the third exit should enter in the inner lane, circulate, then transition to the outer lane only to exit. The subject enters in the outer lane and changes lanes inside the roundabout, which is illegal and likely to confuse other road users, increasing sideswipe/cut-off risk."
            }
        }
    ]

        return examples