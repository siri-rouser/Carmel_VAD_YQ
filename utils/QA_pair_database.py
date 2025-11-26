import random

class QA_pair_database:
    
    def __init__(self):
        self.description_database = [
        "Describe the anomaly events observed in the video.",
        "Could you describe the anomaly events observed in the video?",
        "Could you specify the anomaly events present in the video?",
        "Give a description of the detected anomaly events in this video.",
        "Could you give a description of the anomaly events in the video?",
        "Provide a summary of the anomaly events in the video.",
        "Could you provide a summary of the anomaly events in this video?",
        "What details can you provide about the anomaly in the video?",
        "How would you detail the anomaly events found in the video?",
        "How would you describe the particular anomaly events in the video?",
    ]
        
        self.analysis_database = [
            "Why do you judge this event to be anomalous?",
            "Can you provide the reasons for considering it anomalous?",
            "Can you give the basis for your judgment of this event as an anomaly?",
            "What led you to classify this event as an anomaly?",
            "Could you provide the reasons for considering this event as abnormal?",
            "What evidence do you have to support your judgment of this event as an anomaly?",
            "Can you analyze the factors contributing to this anomalous event?",
            "Could you share your analysis of the anomalous event?",
            "What patterns did you observe that contributed to your conclusion about this event being an anomaly?",
            "How do the characteristics of this event support its classification as an anomaly?",
        ]
        self.severity_database = [
            "On a scale of 0–4, how severe is the anomaly in this video?",
            "Rate the anomaly’s severity from 0 (none) to 4 (critical).",
            "Assign a severity score (0–4) to the anomaly shown.",
            "What severity level (0–4) would you give this anomaly?",
            "Please evaluate the anomaly’s severity on a 0–4 scale.",
            "Provide a 0–4 severity rating for the anomaly.",
            "How would you score the anomaly’s severity (0–4)?",
            "Choose a severity level for the anomaly: 0–4.",
            "Give the anomaly a severity label on the 0–4 scale.",
            "Estimate the anomaly’s severity using a 0–4 rating.",
        ]
        self.category_database = [
            "What types of anomalies are shown in the video clip?",
            "Can you classify the anomaly into a specific category?",
            "Could you identify the category of the anomaly in the video?",
            "Please specify the category of the anomaly observed in the video.",
            "How would you categorize the anomaly present in this video?",
            "What type of anomaly is depicted in the video?",
            "Can you determine the category of the anomaly shown in this video?",
            "Could you provide the classification of the anomaly in the video?",
            "What is the appropriate category for the anomaly observed in this video?",
            "How would you define the category of the anomaly present in this video?",
        ]

        self.category_database_new = [
            "Does this video contain any anomalies? If yes, what type(s) of anomaly are present?",
            "Identify whether there are anomalies in this video. If there are, assign each anomaly to an appropriate category.",
            "Determine if the video shows any anomalies. If it does, indicate the category of each anomaly.",
            "Check the video for anomalies. If any are present, describe them and specify their categories.",
            "Examine the video and determine whether it contains anomalies. If so, list the anomalies and their categories.",
            "Analyze the video for potential anomalies. If anomalies are found, provide the category for each.",
            "Assess whether the video includes any anomalous events. If it does, classify each event into a category.",
            "Inspect the video for anomalies. If any are detected, state their categories.",
            "Determine if the video segment contains anomalous behavior or events. If so, categorize each one.",
            "Review the video to identify anomalies. If anomalies are present, report them along with their categories.",
        ]
        self.four_category_context = """
        #   You are a classification system for identifying anomalous driving behaviors.  
            There are **4 predefined anomaly categories** listed below.  
            1: "speed_trajectory_irregularities": Abnormal speed choice or unstable movement patterns that raise risk (or clearly deviate from “normal” driving), even if they don’t directly create a near-collision event.
            2: "direction_space_violations": The main problem is where the car is relative to legal lanes/roadway/sidewalk, including wrong-way, illegal turns, and using space not meant for vehicles.
            3: "conflict_near_collision": Events where interaction with another road user becomes critical: one vehicle clearly cuts off another, a near crash occurs, or emergency maneuvers (hard braking, swerving) are taken to avoid collision.
            4: "stopped_obstruction_right_of_way": Vehicles that are stopped or nearly stopped in the roadway for special reasons—emergency vehicles, breakdowns, letting pedestrians cross—or that act as an obstruction.
        """

    def question_selection(self,type):
        if type == "description":
            return random.choice(self.description_database)
        elif type == "analysis":
            return random.choice(self.analysis_database)
        elif type == "severity":
            return random.choice(self.severity_database)
        elif type == "category":
            return random.choice(self.category_database)
        elif type == "category_new":
            return random.choice(self.category_database_new)
        else:
            raise ValueError(f"Unknown question type: {type}")

    def question_type_query(self, question: str) -> str:
        """Return which question type the input belongs to."""
        if question in self.description_database:
            return "description"
        elif question in self.analysis_database:
            return "analysis"
        elif question in self.severity_database:
            return "severity"
        elif question in self.category_database:
            return "category"
        elif question in self.category_database_new:
            return "category_new"
        else:
            return "unknown"

    def category_to_index(self, category_name: str) -> str:
        categories = {
            "change of lane": "1",
            "late turn": "2",
            "cutting inside turns": "3",
            "driving on the centerline": "4",
            "yielding to emergency vehicles": "5",
            "waiting or moving out of the way for emergency vehicles": "5",
            "brief wait at an open intersection": "6",
            "long wait at an empty intersection": "7",
            "too far onto the main road while waiting": "8",
            "stopping at an unusual point": "9",
            "slowing at an unusual point": "10",
            "fast driving that appears reckless": "11",
            "slow driving with apparent uncertainty": "12",
            "unusual movement pattern": "13",
            "brief reverse movement": "14",
            "unusual approach toward waiting or slow cars": "15",
            "traffic tie up": "16",
            "almost cut another traffic agent off": "17",
            "cut another traffic agent off clearly": "18",
            "almost collision": "19",
            "into oncoming lane while turning": "20",
            "illegal turn": "21",
            "illegal U-turn inside roundabout": "21",
            "illegal turn (include illegal U-turns and illegal/improper lane changes inside roundabout)": "21",
            "illegal turn (include illegal U-turns and illegal/improper lane changes inside roundabout)": "21",
            "improper lane use/erratic path in roundabout": "21",
            "short wrong way in roundabout then exit": "22",
            "wrong way driver": "23",
            "more than one full turn in a roundabout": "24",
            "broken down vehicle on street": "25",
            "stop mid street to let a person cross": "26",
            "stop at a crosswalk to let a person cross": "27",
            "slight departure from the roadway": "28",
            "slightly getting off the road while driving": "28",
            "on or parking on sidewalk": "29",
            "car getting on / standing on / parking on sidewalk": "29",
            "strong sudden braking": "30",
            "swerve to avoid or maneuver around a vehicle": "31",
            "risky behaviour that does not fit another category": "32",
        }

        if category_name not in categories:
            print("DEBUG category_name repr:", repr(category_name))
            # Optionally compare to the problematic key:
            key = "illegal turn (include illegal U-turns and illegal/improper lane changes inside roundabout)"
            print("DEBUG key repr       :", repr(key))
            print("DEBUG same length?   :", len(category_name), len(key))
            print("DEBUG char diffs:")
            for i, (a, b) in enumerate(zip(category_name, key)):
                if a != b:
                    print(f"  pos {i}: {repr(a)} vs {repr(b)}")
            if len(category_name) != len(key):
                print("  extra chars in one of them")
        return categories.get(category_name)