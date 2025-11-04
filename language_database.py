import random

def description_database():
    sentences = [
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
    return random.choice(sentences)


def analysis_database():
    sentences = [
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
    return random.choice(sentences)

def severity_database():
    sentences = [
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
    return random.choice(sentences)

def category_database():
    sentences = [
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
    return random.choice(sentences)