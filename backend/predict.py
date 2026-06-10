import imageio.v2 as imageio
import torch
import numpy as np
from transformers import (VideoMAEForVideoClassification,VideoMAEImageProcessor)
# CONFIG
MODEL_PATH = "dan-lk/asl-v3-model"
model = VideoMAEForVideoClassification.from_pretrained(MODEL_PATH)
processor = VideoMAEImageProcessor.from_pretrained(MODEL_PATH)
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR,"..","model")
print("MODEL PATH =", MODEL_PATH)
CLASSES = ["hello","yes","no","drink","help"]
NUM_FRAMES = 16
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
# LOAD MODEL
print("Loading model...")
model = VideoMAEForVideoClassification.from_pretrained(MODEL_PATH)
processor = VideoMAEImageProcessor.from_pretrained(MODEL_PATH)
model = model.to(device)
model.eval()
print("Model loaded.")
# FRAME EXTRACTION
def extract_frames(video_path):
    try:
        reader = imageio.get_reader(video_path)
        frames = []
        for frame in reader:
            frames.append(frame)
        reader.close()
        if len(frames) == 0:
            return None
        indices = np.linspace(
            0,
            len(frames) - 1,
            NUM_FRAMES,
            dtype=int
        )
        selected_frames = [
            frames[i]
            for i in indices
        ]

        return selected_frames

    except Exception as e:
        print("Video loading error:", e)
        return None
# PREDICT
def predict_video(video_path):
    frames = extract_frames(video_path)
    if frames is None:
        raise Exception("Impossible de lire la vidéo.")
    inputs = processor(frames,return_tensors="pt")
    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits,dim=1)
    prediction = torch.argmax(probabilities,dim=1).item()
    confidence = torch.max(probabilities).item()
    predicted_label = (model.config.id2label[prediction])
    return (predicted_label,confidence)
# TEST LOCAL
if __name__ == "__main__":
    video_path = input("Video path: ")
    prediction, confidence = (predict_video(video_path))
    print("\nPrediction:",prediction)
    print("Confidence:",round(confidence * 100,2),"%")
