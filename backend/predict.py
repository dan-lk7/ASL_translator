import cv2
import torch
import numpy as np
from transformers import (VideoMAEForVideoClassification,VideoMAEImageProcessor)
# CONFIG
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR,"..","model")
print("MODEL PATH =", MODEL_PATH)
CLASSES = ["hello","yes","no","drink","help"]
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
def extract_frames(video_path,num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None
    indices = np.linspace(0,total_frames - 1,num_frames,dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        return None
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames[:num_frames]
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
