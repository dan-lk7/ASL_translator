import imageio.v2 as imageio
import torch
import numpy as np
from PIL import Image
from transformers import (VideoMAEForVideoClassification,VideoMAEImageProcessor,)
# COnfiguration
MODEL_PATH = "dan-lk/asl-v3-videomae"
NUM_FRAMES = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#LOAD MODEL
print("Loading model from Hugging Face...")
model = VideoMAEForVideoClassification.from_pretrained(MODEL_PATH)
processor = VideoMAEImageProcessor.from_pretrained(MODEL_PATH)
model = model.to(device)
model.eval()
print("Model loaded successfully.")
# FRAME EXTRACTION
def extract_frames(video_path):
    try:
        reader = imageio.get_reader(video_path,format="ffmpeg")
        frames = []
        for frame in reader:
            try:
                frame = Image.fromarray(frame)
                frame = frame.convert("RGB")
                frame = frame.resize((224, 224),Image.Resampling.BILINEAR)
                frames.append(np.array(frame))
            except Exception:
                # Ignore une frame corrompue
                continue
        reader.close()
        if len(frames) == 0:
            raise Exception("Aucune frame valide trouvée.")
        # Vidéo trop courte
        if len(frames) < NUM_FRAMES:
            last_frame = frames[-1]
            while len(frames) < NUM_FRAMES:
                frames.append(last_frame)
        # Sélection uniforme des frames
        indices = np.linspace(0,len(frames) - 1,NUM_FRAMES,dtype=int)
        selected_frames = [frames[i] for i in indices]
        return selected_frames
    except Exception as e:
        print("Video loading error:", e)
        return None
# PREDICTION
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
    predicted_label = model.config.id2label[prediction]
    return predicted_label, confidence
# LOCAL TEST
if __name__ == "__main__":
    video_path = input("Video path: ")
    prediction, confidence = predict_video(video_path)
    print("\nPrediction:", prediction)
    print("Confidence:",round(confidence * 100, 2),"%")
