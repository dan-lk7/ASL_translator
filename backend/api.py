from fastapi import (FastAPI,UploadFile,File)
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from backend.predict import predict_video
app = FastAPI(title="ASL API")
# CORS
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"],)
# HEALTH CHECK
@app.get("/")
def home():
    return {
        "message": "ASL API running"
    }
# PREDICT
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = (f"temp_{file.filename}")
    with open(temp_path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)
    try:
        prediction, confidence = (predict_video(temp_path))
        result = {
            "prediction": prediction,
            "confidence": round(
                confidence * 100,
                2
            )
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return result

