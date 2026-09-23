import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.detector import analyze_audio_file

# This line must be named "app" so Uvicorn can find it
app = FastAPI(title="PhisVoiceGuard API", version="1.0.0")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"status": "online", "system": "PhisVoiceGuard AI Engine"}

@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    allowed_extensions = (".wav", ".mp3", ".ogg", ".flac", ".m4a")
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail="Invalid audio format. Upload .wav or .mp3")

    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = analyze_audio_file(temp_path)
        result["filename"] = file.filename
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)