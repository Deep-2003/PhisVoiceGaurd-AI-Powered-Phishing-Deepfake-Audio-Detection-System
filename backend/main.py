import os
import shutil
import pickle

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.detector import analyze_audio_file
from phishing_detection.src.preprocess import clean_text


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="PhisVoiceGuard API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

UPLOAD_DIR = os.path.join(BASE_DIR, "temp_uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ============================================================
# ROOT / HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "system": "PhisVoiceGuard AI Engine"
    }


# ============================================================
# PHISHING DETECTION MODEL
# ============================================================

PHISHING_MODEL_PATH = os.path.join(
    BASE_DIR,
    "phishing_detection",
    "models",
    "phishing_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "phishing_detection",
    "models",
    "vectorizer.pkl"
)


with open(PHISHING_MODEL_PATH, "rb") as f:
    phishing_model = pickle.load(f)


with open(VECTORIZER_PATH, "rb") as f:
    phishing_vectorizer = pickle.load(f)


class PhishingRequest(BaseModel):
    text: str


# ============================================================
# PHISHING API
# ============================================================

@app.post("/api/predict-phishing")
def predict_phishing(request: PhishingRequest):

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Please provide text to analyze."
        )

    # Same preprocessing used during training
    cleaned_text = clean_text(request.text)

    # Convert text to TF-IDF representation
    features = phishing_vectorizer.transform(
        [cleaned_text]
    )

    # Model prediction
    prediction = phishing_model.predict(features)[0]

    # Prediction probabilities
    probabilities = phishing_model.predict_proba(features)[0]

    # Probability belonging to predicted class
    class_index = list(
        phishing_model.classes_
    ).index(prediction)

    confidence = float(
        probabilities[class_index]
    )

    confidence_percent = round(
        confidence * 100,
        2
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Based on the current project convention,
    # class 1 is treated as phishing.
    # --------------------------------------------------------

    if int(prediction) == 1:
        label = "PHISHING"
    else:
        label = "SAFE"


    # Risk based on confidence
    if label == "PHISHING":

        if confidence_percent >= 80:
            risk_level = "HIGH"

        elif confidence_percent >= 60:
            risk_level = "MEDIUM"

        else:
            risk_level = "LOW"

    else:

        if confidence_percent >= 80:
            risk_level = "LOW"

        elif confidence_percent >= 60:
            risk_level = "MEDIUM"

        else:
            risk_level = "LOW"


    return {
        "prediction": int(prediction),
        "label": label,
        "confidence": confidence_percent,
        "risk_level": risk_level,
        "model": "Multinomial Naive Bayes + TF-IDF"
    }


# ============================================================
# DEEPFAKE AUDIO DETECTION
# ============================================================

@app.post("/api/analyze")
async def analyze_audio(
    file: UploadFile = File(...)
):

    allowed_extensions = (
        ".wav",
        ".mp3",
        ".ogg",
        ".flac",
        ".m4a"
    )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No audio file provided."
        )


    # Prevent unsafe path components in filename
    safe_filename = os.path.basename(
        file.filename
    )


    if not safe_filename.lower().endswith(
        allowed_extensions
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid audio format. "
                "Supported formats: "
                ".wav, .mp3, .ogg, .flac, .m4a"
            )
        )


    temp_path = os.path.join(
        UPLOAD_DIR,
        safe_filename
    )


    try:

        # Save uploaded file temporarily
        with open(temp_path, "wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        # Run audio analysis
        result = analyze_audio_file(
            temp_path
        )


        # Add original filename
        result["filename"] = safe_filename


        return result


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Audio processing error: {str(e)}"
        )


    finally:

        # Delete temporary file
        if os.path.exists(temp_path):

            os.remove(temp_path)