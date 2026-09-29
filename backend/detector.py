import librosa
import numpy as np

def analyze_audio_file(file_path: str) -> dict:
    y, sr = librosa.load(file_path, sr=16000, duration=15.0)

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_variability = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0

    flatness = librosa.feature.spectral_flatness(y=y)
    mean_flatness = float(np.mean(flatness))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_variance = float(np.var(centroid))

    zcr = librosa.feature.zero_crossing_rate(y)
    mean_zcr = float(np.mean(zcr))

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_var = float(np.mean(np.var(mfccs, axis=1)))

    synthetic_score = 0.0
    if pitch_variability < 110.0:
        synthetic_score += 0.35
    elif pitch_variability < 190.0:
        synthetic_score += 0.15

    if mean_flatness > 0.018:
        synthetic_score += 0.25
    elif mean_flatness > 0.007:
        synthetic_score += 0.10

    if centroid_variance < 900000.0:
        synthetic_score += 0.25
    else:
        synthetic_score += 0.05

    if mean_zcr < 0.035 or mean_zcr > 0.22:
        synthetic_score += 0.15

    deepfake_confidence = min(max(synthetic_score, 0.04), 0.97)
    is_synthetic = deepfake_confidence >= 0.50

    return {
        "verdict": "SYNTHETIC / AI DEEPFAKE" if is_synthetic else "AUTHENTIC HUMAN VOICE",
        "is_deepfake": bool(is_synthetic),
        "deepfake_probability": round(deepfake_confidence * 100, 1),
        "acoustic_analysis": {
            "duration_seconds": round(float(librosa.get_duration(y=y, sr=sr)), 2),
            "pitch_jitter_hz": round(pitch_variability, 2),
            "spectral_flatness": round(mean_flatness, 6),
            "centroid_variance": round(centroid_variance, 1),
            "zero_crossing_rate": round(mean_zcr, 4),
            "mfcc_energy_variance": round(mfcc_var, 2)
        }
    }