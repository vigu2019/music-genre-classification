import sys
import pickle
import librosa
import numpy as np


MODEL_PATH   = 'models/best_model.pkl'
SCALER_PATH  = 'models/scaler.pkl'
ENCODER_PATH = 'models/label_encoder.pkl'


def load_artifacts():
    """Load saved model, scaler, and label encoder."""
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
    return model, scaler, le


def extract_features(file_path):
    """Extract the 58 audio features from a .wav file."""
    y, sr = librosa.load(file_path, duration=30)
    features: list[float] = [float(len(y))]

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features += [chroma.mean(), chroma.var()]

    rms = librosa.feature.rms(y=y)
    features += [rms.mean(), rms.var()]

    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features += [spec_cent.mean(), spec_cent.var()]

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features += [spec_bw.mean(), spec_bw.var()]

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features += [rolloff.mean(), rolloff.var()]

    zcr = librosa.feature.zero_crossing_rate(y)
    features += [zcr.mean(), zcr.var()]

    harmony, perceptr = librosa.effects.hpss(y)
    features += [harmony.mean(), harmony.var(), perceptr.mean(), perceptr.var()]

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.append(float(tempo))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for coef in mfcc:
        features += [coef.mean(), coef.var()]

    return features


def predict_genre(file_path):
    """Predict the music genre of a .wav file."""
    model, scaler, le = load_artifacts()
    features = extract_features(file_path)
    features_scaled = scaler.transform([features])
    pred = model.predict(features_scaled)
    genre = le.inverse_transform(pred)[0]
    return genre


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_audio.wav>")
        sys.exit(1)

    audio_file = sys.argv[1]
    print(f"Analysing: {audio_file}")
    genre = predict_genre(audio_file)
    print(f"🎵 Predicted Genre: {genre.upper()}")
