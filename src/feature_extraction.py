"""
feature_extraction.py
---------------------
Extracts audio features from .wav files in the GTZAN dataset
and saves them to a CSV file for model training.

Features extracted:
- Length, Chroma STFT, RMS, Spectral Centroid, Spectral Bandwidth,
  Spectral Rolloff, Zero Crossing Rate, Harmony, Percussive, Tempo,
  MFCCs (20 coefficients)
"""

import os
import librosa
import numpy as np
import pandas as pd


DATASET_PATH = 'archive/Data/genres_original'
OUTPUT_CSV   = 'features_extracted.csv'
GENRES       = ['blues', 'classical', 'country', 'disco', 'hiphop',
                'jazz', 'metal', 'pop', 'reggae', 'rock']


def extract_features(file_path):
    """Extract audio features from a single .wav file."""
    y, sr = librosa.load(file_path, duration=30)
    features = []

    # Length
    features.append(len(y))

    # Chroma STFT
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features += [chroma.mean(), chroma.var()]

    # RMS Energy
    rms = librosa.feature.rms(y=y)
    features += [rms.mean(), rms.var()]

    # Spectral Centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features += [spec_cent.mean(), spec_cent.var()]

    # Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features += [spec_bw.mean(), spec_bw.var()]

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features += [rolloff.mean(), rolloff.var()]

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features += [zcr.mean(), zcr.var()]

    # Harmony & Percussive
    harmony, perceptr = librosa.effects.hpss(y)
    features += [harmony.mean(), harmony.var(), perceptr.mean(), perceptr.var()]

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.append(float(tempo))

    # MFCCs (20 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for coef in mfcc:
        features += [coef.mean(), coef.var()]

    return features


def build_dataset():
    """Loop through all genres and extract features into a DataFrame."""
    rows = []
    for genre in GENRES:
        genre_path = os.path.join(DATASET_PATH, genre)
        files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
        print(f"Processing {genre} ({len(files)} files)...")
        for fname in files:
            file_path = os.path.join(genre_path, fname)
            try:
                feats = extract_features(file_path)
                rows.append([fname] + feats + [genre])
            except Exception as e:
                print(f"  Skipping {fname}: {e}")

    # Column names
    cols = ['filename', 'length',
            'chroma_stft_mean', 'chroma_stft_var',
            'rms_mean', 'rms_var',
            'spectral_centroid_mean', 'spectral_centroid_var',
            'spectral_bandwidth_mean', 'spectral_bandwidth_var',
            'rolloff_mean', 'rolloff_var',
            'zero_crossing_rate_mean', 'zero_crossing_rate_var',
            'harmony_mean', 'harmony_var',
            'perceptr_mean', 'perceptr_var',
            'tempo']
    for i in range(1, 21):
        cols += [f'mfcc{i}_mean', f'mfcc{i}_var']
    cols.append('label')

    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Features saved to {OUTPUT_CSV} ({len(df)} rows)")
    return df


if __name__ == '__main__':
    build_dataset()
