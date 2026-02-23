"""
app.py
------
Premium Streamlit web app for Music Genre Classification.
Enables real-time genre prediction from audio files with high-aesthetic UI.
"""

import streamlit as st
import librosa
import numpy as np
import pickle
import tempfile
import os
import matplotlib.pyplot as plt

# ─── Page configuration ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Music Genre Classifier | AI Intelligence",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS for Premium Design ─────────────────────────────────────────
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1e1e2f 0%, #121212 100%);
        color: #e0e0e0;
    }
    .stApp {
        background-color: transparent;
    }
    h1 {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 3rem !important;
        margin-bottom: 0.5rem !important;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-top: 2rem;
    }
    .metric-card {
        text-align: center;
        padding: 1rem;
        border-radius: 15px;
        background: rgba(0, 210, 255, 0.1);
        border: 1px solid rgba(0, 210, 255, 0.2);
    }
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s cubic-bezier(.25,.8,.25,1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 210, 255, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ─── Load artifacts helper ───────────────────────────────────────────────
@st.cache_resource
def load_assets():
    """Load model, scaler, and label encoder."""
    try:
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        return model, scaler, le
    except FileNotFoundError:
        st.error("❌ Model artifacts not found. Please ensure they are in the `models/` directory.")
        return None, None, None

def extract_audio_features(file_path):
    """Mirror the feature extraction used during training (58 features)."""
    y, sr = librosa.load(file_path, duration=30)
    features = [float(len(y))]
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features += [chroma.mean(), chroma.var()]
    
    # RMS
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
    
    # Harmony & Perceptr
    harmony, perceptr = librosa.effects.hpss(y)
    features += [harmony.mean(), harmony.var(), perceptr.mean(), perceptr.var()]
    
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.append(float(tempo))
    
    # MFCCs (20 coefficients * 2 attributes = 40 features)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for coef in mfcc:
        features += [coef.mean(), coef.var()]
        
    return np.array(features).reshape(1, -1)

GENRE_MAP = {
    'blues': '🎸 Blues', 'classical': '🎻 Classical', 'country': '🤠 Country',
    'disco': '🕺 Disco', 'hiphop': '🎤 Hiphop', 'jazz': '🎷 Jazz',
    'metal': '🤘 Metal', 'pop': '🎶 Pop', 'reggae': '🌴 Reggae', 'rock': '🎸 Rock'
}

model, scaler, le = load_assets()

# Sidebar info
with st.sidebar:
    st.image("https://img.icons8.com/bubbles/200/music.png")
    st.markdown("### About this AI")
    st.info("This model was trained on the GTZAN dataset using a Random Forest classifier. It achieves ~78% accuracy by analyzing 58 distinct audio characteristics.")
    st.divider()
    st.caption("Developed for Music Intelligence Hackathon 2026")

# Main Content
st.title("SoundVoyage AI")
st.text("Intelligent Music Genre Classification & Sentiment Analysis")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📤 Upload Track")
    st.markdown("Drop your music file here to analyze its DNA.")
    
    uploaded_file = st.file_uploader("", type=["wav", "mp3", "ogg", "flac", "m4a"])
    
    if uploaded_file is not None:
        # Detect extension for temp file
        file_extension = os.path.splitext(uploaded_file.name)[1]
        st.audio(uploaded_file)
        
        if st.button("Analyze Sonic DNA"):
            with st.spinner("Decoding audio features..."):
                # Use temp file for librosa.load
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name
                
                try:
                    # 1. Extraction
                    features = extract_audio_features(tmp_path)
                    
                    # 2. Preprocessing
                    features_scaled = scaler.transform(features)
                    
                    # 3. Prediction
                    probs = model.predict_proba(features_scaled)[0]
                    best_idx = np.argmax(probs)
                    genre = le.classes_[best_idx]
                    confidence = probs[best_idx]
                    
                    # 4. Results storage
                    st.session_state['result'] = {
                        'genre': genre,
                        'confidence': confidence,
                        'probs': probs,
                        'classes': le.classes_
                    }
                finally:
                    os.unlink(tmp_path)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if 'result' in st.session_state:
        res = st.session_state['result']
        genre_name = res['genre']
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🎯 Classification Result")
        
        # Display Best Genre
        st.markdown(f'<div class="metric-card"><h2>{GENRE_MAP.get(genre_name, genre_name.upper())}</h2></div>', unsafe_allow_html=True)
        
        st.write("")
        st.metric("Model Confidence", f"{res['confidence']*100:.1f}%")
        
        st.divider()
        st.subheader("Probability Distribution")
        
        # Plotting confidence scores
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_alpha(0) # Transparent figure
        ax.patch.set_alpha(0)
        
        y_pos = np.arange(len(res['classes']))
        sorted_indices = np.argsort(res['probs'])
        
        colors = ['#00d2ff' if i == best_idx else (1, 1, 1, 0.2) for i in range(len(res['classes']))]
        bars = ax.barh(y_pos, res['probs'][sorted_indices], color=[colors[i] for i in sorted_indices])
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([res['classes'][i].capitalize() for i in sorted_indices], color='white')
        ax.set_xlabel('Probability', color='white')
        ax.tick_params(colors='white')
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="glass-card" style="height: 100%; display: flex; align-items: center; justify-content: center; text-align: center; color: rgba(255,255,255,0.3);">', unsafe_allow_html=True)
        st.write("Results will appear here after analysis.")
        st.markdown('</div>', unsafe_allow_html=True)

# Visual Footer
st.write("")
st.write("")
st.divider()
st.caption("Powered by Scikit-learn, Librosa and Streamlit. © 2026 SoundVoyage AI Laboratories.")
