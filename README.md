# 🎵 Music Genre Classification

A machine learning project that classifies music clips into genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock) using audio features extracted with Librosa and ML models built with Scikit-learn.

## 📌 Project Description

This project uses the **GTZAN Genre Collection** dataset — 1,000 audio clips (30 seconds each) across 10 genres. We extract audio features (MFCCs, chroma, spectral features, tempo, etc.) and train multiple ML classifiers to predict the genre of a music clip.

## 🛠️ Tools & Libraries

| Tool | Purpose |
|---|---|
| Python 3.10+ | Programming language |
| Librosa | Audio feature extraction |
| NumPy & Pandas | Data manipulation |
| Scikit-learn | ML models & evaluation |
| Matplotlib & Seaborn | Visualization |
| Jupyter Notebook | Interactive development |

## 📁 Project Structure

```
Music Genre Classification/
│
├── archive/
│   └── Data/
│       ├── genres_original/     ← Raw .wav audio files (GTZAN)
│       ├── features_30_sec.csv  ← Pre-extracted features
│       └── features_3_sec.csv
│
├── music_genre_classification.ipynb  ← Main Jupyter Notebook
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Setup & Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/music-genre-classification.git
cd music-genre-classification
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the GTZAN Dataset**
   - Download from [Kaggle GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
   - Extract into the `archive/Data/` folder

4. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

5. **Open and run** `music_genre_classification.ipynb` — run all cells top to bottom

## 🧠 Solution Approach

1. **Data Loading** — Load pre-extracted features CSV (1000 songs × 60 features)
2. **EDA** — Explore genre distribution and feature statistics
3. **Preprocessing** — Encode labels with `LabelEncoder`, normalize with `StandardScaler`
4. **Model Training** — Train 4 classifiers: Random Forest, SVM, KNN, MLP Neural Network
5. **Evaluation** — Compare accuracy scores, plot confusion matrix for best model
6. **Prediction** — Predict genre of a new audio clip

## 📊 Models Used

- 🌲 **Random Forest** — Ensemble of decision trees
- 🔷 **SVM** (Support Vector Machine) — Finds optimal class boundary
- 📍 **KNN** (K-Nearest Neighbors) — Classifies by similarity
- 🧠 **MLP** (Neural Network) — Multi-layer perceptron

## 📋 Dependencies

See `requirements.txt` for full list.

## 📄 License

This project is for educational purposes.
