#  InfyMix: Cross-Domain Mood Recommender

> **"Because your movie night should match your playlist."**

**InfyMix** is a hybrid recommendation system that bridges the gap between Music and Movies. Unlike standard recommenders that stay in one domain, InfyMix analyzes the *mathematical mood* of the artist you are listening to and recommends movies that match that exact emotional "vibe."

---

## Key Features

* **Cross-Domain Recommendation:** Input an Artist (Spotify) → Output Movies (IMDb/MovieLens).
* **5-Dimensional Mood Engine:** Analyzes content using a weighted vector of:
    *  **Valence:** Positivity/Happiness
    *  **Energy:** Intensity/Loudness
    *  **Danceability:** Rhythm/Groove
    *  **Acousticness:** Organic vs. Electronic
    *  **Tempo:** Pacing/Speed
* **Lightweight Vector Math:** Uses **Cosine Similarity** for high-performance, crash-free vector matching without heavy ML dependencies.
* **Hybrid-Ready Architecture:** Includes **Neural Collaborative Filtering (NCF)**.
* **Interactive UI:** Built with **Streamlit** for real-time interaction and mood visualization.

---

##  Architecture

The system operates on a dual-engine architecture:

1.  **Branch A (Content-Based):**
    * Extracts audio features from Spotify data.
    * Maps Movie Genres to mood coordinates using a Knowledge-Based Rule System (KBRS).
    * Performs Vector matching using 5D Cosine Similarity.

2.  **Branch B (Collaborative Filtering):**
    * Prepares user interaction graphs from the HetRec 2011 dataset.
    * Designed for Neural Collaborative Filtering (NCF) to learn latent user preferences.

---

## 📦 Installation & Setup

### Install dependencies
```bash
pip install pandas numpy streamlit

### How to run
```bash
python build_mood_vectors.py
python -m streamlit run app.py