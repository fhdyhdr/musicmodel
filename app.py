import streamlit as st
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="🎧 Rekomendasi Musik", layout="centered")
st.title("🎧 Sistem Rekomendasi Musik")

# Load model dan data dengan error handling
try:
    model = joblib.load("model.pkl")
    features = joblib.load("features.pkl")
    track_names = joblib.load("track_names.pkl")
except Exception as e:
    st.error("Gagal memuat file model atau data.")
    st.exception(e)
    st.stop()

# Verifikasi tipe dan bentuk data
if not isinstance(track_names, list):
    track_names = list(track_names)

features = np.array(features)

# Cek kesesuaian panjang
if features.shape[0] != len(track_names):
    st.error("Jumlah fitur dan nama lagu tidak cocok.")
    st.stop()

# Pilihan lagu
selected_track = st.selectbox("Pilih lagu favoritmu:", track_names)

if st.button("🎵 Rekomendasikan Lagu Serupa"):
    try:
        idx = track_names.index(selected_track)
        query = np.array(features[idx]).reshape(1, -1)
        
        # Validasi model
        if not hasattr(model, "kneighbors") or not callable(model.kneighbors):
            st.error("Model tidak valid: pastikan ini adalah NearestNeighbors.")
            st.stop()

        distances, indices = model.kneighbors(query, n_neighbors=6)

        st.subheader("🎶 Lagu yang Direkomendasikan:")
        for i in range(1, len(indices[0])):  # Lewati lagu itu sendiri
            similar_index = indices[0][i]
            st.write(f"{i}. {track_names[similar_index]}")

    except Exception as e:
        st.error("Terjadi kesalahan saat mencari rekomendasi.")
        st.exception(e)

st.markdown("---")
st.caption("📻 Dibuat dengan Streamlit dan scikit-learn.")









