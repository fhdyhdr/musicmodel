import streamlit as st
import numpy as np
import joblib

# Load model dan data
model = joblib.load("model.pkl")           # NearestNeighbors
features = joblib.load("features.pkl")     # NumPy array 2D (fitur dari semua lagu)
track_names = joblib.load("track_names.pkl")  # List atau Series nama-nama lagu

st.set_page_config(page_title="Rekomendasi Musik", layout="centered")
st.title("🎧 Sistem Rekomendasi Musik")

# Pilihan lagu
selected_track = st.selectbox("Pilih lagu favoritmu:", track_names)

if st.button("Rekomendasikan Lagu Serupa"):
    # Temukan index dari lagu yang dipilih
    idx = track_names.index(selected_track)
    distances, indices = model.kneighbors(query, n_neighbors=6)


    st.subheader("🎵 Lagu yang Mirip:")
    for i in range(1, len(indices[0])):  # Mulai dari 1 untuk skip lagu itu sendiri
        similar_index = indices[0][i]
        st.write(f"{i}. {track_names[similar_index]}")

st.markdown("---")
st.caption("Sistem rekomendasi musik berbasis fitur audio dan kemiripan cosine.")








