import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
data = pd.read_csv("final_music_genre.csv")

# Pastikan kolom nama lagu ada
if 'track_name' not in data.columns:
    st.error("Dataset tidak memiliki kolom 'track_name'. Harap tambahkan kolom tersebut.")
    st.stop()

# Simpan nama lagu
track_names = data['track_name'].tolist()

# Ambil hanya kolom fitur numerik (kecuali kolom genre/nama)
feature_columns = [col for col in data.columns if col not in ['track_name', 'label']]
features = data[feature_columns]

# Hitung similarity matrix (cosine similarity antar lagu)
similarity = cosine_similarity(features)

# Judul aplikasi
st.title("🎶 Rekomendasi Lagu Serupa")
st.write("Pilih satu lagu untuk melihat rekomendasi lagu lain yang mirip.")

# Pilih lagu
selected_track = st.selectbox("Pilih Lagu", track_names)

# Proses jika lagu dipilih
if st.button("Tampilkan Rekomendasi"):
    index = track_names.index(selected_track)
    sim_scores = list(enumerate(similarity[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    st.subheader("🎧 Rekomendasi Lagu Serupa:")
    count = 0
    for i, score in sim_scores[1:]:  # skip index 0 (itu dirinya sendiri)
        st.write(f"{data.iloc[i]['track_name']} (Skor kemiripan: {score:.2f})")
        count += 1
        if count == 5:  # tampilkan 5 lagu
            break




