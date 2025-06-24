import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df_raw = pd.read_csv("music_genre.csv")              # ada 'track_name'
df_final = pd.read_csv("final_music_genre.csv")      # hasil preprocessing (fitur numerik)

# Tambahkan kolom track_name dari dataset mentah
if 'track_name' in df_raw.columns:
    df_final['track_name'] = df_raw['track_name']
else:
    st.error("Kolom 'track_name' tidak ditemukan di music_genre.csv")
    st.stop()

# Simpan nama lagu
track_names = df_final['track_name'].tolist()

# Ambil fitur numerik (tanpa label dan nama lagu)
feature_columns = [col for col in df_final.columns if col not in ['track_name', 'label']]
features = df_final[feature_columns]

# Hitung cosine similarity antar lagu
similarity = cosine_similarity(features)

# Judul aplikasi
st.title("🎶 Rekomendasi Lagu Serupa")
st.write("Pilih satu lagu untuk melihat rekomendasi lagu lain yang mirip.")

# Pilih lagu
selected_track = st.selectbox("Pilih Lagu", track_names)

# Tampilkan rekomendasi saat tombol diklik
if st.button("Tampilkan Rekomendasi"):
    index = track_names.index(selected_track)
    sim_scores = list(enumerate(similarity[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    st.subheader("🎧 Rekomendasi Lagu Serupa:")
    count = 0
    for i, score in sim_scores[1:]:  # Skip lagu itu sendiri
        st.write(f"{df_final.iloc[i]['track_name']} (Skor kemiripan: {score:.2f})")
        count += 1
        if count == 5:
            break






