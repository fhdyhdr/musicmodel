import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data preprocessed (fitur untuk similarity)
final_data = pd.read_csv("final_music_genre.csv")

# Load data mentah (berisi nama lagu)
raw_data = pd.read_csv("music_genre.csv")

# Tambahkan kolom track_name dari raw ke final (asumsi urutan sama)
if 'track_name' in raw_data.columns:
    final_data['track_name'] = raw_data['track_name']
else:
    st.error("Kolom 'track_name' tidak ditemukan di music_genre.csv")
    st.stop()

# Siapkan data
track_names = final_data['track_name'].tolist()

# Ambil kolom fitur (tanpa label dan track_name)
feature_columns = [col for col in final_data.columns if col not in ['label', 'track_name']]
features = final_data[feature_columns]

# Hitung similarity
similarity = cosine_similarity(features)

# Judul aplikasi
st.title("🎶 Rekomendasi Lagu Serupa")
st.write("Pilih satu lagu untuk melihat rekomendasi lagu lain yang mirip berdasarkan fitur audio.")

# Pilihan lagu
selected_track = st.selectbox("Pilih Lagu", track_names)

# Tampilkan rekomendasi saat tombol diklik
if st.button("Tampilkan Rekomendasi"):
    index = track_names.index(selected_track)
    sim_scores = list(enumerate(similarity[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    st.subheader("🎧 Lagu yang Mirip:")
    count = 0
    for i, score in sim_scores[1:]:  # skip diri sendiri
        st.write(f"**{final_data.iloc[i]['track_name']}** (Skor kemiripan: {score:.2f})")
        count += 1
        if count == 5:
            break







