import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('music_model.pkl')

# Load dataset (pastikan dataset yang sama dengan model, tanpa label)
@st.cache_data
def load_data():
    # Contoh fitur (gantilah sesuai dataset asli)
    # Harus sama urutannya dengan data yang dipakai saat training model
    data = pd.read_csv("music_features.csv", index_col=0)
    return data

data = load_data()

st.title("🎵 Music Recommendation System")
st.write("Pilih lagu, dan kami akan merekomendasikan lagu serupa.")

# Dropdown pilihan lagu berdasarkan indeks dataset
selected_song = st.selectbox("Pilih lagu:", data.index.tolist())

# Tombol untuk rekomendasi
if st.button("Rekomendasikan Lagu Serupa"):
    # Ambil fitur lagu yang dipilih
    selected_features = data.loc[selected_song].values.reshape(1, -1)

    # Cari 5 lagu paling mirip
    distances, indices = model.kneighbors(selected_features, n_neighbors=6)

    # Tampilkan hasil (kecuali lagu itu sendiri)
    st.subheader("🎧 Lagu Serupa yang Direkomendasikan:")
    recommended = data.iloc[indices[0][1:]]  # [1:] untuk melewati lagu itu sendiri
    for idx in recommended.index:
        st.write(f"- {idx}")








