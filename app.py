import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('music_model.pkl')

# Header
st.title("🎵 Music Recommendation App")
st.write("Cari lagu favoritmu dan dapatkan rekomendasi lagu serupa!")

# Contoh lagu yang tersedia (nama lagu harus sesuai dataset training)
available_songs = [
    "Song A", "Song B", "Song C", "Song D", "Song E"
]

# Input dari user
selected_song = st.selectbox("Pilih lagu favoritmu:", available_songs)

# Tombol prediksi
if st.button("🎧 Rekomendasikan Lagu Serupa"):
    try:
        # Ubah lagu terpilih menjadi fitur numerik (simulasi)
        # Misal kita punya fitur one-hot/embedding: model akan mengharapkan bentuk numerik
        # Di sini kamu HARUS menyesuaikan input dengan apa yang digunakan saat training

        # Simulasi: Ambil index lagu sebagai input
        song_index = available_songs.index(selected_song)
        X_input = np.array([[song_index]])  # Model hanya menerima input numerik

        # Prediksi
        predictions = model.kneighbors(X_input, n_neighbors=5, return_distance=False)

        # Tampilkan rekomendasi
        st.subheader("🎶 Lagu Rekomendasi:")
        for idx in predictions[0]:
            if idx != song_index:  # Hindari lagu itu sendiri
                st.write(f"- {available_songs[idx]}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat merekomendasikan: {e}")

