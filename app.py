import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("music_model.pkl")

# Contoh data lagu (harus sesuai urutan fiturnya dengan data pelatihan model)
# Gantilah daftar ini dengan daftar lagu aslimu jika ada
music_data = pd.DataFrame({
    'title': [
        'Let It Be', 'Bohemian Rhapsody', 'Imagine', 'Hotel California', 'Smells Like Teen Spirit',
        'Billie Jean', 'Shape of You', 'Blinding Lights', 'Someone Like You', 'Lose Yourself'
    ],
    'feature1': [0.5, 0.8, 0.6, 0.7, 0.9, 0.85, 0.4, 0.6, 0.3, 0.7],
    'feature2': [0.7, 0.6, 0.5, 0.8, 0.9, 0.9, 0.4, 0.5, 0.6, 0.8],
    'feature3': [0.3, 0.9, 0.4, 0.6, 0.95, 0.8, 0.35, 0.65, 0.45, 0.75]
})

# Fitur yang digunakan untuk prediksi
feature_columns = ['feature1', 'feature2', 'feature3']

# Streamlit App
st.set_page_config(page_title="Music Recommender", layout="centered")
st.title("🎵 Music Recommendation System")

# Dropdown pilih lagu
selected_song = st.selectbox("Pilih lagu favoritmu:", music_data['title'].tolist())

if st.button("Rekomendasikan Lagu 🎧"):
    # Ambil fitur dari lagu yang dipilih
    selected_index = music_data[music_data['title'] == selected_song].index[0]
    selected_features = music_data.loc[selected_index, feature_columns].values.reshape(1, -1)

    # Cari rekomendasi dengan model KNN
    distances, indices = model.kneighbors(selected_features)

    # Tampilkan hasil (hindari lagu yang dipilih sendiri)
    st.subheader("Lagu yang Direkomendasikan:")
    for i in range(1, len(indices[0])):
        idx = indices[0][i]
        st.write(f"🎵 {music_data.iloc[idx]['title']}")


