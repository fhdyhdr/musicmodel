import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("music_model.pkl")
n_features = model.n_features_in_  # Jumlah fitur yang diharapkan

# Contoh dummy data lagu — HARUS kamu sesuaikan jika punya dataset asli
music_data = pd.DataFrame({
    'title': [
        'Let It Be', 'Bohemian Rhapsody', 'Imagine', 'Hotel California', 'Smells Like Teen Spirit',
        'Billie Jean', 'Shape of You', 'Blinding Lights', 'Someone Like You', 'Lose Yourself'
    ],
    # Contoh: jika model butuh 5 fitur, tambahkan lebih banyak fitur dummy
    'feature1': [0.5, 0.8, 0.6, 0.7, 0.9, 0.85, 0.4, 0.6, 0.3, 0.7],
    'feature2': [0.7, 0.6, 0.5, 0.8, 0.9, 0.9, 0.4, 0.5, 0.6, 0.8],
    'feature3': [0.3, 0.9, 0.4, 0.6, 0.95, 0.8, 0.35, 0.65, 0.45, 0.75],
    'feature4': [0.1, 0.3, 0.2, 0.6, 0.5, 0.7, 0.4, 0.2, 0.3, 0.6],
    'feature5': [0.4, 0.6, 0.2, 0.5, 0.8, 0.6, 0.3, 0.5, 0.2, 0.7],
})

# Ambil nama kolom fitur sesuai jumlah yang dibutuhkan model
all_feature_cols = [col for col in music_data.columns if col != 'title']
feature_columns = all_feature_cols[:n_features]

# Streamlit UI
st.set_page_config(page_title="Music Recommender", layout="centered")
st.title("🎵 Music Recommendation System")

selected_song = st.selectbox("Pilih lagu favoritmu:", music_data['title'].tolist())

if st.button("Rekomendasikan Lagu 🎧"):
    try:
        selected_index = music_data[music_data['title'] == selected_song].index[0]
        selected_features = music_data.loc[selected_index, feature_columns].values.reshape(1, -1)

        distances, indices = model.kneighbors(selected_features)

        st.subheader("Lagu yang Direkomendasikan:")
        for i in range(1, len(indices[0])):
            idx = indices[0][i]
            st.write(f"🎵 {music_data.iloc[idx]['title']}")
    except Exception as e:
        st.error(f"Gagal merekomendasikan lagu: {e}")
