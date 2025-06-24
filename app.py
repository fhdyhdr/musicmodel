import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('music_model.pkl')

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('music_genre.csv')
    return df

df = load_data()

# Cek apakah kolom track_name dan genre ada
if 'track_name' not in df.columns:
    st.error("Dataset tidak memiliki kolom 'track_name'.")
    st.stop()

# Tentukan fitur numerik yang sesuai
feature_columns = ['danceability', 'energy', 'key', 'loudness', 'mode',
                   'speechiness', 'acousticness', 'instrumentalness',
                   'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']

# Pastikan semua kolom fitur tersedia
missing_cols = [col for col in feature_columns if col not in df.columns]
if missing_cols:
    st.error(f"Kolom berikut hilang dari dataset: {missing_cols}")
    st.stop()

X = df[feature_columns]

# UI
st.title("🎵 Music Recommendation System")
st.write("Pilih lagu berdasarkan nama, dan kami akan merekomendasikan lagu serupa berdasarkan fitur audio.")

# Pilih berdasarkan nama lagu
selected_track = st.selectbox("Pilih Lagu:", df['track_name'].unique().tolist())

# Ambil index dari lagu terpilih
selected_index = df[df['track_name'] == selected_track].index[0]

# Tombol rekomendasi
if st.button("Rekomendasikan Lagu Serupa"):
    try:
        selected_features = X.loc[selected_index].values.reshape(1, -1)
        distances, indices = model.kneighbors(selected_features, n_neighbors=6)

        st.subheader("🎧 Lagu yang Direkomendasikan:")
        for i in indices[0][1:]:  # skip lagu itu sendiri
            track = df.loc[i, 'track_name']
            genre = df.loc[i, 'genre'] if 'genre' in df.columns else 'Unknown'
            st.markdown(f"- **{track}** ({genre})")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mencari rekomendasi: {e}")













