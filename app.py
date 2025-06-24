import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('music_model.pkl')

# Load dataset asli
@st.cache_data
def load_music_data():
    df = pd.read_csv("music_genre.csv")
    return df

df = load_music_data()

# Tampilkan judul
st.title("🎧 Music Recommendation App")
st.write("Pilih lagu dan dapatkan rekomendasi musik serupa berdasarkan fitur-fitur audio.")

# Pastikan kolom fitur sesuai model
feature_columns = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                   'instrumentalness', 'liveness', 'valence', 'tempo']
title_column = 'title'

# Validasi kolom ada
missing_cols = [col for col in feature_columns + [title_column] if col not in df.columns]
if missing_cols:
    st.error(f"Kolom berikut tidak ditemukan dalam dataset: {missing_cols}")
    st.stop()

# Dropdown lagu
song_titles = df[title_column].dropna().unique()
selected_song = st.selectbox("Pilih Lagu", song_titles)

# Ambil data fitur dari lagu yang dipilih
selected_index = df[df[title_column] == selected_song].index[0]
selected_features = df.loc[selected_index, feature_columns].values.reshape(1, -1)

# KNN: cari lagu terdekat
try:
    distances, indices = model.kneighbors(selected_features, n_neighbors=6)
except ValueError as e:
    st.error(f"Error saat mencari tetangga terdekat: {str(e)}")
    st.stop()

# Tampilkan rekomendasi (kecuali lagu itu sendiri)
st.subheader("🎵 Rekomendasi Lagu Serupa:")
for i in indices[0]:
    if i != selected_index:
        recommended_title = df.iloc[i][title_column]
        st.write(f"- {recommended_title}")





