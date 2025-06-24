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

# Ambil kolom fitur numerik untuk input model
feature_columns = ['danceability', 'energy', 'loudness', 'speechiness', 
                   'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
X = df[feature_columns]

st.title("🎵 Music Recommendation System")
st.write("Pilih lagu, dan kami akan merekomendasikan lagu serupa berdasarkan fitur audio.")

# Pilih lagu dari nama
selected_song = st.selectbox("Pilih Lagu:", df['song_name'])

# Cari indeks dari lagu terpilih
song_index = df[df['song_name'] == selected_song].index[0]

# Rekomendasi saat tombol diklik
if st.button("Rekomendasikan Lagu Serupa"):
    # Ambil fitur dari lagu yang dipilih
    selected_features = X.loc[song_index].values.reshape(1, -1)

    # Cari tetangga terdekat (termasuk dirinya sendiri)
    distances, indices = model.kneighbors(selected_features, n_neighbors=6)

    st.subheader("🎧 Lagu yang Direkomendasikan:")
    for i in indices[0][1:]:  # [1:] untuk menghindari lagu itu sendiri
        st.markdown(f"- {df.loc[i, 'song_name']} ({df.loc[i, 'genre']})")









