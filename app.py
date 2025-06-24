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

# Kolom fitur numerik
feature_columns = ['danceability', 'energy', 'loudness', 'speechiness', 
                   'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
X = df[feature_columns]

st.title("🎵 Music Recommendation System")
st.write("Pilih lagu (berdasarkan ID), dan kami akan merekomendasikan lagu serupa.")

# Pilih lagu berdasarkan instance_id
selected_song_id = st.selectbox("Pilih Lagu (ID):", df['instance_id'])

# Cari indeks dari lagu terpilih
song_index = df[df['instance_id'] == selected_song_id].index[0]

# Rekomendasi saat tombol diklik
if st.button("Rekomendasikan Lagu Serupa"):
    selected_features = X.loc[song_index].values.reshape(1, -1)
    distances, indices = model.kneighbors(selected_features, n_neighbors=6)

    st.subheader("🎧 Lagu yang Direkomendasikan:")
    for i in indices[0][1:]:  # skip lagu itu sendiri
        st.markdown(f"- ID: {df.loc[i, 'instance_id']} | Genre: {df.loc[i, 'genre']}")










