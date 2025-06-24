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

# Kolom fitur numerik yang sesuai dengan pelatihan model
feature_columns = ['danceability', 'energy', 'loudness', 'speechiness', 
                   'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

X = df[feature_columns]

st.title("🎵 Music Recommendation System")
st.write("Pilih lagu (berdasarkan nomor baris), dan kami akan merekomendasikan lagu serupa berdasarkan fitur audio.")

# Dropdown: pilih berdasarkan indeks baris
selected_index = st.selectbox("Pilih Lagu (Index):", df.index.tolist())

# Tombol rekomendasi
if st.button("Rekomendasikan Lagu Serupa"):
    selected_features = X.loc[selected_index].values.reshape(1, -1)
    distances, indices = model.kneighbors(selected_features, n_neighbors=6)

    st.subheader("🎧 Lagu yang Direkomendasikan:")
    for i in indices[0][1:]:
        genre = df.loc[i, 'genre']
        st.markdown(f"- Index: {i} | Genre: {genre}")











