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

# Ambil fitur numerik yang sesuai dengan model
feature_columns = [col for col in model.feature_names_in_]  # dari model saat training
X = df[feature_columns]

st.title("🎵 Music Recommendation System")
st.write("Pilih lagu berdasarkan nomor baris, dan kami akan merekomendasikan lagu serupa.")

selected_index = st.selectbox("Pilih Lagu (Index):", df.index.tolist())

if st.button("Rekomendasikan Lagu Serupa"):
    try:
        selected_features = X.loc[selected_index].values.reshape(1, -1)
        distances, indices = model.kneighbors(selected_features, n_neighbors=6)

        st.subheader("🎧 Lagu yang Direkomendasikan:")
        for i in indices[0][1:]:  # Skip lagu itu sendiri
            genre = df.iloc[i]['genre'] if 'genre' in df.columns else 'Tidak diketahui'
            st.markdown(f"- Index: {i} | Genre: {genre}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")















