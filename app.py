import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('music_model.pkl')

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('music_genre.csv')
    return df

df = load_data()

# Ambil semua kolom numerik (kecuali 'genre' jika ingin tampilkan)
feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
if 'duration_ms' in feature_columns and 'duration_ms' not in model.feature_names_in_:
    feature_columns.remove('duration_ms')  # contoh pengecualian

# Pastikan urutan kolom sama dengan saat pelatihan
feature_columns = [col for col in model.feature_names_in_]  # pakai atribut dari model

X = df[feature_columns]

st.title("🎵 Music Recommendation System")
st.write("Pilih lagu berdasarkan nomor baris, dan kami akan merekomendasikan lagu serupa.")

selected_index = st.selectbox("Pilih Lagu (Index):", df.index.tolist())

if st.button("Rekomendasikan Lagu Serupa"):
    try:
        selected_features = X.loc[selected_index].values.reshape(1, -1)
        distances, indices = model.kneighbors(selected_features, n_neighbors=6)

        st.subheader("🎧 Lagu yang Direkomendasikan:")
        for i in indices[0][1:]:  # skip lagu itu sendiri
            genre = df.loc[i, 'genre'] if 'genre' in df.columns else 'Unknown'
            st.markdown(f"- Index: {i} | Genre: {genre}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mencari rekomendasi: {e}")














