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

# Ambil fitur numerik dari model
feature_columns = list(model.feature_names_in_)
X = df[feature_columns]

# Pastikan track_name tersedia
if 'track_name' not in df.columns:
    st.error("Kolom 'track_name' tidak ditemukan dalam dataset.")
    st.stop()

st.title("🎵 Music Recommendation System")
st.write("Pilih lagu berdasarkan nama, dan kami akan merekomendasikan lagu serupa berdasarkan fitur audio.")

# Pilih berdasarkan nama lagu
selected_track = st.selectbox("Pilih Lagu:", df['track_name'].unique())

# Cari index lagu berdasarkan track_name
selected_index = df[df['track_name'] == selected_track].index[0]

if st.button("Rekomendasikan Lagu Serupa"):
    try:
        selected_features = X.loc[selected_index].values.reshape(1, -1)
        distances, indices = model.kneighbors(selected_features, n_neighbors=6)

        st.subheader("🎧 Lagu Serupa yang Direkomendasikan:")
        for i in indices[0][1:]:  # Skip lagu itu sendiri
            title = df.iloc[i]['track_name']
            genre = df.iloc[i]['genre'] if 'genre' in df.columns else 'Tidak diketahui'
            st.markdown(f"- 🎵 **{title}** | Genre: *{genre}*")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mencari rekomendasi: {e}")
















