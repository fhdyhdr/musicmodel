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

# Cek apakah kolom track_name ada
if 'track_name' not in df.columns:
    st.error("❌ Kolom 'track_name' tidak ditemukan dalam file CSV.")
    st.stop()

# Cek apakah ada nilai NaN pada track_name
df = df.dropna(subset=['track_name'])

# Ambil fitur yang digunakan oleh model
feature_columns = list(model.feature_names_in_)
X = df[feature_columns]

st.title("🎵 Music Recommendation System")
st.write("Pilih lagu berdasarkan nama, dan kami akan merekomendasikan lagu serupa.")

selected_track = st.selectbox("Pilih Lagu:", df['track_name'].unique())

# Cari index dari track yang dipilih
selected_index = df[df['track_name'] == selected_track].index[0]

if st.button("Rekomendasikan Lagu Serupa"):
    try:
        selected_features = X.loc[selected_index].values.reshape(1, -1)
        distances, indices = model.kneighbors(selected_features, n_neighbors=6)

        st.subheader("🎧 Lagu Serupa yang Direkomendasikan:")
        for i in indices[0][1:]:
            track = df.iloc[i]['track_name']
            genre = df.iloc[i]['genre'] if 'genre' in df.columns else 'Tidak diketahui'
            st.markdown(f"- 🎵 **{track}** | Genre: *{genre}*")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mencari rekomendasi: {e}")

















