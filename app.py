import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model KNN
model = joblib.load('music_model.pkl')

# Load data lengkap
@st.cache_data
def load_data():
    df = pd.read_csv('music_genre.csv')
    return df

df_full = load_data()

# Pastikan kolom track_name ada
if 'track_name' not in df_full.columns:
    st.error("❌ Kolom 'track_name' tidak ditemukan di dalam dataset.")
    st.stop()

# Hapus baris dengan nilai kosong di track_name
df_full = df_full[df_full['track_name'].notna()]

# Ambil nama kolom fitur dari model
try:
    feature_columns = list(model.feature_names_in_)
except AttributeError:
    st.error("❌ Model tidak menyimpan informasi fitur. Pastikan menggunakan scikit-learn 1.0+ dan model disimpan setelah training.")
    st.stop()

# Ambil hanya kolom fitur
try:
    df_features = df_full[feature_columns]
except KeyError as e:
    st.error(f"❌ Kolom berikut tidak ditemukan di CSV: {e}")
    st.stop()

# Sinkronkan indeks df_features dan df_full
df_features = df_features.loc[df_full.index]

# UI Streamlit
st.title("🎵 Music Recommendation System")
st.write("Pilih lagu berdasarkan nama, dan dapatkan rekomendasi lagu serupa berdasarkan fitur audio.")

# Dropdown lagu
selected_track = st.selectbox("Pilih Lagu:", df_full['track_name'].unique())

# Ambil index dari lagu yang dipilih
selected_index = df_full[df_full['track_name'] == selected_track].index[0]

# Tombol untuk rekomendasi
if st.button("Rekomendasikan Lagu Serupa"):
    try:
        selected_features = df_features.loc[selected_index].values.reshape(1, -1)
        distances, indices = model.kneighbors(selected_features, n_neighbors=6)

        st.subheader("🎧 Lagu Serupa yang Direkomendasikan:")
        for i in indices[0][1:]:  # Skip lagu itu sendiri
            title = df_full.iloc[i]['track_name']
            genre = df_full.iloc[i]['genre'] if 'genre' in df_full.columns else 'Tidak diketahui'
            st.markdown(f"- 🎵 **{title}** | Genre: *{genre}*")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat mencari rekomendasi: {e}")



















