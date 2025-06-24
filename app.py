import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model KNN
model = joblib.load('music_model.pkl')

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('music_genre.csv')
    except UnicodeDecodeError:
        df = pd.read_csv('music_genre.csv', encoding='latin1')
    return df

df_raw = load_data()

# Normalisasi nama kolom
df_raw.columns = [col.strip().lower() for col in df_raw.columns]

# Pastikan track_name ada
if 'track_name' not in df_raw.columns:
    st.error("❌ Kolom 'track_name' tidak ditemukan.")
    st.stop()

# Hapus baris tanpa track_name
df_full = df_raw[df_raw['track_name'].notna()].copy()

# Pisahkan kolom fitur numerik
numeric_cols = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo']

if not all(col in df_full.columns for col in numeric_cols + ['genre']):
    st.error("❌ Beberapa kolom numerik atau 'genre' tidak tersedia di CSV.")
    st.stop()

X_num = df_full[numeric_cols]

# One-hot encode genre
X_genre = pd.get_dummies(df_full['genre'], prefix="music_genre")

# Gabungkan fitur akhir
X_full = pd.concat([X_num, X_genre], axis=1)

# Sinkronkan urutan kolom dengan model
try:
    model_features = list(model.feature_names_in_)
except AttributeError:
    st.error("❌ Model tidak menyimpan fitur. Gunakan scikit-learn >= 1.0.")
    st.stop()

# Cek apakah semua fitur yang dibutuhkan model tersedia
missing_features = [f for f in model_features if f not in X_full.columns]
if missing_features:
    st.error(f"❌ Fitur berikut tidak ditemukan: {missing_features}")
    st.stop()

# Susun X sesuai urutan fitur model
X_final = X_full[model_features]

# UI
st.title("🎵 Music Recommendation System")
st.write("Pilih lagu berdasarkan nama, dan sistem akan merekomendasikan lagu serupa berdasarkan fitur audio dan genre.")

selected_track = st.selectbox("Pilih Lagu:", df_full['track_name'].unique())

# Cari index
selected_index = df_full[df_full['track_name'] == selected_track].index[0]

if st.button("Rekomendasikan Lagu Serupa"):
    try:
        selected_features = X_final.loc[selected_index].values.reshape(1, -1)
        distances, indices = model.kneighbors(selected_features, n_neighbors=6)

        st.subheader("🎧 Lagu Serupa yang Direkomendasikan:")
        for i in indices[0][1:]:  # Skip lagu itu sendiri
            title = df_full.iloc[i]['track_name']
            genre = df_full.iloc[i]['genre'] if 'genre' in df_full.columns else 'Tidak diketahui'
            st.markdown(f"- 🎵 **{title}** | Genre: *{genre}*")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat rekomendasi: {e}")
