import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('music_model.pkl')

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('music_genre.csv')
    except UnicodeDecodeError:
        df = pd.read_csv('music_genre.csv', encoding='latin1')
    return df

# Ambil data mentah
df_raw = load_data()

# Normalisasi kolom (hapus spasi, lowercase)
df_raw.columns = [col.strip().lower().replace(" ", "_") for col in df_raw.columns]

# Cek dan tangani kolom track_name
track_col = 'track_name'
if track_col not in df_raw.columns:
    st.error(f"❌ Kolom '{track_col}' tidak ditemukan. Kolom tersedia: {df_raw.columns.tolist()}")
    st.stop()

# Hapus baris kosong pada track_name
df_full = df_raw[df_raw[track_col].notna()].copy()

# Tampilkan isi kolom untuk debug jika perlu
# st.write("📋 Kolom:", df_full.columns.tolist())

# Deteksi fitur numerik otomatis (tipe numerik saja)
df_numeric = df_full.select_dtypes(include=[np.number])

# Tangani kolom genre jika ada
if 'genre' in df_full.columns:
    df_genre = pd.get_dummies(df_full['genre'], prefix='music_genre')
    df_features = pd.concat([df_numeric, df_genre], axis=1)
else:
    df_features = df_numeric.copy()

# Sinkronisasi dengan fitur dari model
try:
    model_features = list(model.feature_names_in_)
except AttributeError:
    st.error("❌ Model tidak menyimpan informasi fitur. Gunakan scikit-learn ≥ 1.0.")
    st.stop()

# Pastikan semua fitur tersedia
missing = [col for col in model_features if col not in df_features.columns]
if missing:
    st.error(f"❌ Fitur berikut hilang dari CSV: {missing}")
    st.stop()

# Susun ulang fitur agar urutannya sesuai dengan model
X_final = df_features[model_features]

# UI: Judul dan Pilihan Lagu
st.title("🎵 Music Recommendation System")
st.write("Pilih lagu berdasarkan nama, dan sistem akan merekomendasikan lagu serupa berdasarkan fitur audio.")

# Pilihan lagu berdasarkan nama
selected_track = st.selectbox("Pilih Lagu:", df_full[track_col].unique())

# Ambil index lagu yang dipilih
selected_index = df_full[df_full[track_col] == selected_track].index[0]

# Tombol rekomendasi
if st.button("Rekomendasikan Lagu Serupa"):
    try:
        selected_features = X_final.loc[selected_index].values.reshape(1, -1)
        distances, indices = model.kneighbors(selected_features, n_neighbors=6)

        st.subheader("🎧 Lagu Serupa yang Direkomendasikan:")
        for i in indices[0][1:]:  # Skip lagu itu sendiri
            title = df_full.iloc[i][track_col]
            genre = df_full.iloc[i]['genre'] if 'genre' in df_full.columns else 'Tidak diketahui'
            st.markdown(f"- 🎵 **{title}** | Genre: *{genre}*")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat mencari rekomendasi: {e}")

