import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('music_model.pkl')

# Load dataset lengkap
@st.cache_data
def load_data():
    df = pd.read_csv('music_genre.csv')
    return df

df_full = load_data()

# Buat dataframe fitur numerik (sesuai model)
feature_columns = list(model.feature_names_in_)
df_features = df_full[feature_columns]

# Drop baris NaN dari track_name (untuk dropdown)
df_full = df_full.dropna(subset=['track_name'])
df_features = df_features.loc[df_full.index]  # sinkronkan index

st.title("🎵 Music Recommendation System")
st.write("Pilih lagu berdasarkan nama, dan kami akan merekomendasikan lagu serupa berdasarkan fitur audio.")

selected_track = st.selectbox("Pilih Lagu:", df_full['track_name'].unique())

# Ambil index dari lagu pilihan
selected_index = df_full[df_full['track_name'] == selected_track].index[0]

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
        st.error(f"Terjadi kesalahan saat mencari rekomendasi: {e}")


















