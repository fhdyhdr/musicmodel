import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model dan data
model = joblib.load("music_model.pkl")

# Misalkan kita punya fitur dan daftar lagu
# Ini perlu disesuaikan dengan fitur yang digunakan saat training
# Contoh data dummy (ganti dengan data asli jika ada)
music_data = pd.DataFrame({
    'title': ['Lagu A', 'Lagu B', 'Lagu C', 'Lagu D', 'Lagu E'],
    'feature1': [0.1, 0.2, 0.15, 0.4, 0.3],
    'feature2': [0.8, 0.7, 0.75, 0.5, 0.6],
    'feature3': [0.05, 0.07, 0.06, 0.03, 0.04]
})

# Ekstrak fitur dan judul lagu
X = music_data.drop(columns=['title']).values
titles = music_data['title'].values

st.title("🎵 Sistem Rekomendasi Musik")
st.write("Pilih sebuah lagu, dan sistem akan merekomendasikan musik yang serupa.")

# Pilihan lagu
selected_title = st.selectbox("Pilih Lagu:", titles)

# Cari index lagu yang dipilih
if selected_title:
    selected_index = np.where(titles == selected_title)[0][0]
    selected_features = X[selected_index].reshape(1, -1)

    # Temukan rekomendasi menggunakan model KNN
    distances, indices = model.kneighbors(selected_features, n_neighbors=4)  # 1 lagu asli + 3 rekomendasi

    # Tampilkan rekomendasi
    st.subheader("🎧 Rekomendasi Musik:")
    for idx in indices[0]:
        if idx != selected_index:  # Hindari menampilkan lagu yang dipilih
            st.write(f"- {titles[idx]}")



