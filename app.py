import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model dan data
model = joblib.load('music_model.pkl')
data = pd.read_csv('final_music_genre.csv')

# Judul aplikasi
st.title("🎵 Music Genre Recommendation App")
st.write("Masukkan fitur-fitur musik untuk mendapatkan prediksi genre!")

# Ambil hanya kolom fitur dari data (tanpa label)
feature_columns = [col for col in data.columns if col != 'label']  # sesuaikan 'label' jika kolom genre bernama lain

# Form input fitur
user_input = {}
for col in feature_columns:
    # Asumsikan semua fitur numerik, bisa diubah jika perlu
    val = st.number_input(f"{col}", min_value=float(data[col].min()), max_value=float(data[col].max()), value=float(data[col].mean()))
    user_input[col] = val

# Prediksi jika tombol ditekan
if st.button("Rekomendasikan Genre Musik"):
    input_array = np.array([list(user_input.values())])
    prediction = model.predict(input_array)[0]
    st.success(f"🎧 Genre Musik yang Direkomendasikan: **{prediction}**")


