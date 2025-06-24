import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model
model = joblib.load('music_model.pkl')

# Coba ambil fitur dari model
try:
    feature_columns = model.feature_names_in_
except AttributeError:
    st.error("Model tidak memiliki atribut 'feature_names_in_'. Harap pastikan model dilatih dengan scikit-learn >=1.0.")
    st.stop()

# Judul aplikasi
st.title("🎵 Music Genre Recommendation App")
st.write("Masukkan fitur-fitur musik untuk mendapatkan prediksi genre!")

# Form input fitur
user_input = {}
for col in feature_columns:
    # Asumsi numerik, bisa disesuaikan untuk one-hot feature (0/1)
    default_val = 0.0
    if "music_genre_" in col:
        default_val = 0.0
        val = st.selectbox(f"{col}", [0, 1], index=0)
    else:
        val = st.number_input(f"{col}", value=default_val)
    user_input[col] = val

# Prediksi jika tombol ditekan
if st.button("Rekomendasikan Genre Musik"):
    input_array = np.array([list(user_input.values())])
    prediction = model.predict(input_array)[0]
    st.success(f"🎧 Genre Musik yang Direkomendasikan: **{prediction}**")



