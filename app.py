import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('music_model.pkl')

st.set_page_config(page_title="Music Recommendation", layout="centered")

st.title("🎵 Music Recommendation App")
st.write("Masukkan informasi untuk mendapatkan rekomendasi genre musik.")

# Contoh input: fitur numerik berdasarkan dataset pelatihan
# Silakan sesuaikan label input sesuai dengan fitur model Anda
age = st.slider("Umur", 10, 80, 25)
gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
education = st.selectbox("Pendidikan Terakhir", ["SMA", "Diploma", "S1", "S2/S3"])

# Preprocessing sederhana (ubah ke numerik)
gender_num = 1 if gender == "Laki-laki" else 0
education_map = {
    "SMA": 0,
    "Diploma": 1,
    "S1": 2,
    "S2/S3": 3
}
education_num = education_map[education]

# Buat input array untuk prediksi
user_input = np.array([[age, gender_num, education_num]])

# Tombol prediksi
if st.button("Rekomendasikan Genre Musik"):
    prediction = model.predict(user_input)
    st.success(f"🎧 Rekomendasi Genre Musik: **{prediction[0]}**")


