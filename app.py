import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model KNN dari file pickle
model = joblib.load('music_model.pkl')

# Simulasi data fitur musik (kamu bisa ganti dengan dataset aslinya)
# Misalnya kita punya 100 lagu dengan 5 fitur
# Kalau kamu punya data asli, ganti bagian ini dengan membaca dari CSV
@st.cache_data
def load_music_features():
    # Contoh dummy dataset: ganti dengan dataset aslimu
    np.random.seed(42)
    df = pd.DataFrame(np.random.rand(100, 5), columns=['Danceability', 'Energy', 'Valence', 'Tempo', 'Acousticness'])
    df['Title'] = [f"Song {i+1}" for i in range(len(df))]
    return df

df_features = load_music_features()

# UI Streamlit
st.title("🎧 Music Recommendation App")
st.write("Pilih lagu untuk mendapatkan rekomendasi musik serupa.")

# Pilihan lagu dari judul
selected_song = st.selectbox("Pilih Lagu", df_features['Title'].tolist())

# Cari index lagu yang dipilih
selected_index = df_features[df_features['Title'] == selected_song].index[0]

# Ambil fitur dari lagu yang dipilih
selected_features = df_features.iloc[selected_index][['Danceability', 'Energy', 'Valence', 'Tempo', 'Acousticness']].values.reshape(1, -1)

# Prediksi lagu serupa
distances, indices = model.kneighbors(selected_features, n_neighbors=6)  # +1 untuk menyertakan diri sendiri

# Tampilkan hasil (kecuali diri sendiri)
st.subheader("🎵 Rekomendasi Lagu Serupa:")
for i in indices[0]:
    if i != selected_index:
        st.write(f"- {df_features.iloc[i]['Title']}")




