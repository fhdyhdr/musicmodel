import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load dataset
df = pd.read_csv('music_genre.csv')

# Pilih hanya fitur yang dipakai di Streamlit app
feature_columns = ['danceability', 'energy', 'loudness', 'speechiness',
                   'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Target dummy: kita butuh target untuk fit() walaupun rekomendasi nanti tidak pakai target
# Jadi kita pakai genre saja jika tersedia
if 'genre' in df.columns:
    y = df['genre']
else:
    # Jika tidak ada, buat dummy target
    y = [0] * len(df)

X = df[feature_columns]

# Latih model KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# Simpan model
joblib.dump(knn, 'music_model.pkl')

print("Model berhasil dilatih dan disimpan sebagai music_model.pkl")







