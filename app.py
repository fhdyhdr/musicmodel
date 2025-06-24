import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Gunakan joblib yang lebih aman untuk model sklearn

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('music_model.pkl')  # Gunakan joblib
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_model()

# Feature names based on the model
feature_names = [
    'popularity', 'acousticness', 'danceability', 'duration_ms', 
    'energy', 'instrumentalness', 'liveness', 'loudness', 
    'speechiness', 'tempo', 'valence'
]

# Music genres the model can predict
genres = [
    'Anime', 'Blues', 'Classical', 'Country', 
    'Electronic', 'Hip-Hop', 'Jazz', 'Rap'
]

# Create the Streamlit app
st.title('🎵 Music Genre Recommendation System')
st.write("""
This app predicts the music genre based on audio features.
Adjust the sliders to set the audio features and get a genre prediction.
""")

# Sidebar with user input features
st.sidebar.header('Input Audio Features')

def user_input_features():
    popularity = st.sidebar.slider('Popularity', 0, 100, 50)
    acousticness = st.sidebar.slider('Acousticness', 0.0, 1.0, 0.5)
    danceability = st.sidebar.slider('Danceability', 0.0, 1.0, 0.5)
    duration_ms = st.sidebar.slider('Duration (ms)', 0, 600000, 180000, step=1000)
    energy = st.sidebar.slider('Energy', 0.0, 1.0, 0.5)
    instrumentalness = st.sidebar.slider('Instrumentalness', 0.0, 1.0, 0.5)
    liveness = st.sidebar.slider('Liveness', 0.0, 1.0, 0.5)
    loudness = st.sidebar.slider('Loudness (dB)', -60.0, 0.0, -20.0)
    speechiness = st.sidebar.slider('Speechiness', 0.0, 1.0, 0.5)
    tempo = st.sidebar.slider('Tempo (BPM)', 0.0, 250.0, 120.0)
    valence = st.sidebar.slider('Valence (Positivity)', 0.0, 1.0, 0.5)
    
    data = {
        'popularity': popularity,
        'acousticness': acousticness,
        'danceability': danceability,
        'duration_ms': duration_ms,
        'energy': energy,
        'instrumentalness': instrumentalness,
        'liveness': liveness,
        'loudness': loudness,
        'speechiness': speechiness,
        'tempo': tempo,
        'valence': valence
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the user input features
st.subheader('User Input Features')
st.write(input_df)

if model:
    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Prediction')
    predicted_genre = genres[prediction[0]]
    st.write(f'Predicted Genre: **{predicted_genre}**')

    # Show prediction probabilities
    st.subheader('Prediction Probability')
    proba_df = pd.DataFrame({
        'Genre': genres,
        'Probability': prediction_proba[0]
    })
    proba_df = proba_df.sort_values('Probability', ascending=False)

    # Display as bar chart
    st.bar_chart(proba_df.set_index('Genre'))

    # Genre descriptions
    genre_descriptions = {
        'Anime': 'Music from Japanese anime, often featuring high-energy pop or orchestral elements.',
        'Blues': 'Rooted in African-American traditions, featuring soulful vocals and guitar work.',
        'Classical': 'Orchestral and instrumental music with complex compositions.',
        'Country': 'Storytelling songs with acoustic guitars and folk influences.',
        'Electronic': 'Created with electronic instruments and technology, often for dancing.',
        'Hip-Hop': 'Rhythmic music with rhyming speech (rapping) and strong beats.',
        'Jazz': 'Improvisational music with swing and blue notes, complex chords.',
        'Rap': 'Rhythmic and rhyming speech performed over beats.'
    }

    st.subheader('About the Predicted Genre')
    st.write(genre_descriptions[predicted_genre])

    # Sample artists
    sample_artists = {
        'Anime': ['Yoko Kanno', 'Hiroyuki Sawano', 'Lisa', 'Aimer'],
        'Blues': ['B.B. King', 'Muddy Waters', 'Etta James', 'Robert Johnson'],
        'Classical': ['Mozart', 'Beethoven', 'Bach', 'Chopin'],
        'Country': ['Johnny Cash', 'Dolly Parton', 'Willie Nelson', 'Taylor Swift'],
        'Electronic': ['Daft Punk', 'The Chemical Brothers', 'Deadmau5', 'Aphex Twin'],
        'Hip-Hop': ['Kendrick Lamar', 'Nas', 'OutKast', 'Wu-Tang Clan'],
        'Jazz': ['Miles Davis', 'John Coltrane', 'Ella Fitzgerald', 'Duke Ellington'],
        'Rap': ['Eminem', 'Jay-Z', 'Tupac', 'Notorious B.I.G.']
    }

    st.subheader('Sample Artists in This Genre')
    st.write(", ".join(sample_artists[predicted_genre]))

# Footer
st.markdown("""
---
Built with ❤️ using Streamlit  
Model: KNeighborsClassifier  
Features: Spotify audio features
""")

