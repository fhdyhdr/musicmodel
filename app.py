import streamlit as st
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors

# Load the pre-trained model
@st.cache_resource
def load_model():
    with open('music_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Feature names (from the model)
feature_names = [
    'popularity', 'acousticness', 'danceability', 'duration_ms', 
    'energy', 'instrumentalness', 'liveness', 'loudness', 
    'speechiness', 'tempo', 'valence'
]

# Music genres (from the model)
genres = [
    'Anime', 'Blues', 'Classical', 'Country', 
    'Electronic', 'Hip-Hop', 'Jazz', 'Rap'
]

# Create the Streamlit app
st.title('🎵 Music Recommendation System')

st.write("""
This app recommends music tracks based on your preferences for various audio features.
Adjust the sliders to set your preferences and get recommendations!
""")

# Sidebar with user input
st.sidebar.header('Your Music Preferences')

# Create sliders for each feature
user_input = {}
for feature in feature_names:
    if feature == 'duration_ms':
        # Convert duration from ms to minutes for display
        max_val = 10 * 60 * 1000  # 10 minutes in ms
        default_val = 3 * 60 * 1000  # 3 minutes in ms
        value = st.sidebar.slider(
            f'{feature} (minutes)',
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=0.5
        )
        user_input[feature] = value * 60 * 1000  # Convert back to ms
    elif feature == 'tempo':
        # Tempo typically ranges from 50 to 200 BPM
        user_input[feature] = st.sidebar.slider(
            f'{feature} (BPM)',
            min_value=50.0,
            max_value=200.0,
            value=120.0,
            step=1.0
        )
    else:
        # Most features are between 0 and 1
        user_input[feature] = st.sidebar.slider(
            feature,
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01
        )

# Genre preference
selected_genre = st.sidebar.selectbox('Preferred Genre', genres)

# Convert user input to array format for the model
input_array = np.array([[user_input[feature] for feature in feature_names]])

# Add genre preference (one-hot encoded)
genre_array = np.zeros(len(genres))
genre_index = genres.index(selected_genre)
genre_array[genre_index] = 1
input_array = np.concatenate([input_array, genre_array.reshape(1, -1)], axis=1)

# Recommendation button
if st.sidebar.button('Get Recommendations'):
    # Get nearest neighbors
    distances, indices = model.kneighbors(input_array, n_neighbors=5)
    
    st.subheader('Recommended Tracks for You')
    
    # Display recommendations (in a real app, you'd have actual track data)
    for i, (distance, index) in enumerate(zip(distances[0], indices[0])):
        st.write(f"Recommendation #{i+1} (distance: {distance:.2f})")
        
        # Display feature values for the recommendation
        recommended_features = model._fit_X[index]
        
        # Split features and genre probabilities
        rec_features = recommended_features[:len(feature_names)]
        rec_genres = recommended_features[len(feature_names):]
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Audio Features:**")
            for feature, value in zip(feature_names, rec_features):
                st.write(f"- {feature}: {value:.3f}")
        
        with col2:
            st.write("**Genre Probabilities:**")
            for genre, prob in zip(genres, rec_genres):
                if prob > 0.5:  # Only show likely genres
                    st.write(f"- {genre}: {prob:.0%}")
        
        st.write("---")

# Add some app info
st.sidebar.markdown("""
---
### About
This recommendation system uses a k-nearest neighbors algorithm to find tracks with similar audio characteristics to your preferences.

Adjust the sliders to explore different musical styles!
""")

