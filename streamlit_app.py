# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('songs.csv')  # columns: 'track_name', 'artist', 'cluster', feature1, feature2, ...
    return df

df = load_data()

st.title("🎵 Taylor Swift Music Recommender")
st.write("Discover similar songs from Taylor Swift's discography using audio features & KMeans clustering.")

# Song selection
song_list = df['name'].unique()
selected_song = st.selectbox("Choose a Taylor Swift song:", sorted(song_list))

# Get features of the selected song
selected_song_data = df[df['name'] == selected_song].iloc[0]
selected_features = selected_song_data.drop(['name', 'cluster'])

# Compute similarity
feature_columns = df.columns.difference(['name', 'cluster'])
similarities = cosine_similarity([selected_features], df[feature_columns])[0]

# Add similarity scores
df['similarity'] = similarities
recommendations = df[df['name'] != selected_song].sort_values(by='similarity', ascending=False).head(5)

# Display results
st.subheader("🎧 Songs You Might Like:")
for idx, row in recommendations.iterrows():
    st.write(f"**{row['name']} — Similarity: {row['similarity']:.2f}")


