# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import euclidean_distances

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('songs.csv')  # columns: 'track_name', 'artist', 'cluster', feature1, feature2, ...
    return df
    
df = load_data()

st.title("🎵 Taylor Swift Music Recommender")
st.write("Discover similar songs from Taylor Swift's discography using audio features & KMeans clustering.")

# Song selection
song_list = df['name']#.unique()
selected_song = st.selectbox("Choose a Taylor Swift song:", sorted(song_list))

# Get features of the selected song
selected_song_data = df[df['name'] == selected_song].iloc[0]
#selected_features = selected_song_data.drop(['name', 'cluster'])

# Compute similarity
#feature_columns = df.columns.difference(['name', 'cluster'])
#similarities = cosine_similarity([selected_features], df[feature_columns])[0]

# Add similarity scores
#df['similarity'] = similarities
#recommendations = df[df['name'] != selected_song].sort_values(by='similarity', ascending=False).head(5)

#member_index = df.index[df['name'] == selected_song].tolist()[0]
member_index = df.index[df['name'] == selected_song].tolist()[0]

cluster_id = df.loc[member_index, 'cluster']
same_cluster_data = df[df['cluster'] == cluster_id]

if len(same_cluster_data) < 11:
    suggestions = list(df[df['cluster'] == cluster_id]['name'])
else:
    numeric_cols = same_cluster_data.select_dtypes(include = ['number']).columns
    numeric_vals = same_cluster_data.select_dtypes(include = ['number']).values

    # Compute distances between the selected member and others
    selected_member_data = numeric_vals[same_cluster_data.index.get_loc(member_index)].reshape(1, -1)
    distances = euclidean_distances(selected_member_data, numeric_vals).flatten()

    nearest_indices = np.argsort(distances)[1:11]

    nearest_members = same_cluster_data.iloc[nearest_indices]

    # Display or save the results
    recommendations = nearest_members['name'].tolist()
    
suggestion_list  =[]
for song in recommendations:
    if song == selected_song: continue
    suggestion_list.append([song, list(df[df['name'] == song]['album'])[0]])

# Display results
st.subheader("🎧 Songs You Might Like:")
#st.write(df.head())
for i in len(suggestion_list):
#    st.write(f"**{row['name']} — Similarity: {row['similarity']:.2f}")
    st.write(f"**suggestion_list[i]")
    
    


