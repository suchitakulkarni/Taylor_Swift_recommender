# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import euclidean_distances

st.set_page_config(layout="wide")  # Use full-width layout

# Load data
@st.cache_data
def load_data(filename):
    df = pd.read_csv(filename)  # columns: 'track_name', 'artist', 'cluster', feature1, feature2, ...
    return df
    
def draw_barplot(df, variable):
    temp_df = pd.DataFrame()
    temp_df['album'] = df['album'].astype(str)
    temp_df['size'] = df['size']
    temp_df['total_duration_ms'] = df['total_duration_ms']
    temp_df['speechiness'] = abs(df['speechiness_min']-album_summary['speechiness_max'])
    temp_df['loudness'] = abs(df['loudness_min'])-abs(album_summary['loudness_max'])
    temp_df['popularity'] = df['mean_popularity']
    temp_df = temp_df.sort_values(by=variable, ascending=False)
    p=sns.barplot(data=temp_df, y='album', x=variable)#, palette="Blues")
    p.set(ylabel=None)
    return 0

def draw_violin(df, variable):
    sns.set_theme(rc={'figure.figsize':(10,6)}, style = 'white')
    filtered_df = df.sort_values(by=variable, ascending=False)
    p=sns.violinplot(data = filtered_df, y=variable, x='album', palette="Blues")#, palette=color_palette)
    p.set_xticks(p.get_xticks());
    p.set_xticklabels(p.get_xticklabels(), rotation=90)
    if variable == 'instrumentalness': plt.yscale('log')
    if variable == 'speechiness': plt.yscale('log')
    p.set(xlabel=None)
    return 0
    
df = load_data('songs.csv')
df_album_summary = load_data('album_summary.csv')

st.title("🎵 Taylor Swift Music Recommender")
st.write("Discover similar songs from Taylor Swift's discography using audio features & KMeans clustering.")

# Divide page into two columns
col1, col2 = st.columns(2)

# === HISTOGRAM COLUMN ===
with col1:

    st.subheader("🎶 Overall feature Distribution Explorer")
    # Dropdown like ipywidgets interact
    selected_feature = st.selectbox(
        "Select a feature to visualize:",
        df.select_dtypes('number').columns
    )
    bins = st.slider("Number of bins:", min_value=5, max_value=100, value=30)

    # Plot output
    fig1, ax1 = plt.subplots(figsize=(2, 2))
    sns.histplot(df[selected_feature], kde=True, bins = bins, color='mediumpurple', ax=ax1)
    ax1.set_title(f"Distribution of {selected_feature}")
    st.pyplot(fig1)


# ===  CORRELATION COLUMN ===
with col2:
    criteria = ['size', 'total_duration_ms', 'speechiness', 'loudness','popularity']
    criteria2 = ['acousticness', 'danceability', 'energy', 'instrumentalness',
       'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'popularity',
       'duration_ms']
    critdict={'violinplot':criteria2,'barplot':criteria}
    #st.subheader("📈 Correlation Heatmap")
    st.subheader("📈 Features per album")
    selected_plt_type = 'barplot'
    selected_plt_type = st.selectbox(
        "Select plot type for plot:",
        ("barplot", "violinplot")
    )
    selected_feature = st.selectbox(
        "Select a feature to visualize:", critdict.get(selected_plt_type)
        )
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    if selected_plt_type == 'barplot': draw_barplot(df_album_summary, selected_feature)
    if selected_plt_type == 'violinplot': draw_violin(df_album_summary, selected_feature)
    #sns.heatmap(df.select_dtypes('number').corr(), annot=True, cmap="coolwarm", ax=ax2)
    #ax2.set_title("Feature Correlations")
    st.pyplot(fig2)

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
for i in range(len(suggestion_list)):
#    st.write(f"**{row['name']} — Similarity: {row['similarity']:.2f}")
    st.write(f"**{suggestion_list[i][0]} from album ({suggestion_list[i][1]})")
    
    


