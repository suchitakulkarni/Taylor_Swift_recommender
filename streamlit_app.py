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
    temp_df['speechiness'] = abs(df['speechiness_min']-df['speechiness_max'])
    temp_df['loudness'] = abs(df['loudness_min'])-abs(df['loudness_max'])
    temp_df['popularity'] = df['mean_popularity']
    temp_df = temp_df.sort_values(by=variable, ascending=False)
    p=sns.barplot(data=temp_df, y='album', x=variable)#, palette="Blues")
    p.set(ylabel=None)
    return 0

def draw_violin(df, variable):
    sns.set_theme(rc={'figure.figsize':(5,3)}, style = 'white')
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

st.title("🎵 Taylor Swift Discography analyzer and Music Recommender")
st.write("Discover features of Taylor's discography and find similar songs from using audio features & KMeans clustering.")

st.write("Let's look at the data summary")
st.write(f"{df_album_summary.head()}")
# Divide page into two columns

col1_head, col2_head = st.columns(2)
with col1_head:
    st.subheader("🎶 Taylor's most popular")
with col2_head:
    st.subheader("🎶 Overall feature Distribution Explorer")

col1, col2, col3 = st.columns(3)
# === HISTOGRAM COLUMN ===
with col1:
    
    fig0, ax0 = plt.subplots(figsize = (5,3))#, sharex=True)#, sharey=True)
    temp_df=pd.DataFrame()
    temp_df['album']=df_album_summary['album'].astype(str)
    temp_df['size']=df_album_summary['size']
    temp_df['total_duration_ms']=df_album_summary['total_duration_ms']
    temp_df['speechiness']= abs(df_album_summary['speechiness_min']-df_album_summary['speechiness_max'])
    temp_df['loudness']= abs(df_album_summary['loudness_min'])-abs(df_album_summary['loudness_max'])
    temp_df['popularity']= df_album_summary['mean_popularity']

    selected_summary = st.selectbox(
        "Select a feature to visualize:",
        ('size', 'total_duration_ms', 'speechiness', 'loudness','popularity'), key = 0
    )

    temp_df = temp_df.sort_values(by=[selected_summary])
    ax0.barh(temp_df['album'],temp_df[selected_summary])#, color = 'xkcd:sky blue')
    ax0.set_yticks(range(len(list(temp_df['album']))), temp_df['album'], fontsize=12)#, color = 'xkcd:steel');
    ax0.set_title(f"Albums with most {selected_summary}")
    st.pyplot(fig0)



# ===  CORRELATION COLUMN ===
with col2:
    col_left, col_right = st.columns(2)
    # Dropdown like ipywidgets interact
    with col_left:
        selected_feature = st.selectbox(
            "Select a feature to visualize:",
            df.select_dtypes('number').columns, key = 1
        )
    with col_right:
        bins = st.slider("Number of bins:", min_value=5, max_value=100, value=30)

    # Plot output
    fig1, ax1 = plt.subplots(figsize=(4, 2.4))
    sns.histplot(df[selected_feature], kde=True, bins = bins, color='mediumpurple', ax=ax1)
    ax1.set_title(f"{selected_feature} across all albums")
    st.pyplot(fig1)
    
with col3:
    #st.divider()
    #st.markdown("###")
    criteria = ['size', 'total_duration_ms', 'speechiness', 'loudness','popularity']
    criteria2 = ['acousticness', 'danceability', 'energy', 'instrumentalness',
       'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'popularity',
       'duration_ms']
    critdict={'violinplot':criteria2,'barplot':criteria}
    #st.subheader("📈 Correlation Heatmap")
    #st.text("📈 Features per album")
    col1_left, col1_right = st.columns(2)
    with col1_left:
        selected_plt_type = st.selectbox(
            "Select plot type for plot:",
            ("barplot", "violinplot"), key = 2
        )
    with col1_right:
        selected_feature_per_album = st.selectbox(
            "Select a feature to visualize:", critdict.get(selected_plt_type), key = 3
            )
    
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    if selected_plt_type == 'barplot': draw_barplot(df_album_summary, selected_feature_per_album)
    if selected_plt_type == 'violinplot': draw_violin(df, selected_feature_per_album)
    #sns.heatmap(df.select_dtypes('number').corr(), annot=True, cmap="coolwarm", ax=ax2)
    #ax2.set_title("Feature Correlations")
    ax2.set_title(f"{selected_feature_per_album} per album")
    st.pyplot(fig2)

# Song selection
st.subheader("Depending on your taste, we can recommend you more of Taylor's songs")
song_list = df['name']#.unique()
selected_song = st.selectbox("Choose a Taylor Swift song:", sorted(song_list), key = 4)

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
    st.write(f"**{suggestion_list[i][0]} from album ({suggestion_list[i][1]})")
    
    






