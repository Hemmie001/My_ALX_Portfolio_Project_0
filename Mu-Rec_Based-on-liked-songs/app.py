#!/usr/bin/env python3

from flask import Flask, render_template
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/')
def landing_page():
    # Load user-song data
    triplets_file = 'user_song_data.csv'
    songs_metadata_file = 'song_data.csv'

    # Read user-song data
    song_df_a = pd.read_csv(triplets_file, header=None)
    song_df_a.columns = ['user_id', 'song_id', 'listen_count']

    # Read song metadata
    song_df_b = pd.read_csv(songs_metadata_file)

    # Merge dataframes
    song_df1 = pd.merge(song_df_a, song_df_b.drop_duplicates(['song_id']), on="song_id", how="left")

    # Merge song title and artist_name columns to make a new column
    song_df1['song'] = song_df1['title'].map(str) + " - " + song_df1['artist_name']

    # Group by song to get popularity statistics
    song_gr = song_df1.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
    grouped_sum = song_gr['listen_count'].sum()
    song_gr['percentage'] = (song_gr['listen_count'] / grouped_sum) * 100

    # Get top 10 songs by popularity
    top_songs = song_gr.sort_values(['listen_count', 'song'], ascending=[2, 1]).head(10)

    return render_template('landing_page.html', top_songs=top_songs)

if __name__ == '__main__':
    app.run(debug=True)

