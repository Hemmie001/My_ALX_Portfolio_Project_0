#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split

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

# Display top songs by popularity
print(song_gr.sort_values(['listen_count', 'song'], ascending=[0, 1]))

# Split data into training and testing sets
train_data, test_data = train_test_split(song_df1, test_size=0.20, random_state=0)
print(train_data.head(5))

# Class for Popularity based Recommender System model
class PopularityRecommender:
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.popularity_recommendations = None

    # Create the popularity based recommender system model
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        # Get a count of user_ids for each unique item as recommendation score
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        train_data_grouped.rename(columns={'user_id': 'score'}, inplace=True)

        # Sort the items based upon recommendation score
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending=[0, 1])

        # Generate a recommendation rank based upon score
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')

        # Get the top 10 recommendations
        self.popularity_recommendations = train_data_sort.head(10)

    # Use the popularity based recommender system model to make recommendations
    def recommend(self, user_id):
        user_recommendations = self.popularity_recommendations

        # Add user_id column for which the recommendations are being generated
        user_recommendations['user_id'] = user_id

        # Bring user_id column to the front
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]

        return user_recommendations

# Instantiate the popularity recommender
popularity_recommender = PopularityRecommender()

# Create the popularity recommender model
popularity_recommender.create(train_data, 'user_id', 'song_id')

# Get recommendations for a user
user_recommendations = popularity_recommender.recommend(user_id='some_user_id')
print(user_recommendations)
