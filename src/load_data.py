# when we are at the current song and the user likes it.
# we recommend the 10 closest songs to it.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

class MusicRecommender():

    scaler = StandardScaler() # initialise the standard scaler


    """
    Loads the song metadata file and returns a DataFrame with one row per track
    and all the features from the dataset.
    """
    def load_meta(self, path):
        df = pd.read_csv(path)
        print(df.info())
        return df


    """
    Returns a dataframe with one row per track and all the numeric features from data set with the track_id to uniquely identify
    """
    def loadFeatures(self, path, features=['track_id', 'popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']):
        df = pd.read_csv(path)
        track_ids = df['track_id']  # keep the track ids seperate
        numericFeatures = [feature for feature in features[1:]] # contains all features other than track id
        scaled_data = self.scaler.fit_transform(df[numericFeatures])
        X = scaled_data
        print("Shape of X: " + str(X.shape))
        print("X: " + str(X))
        print("mean: " + str(X.mean(axis=0)))
        print("standard deviation: " + str(X.std(axis=0)))

        return df

musicRecommender = MusicRecommender()

meta_df = musicRecommender.load_meta('data/raw/dataset.csv')
feats_df = musicRecommender.loadFeatures('data/raw/dataset.csv')

# loudness needs to be normalised
# tempo needs to be normalised
# popularity needs to be normalised