# when we are at the current song and the user likes it.
# we recommend the 10 closest songs to it.
# direction matters more than magnitude, so we will use cosine similarity to recommend K nearest neighbors

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

class MusicRecommender():

    def __init__(self):
        self.scaler = StandardScaler() # initialise the standard scaler
        self.features_df = None
        self.scaled_features = None
        self.track_ids = None

    """ 
    Returns the cosine of the angle between two vectors
    """
    def cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) 
    
    """
    Returns the track ids of the 10 most similar songs to the track that the user says they like
    """
    def get_recommendations(self, track_id, n_recommendations=10):



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
    Standardises all the numeric features
    """
    def loadFeatures(self, path, features=['track_id', 'popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']):
        df = pd.read_csv(path)
        self.track_ids = df['track_id'].values  # Store (np array of) track IDs
        print(self.track_ids)
        numeric_features = [feature for feature in features[1:]] # contains all features other than track id
        self.scaled_features = self.scaler.fit_transform(df[numeric_features]) # standardise numeric features and store the mean and standard deviation used to standardise

        # Create a new dataframe with scaled features
        scaled_df = pd.DataFrame(self.scaled_features, columns=numeric_features)
        scaled_df['track_id'] = self.track_ids # creates the track_id column with the track ids that we previously stored


        # print("Shape of X: " + str(X.shape))
        # print("X: " + str(X))
        # print("mean: " + str(X.mean(axis=0)))
        # print("standard deviation: " + str(X.std(axis=0)))

        self.features_df = df # Store original dataframe
        return scaled_df


musicRecommender = MusicRecommender()

meta_df = musicRecommender.load_meta('data/raw/dataset.csv')
feats_df = musicRecommender.loadFeatures('data/raw/dataset.csv')