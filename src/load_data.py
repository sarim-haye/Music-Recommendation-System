# when we are at the current song and the user likes it.
# we recommend the 10 closest songs to it.
# direction matters more than magnitude, so we will use cosine similarity to recommend K nearest neighbors

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import heapq

class MusicRecommender():

    def __init__(self):
        self.scaler = StandardScaler() # initialise the standard scaler
        self.scaled_features = None
        self.track_ids = None
        self.features_df = None

    """ 
    Returns the cosine of the angle between two vectors
    """
    def cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) 
    
    """
    Returns the track ids of the 10 most similar songs to the track that the user says they like
    """
    def get_recommendations(self, track_id, n_recommendations=10):
        minHeap = []
        n_closest_tracks = []
        # 1. Find the index of the target song in your dataset

        idx = np.where(self.track_ids == track_id)[0][0]

        # 2. Get its feature vector

        feature_vector = self.scaled_features[idx]

        # 3. Compare it to ALL other songs using cosine_similarity function and maintain a minHeap

        for i in range(len(self.scaled_features)):


            # Don't want to include the song itself that we are comparing to in our heap
            if i == idx:
                continue

            # Push to heap the cosine similarity and the track id if we haven't reached capacity
            if len(minHeap) < n_recommendations:
                heapq.heappush(minHeap, (self.cosine_similarity(feature_vector, self.scaled_features[i]), self.track_ids[i]))

            else:
                if self.cosine_similarity(feature_vector, self.scaled_features[i]) > minHeap[0][0]:
                    heapq.heappop(minHeap)
                    heapq.heappush(minHeap, (self.cosine_similarity(feature_vector, self.scaled_features[i]), self.track_ids[i]))
        
        # Add the n closest tracks into an array and return result

        for sim, track_id in minHeap:
            n_closest_tracks.append((sim, track_id))
        
        print(n_closest_tracks)
        print(self.features_df)

        for _, track_id in n_closest_tracks:
            song_info = self.features_df[self.features_df['track_id'] == track_id]
            song_name = song_info['track_name'].iloc[0]
            artist_name = song_info['artists'].iloc[0]

            print("song: " + str(song_name) + " artist: " + str(artist_name))
            
        
        return n_closest_tracks


    """
    Loads the song metadata file and returns a DataFrame with one row per track
    and all the features from the dataset.
    """
    def load_meta(self, path):
        df = pd.read_csv(path)
        return df


    """
    Returns a dataframe with one row per track and all the numeric features from data set with the track_id to uniquely identify
    Standardises all the numeric features
    """
    def loadFeatures(self, path, features=['track_id', 'popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']):
        df = pd.read_csv(path)
        df = df.drop_duplicates(subset=['track_id'], keep='first') # remove any duplicate track_id data since it is the same
        self.features_df = df # Keep this for song names, artists etc
        self.track_ids = df['track_id'].values  # Store (np array of) track IDs
        numeric_features = [feature for feature in features[1:]] # contains all features other than track id
        self.scaled_features = self.scaler.fit_transform(df[numeric_features]) # standardise numeric features and store the mean and standard deviation used to standardise


        """
        Below is only for debugging purposes
        """
        # Create a new dataframe with scaled features
        scaled_df = pd.DataFrame(self.scaled_features, columns=numeric_features)
        scaled_df['track_id'] = self.track_ids # creates the track_id column with the track ids that we previously stored

        scaled_df.set_index('track_id', inplace=True)
        print(scaled_df.info())


        # print("Shape of X: " + str(X.shape))
        # print("X: " + str(X))
        # print("mean: " + str(X.mean(axis=0)))
        # print("standard deviation: " + str(X.std(axis=0)))

        return scaled_df


musicRecommender = MusicRecommender()

meta_df = musicRecommender.load_meta('data/raw/dataset.csv')
feats_df = musicRecommender.loadFeatures('data/raw/dataset.csv')
musicRecommender.get_recommendations('5SuOikwiRyPMVoIQDJUgSV')