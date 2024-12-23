import pandas as pd
from urllib import request
from gensim.models import Word2Vec
import numpy as np

def print_recommendations(song_id, model, song_df):
    similar_songs = np.array(model.wv.most_similar(positive=str(song_id), topn=5))[:,0]
    return songs_df.iloc[similar_songs]

# Get the playlist dataset file
data = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt')

# Parse the playlist dataset. Skip the first two lines as they only contain metada.
lines = data.read().decode('utf-8').split('\n')[2:]

# Remove playlists with only one song
playlists = [s.rstrip().split() for s in lines if len(s.split()) > 1]

# Load song metadata
songs_file = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt')
songs_file = songs_file.read().decode('utf-8').split('\n')
songs = [s.rstrip().split('\t') for s in songs_file]
songs_df = pd.DataFrame(data=songs, columns=['id', 'title', 'artist'])
songs_df = songs_df.set_index('id')

#print( 'Playlist #1:\n ', playlists[0], '\n')
#print( 'Playlist #2:\n ', playlists[1])

model = Word2Vec(sentences=playlists, vector_size=32, window=20, negative=50, min_count=1, workers=8)

song_id = 2172

print(songs_df.iloc[2172])

# Ask the model for songs similar to song #2172.
# print(model.wv.most_similar(positive=str(song_id)))

print(print_recommendations(song_id, model, songs_df))