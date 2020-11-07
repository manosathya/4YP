import pandas as pd
import sqlite3
from os.path import join
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from tqdm import tqdm
from functools import partial
import pickle as pkl
import numpy as np
tqdm100 = partial(tqdm, ncols=100)

"""Load Data"""
headers = ['track_id', 'genre', 'album_id', 'set', 'artist_id']
dataset = pd.read_csv(join("msdi-data","splits", "MSD-I_dataset.tsv"), header = 0, names = headers + [''] , sep = '\t', usecols = headers)
split_df=dict()

for k, v in dataset.groupby('set'):
    split_df[k] = v
del k,v
#for split in ["train", "val", "test"]:
split = "val"
subset = []
dataset = split_df[split]
artist_track = []

for k, v in dataset.groupby('artist_id'):
    indexes = []
    subset.append(v['track_id'][v['track_id'].index[0]])
    for i in range(len(v)):
        indexes.append(v['track_id'].index[i])
    artist_track.append(indexes)

path_sql = "C:/Users/Mano/Downloads/track_metadata.db"
conn = sqlite3.connect(path_sql)
artist_names = []
cur = conn.cursor()
for i in range(len(subset)):
    cur.execute("SELECT * FROM songs WHERE track_id=?", (subset[i],))
    row = cur.fetchall()
    artist_names.append(row[0][6])


track_data = pd.DataFrame({'track_id': subset,
                            'artist_name': artist_names})

client_credentials_manager = SpotifyClientCredentials(client_id=, 
                                                      client_secret=)

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
uris =[]

for i in tqdm100(range(len(artist_names))):
    results = sp.search(q='artist:' + artist_names[i], type='artist')
    try:
        uris.append(results['artists']['items'][0]['uri'])
    except:
        uris.append(0)
art = dict()
art_idx = dict()
track_idx = dict()
for artist in tqdm100(range(len(uris))):
    try:
        h = sp.artist_related_artists(uris[artist])
    except:
        pass
    related =[]
    a_idx = []
    t_idx = []
    try:
        for count in range(len(h['artists'])):
            try:
                related_artists = h['artists'][count]['name']
                related_artists = related_artists 
                if related_artists in artist_names:
                    related.append(related_artists)
                    posn = artist_names.index(related_artists)
                    a_idx.append(posn)
                    t_idx.append(artist_track[posn])
                art[artist_names[artist]] = related  
                art_idx[artist_names[artist]] = a_idx     
                track_idx[artist_names[artist]] = t_idx 
            except:
                pass
    except:
        pass
    
all_data = {"name":artist_names,
            "artist_tracks": artist_track,
            "related_artist_idx": art_idx,
            "related_track_idx": track_idx,
            "related_names":art}       

f = open("all_data_" + split +".pkl","wb")
pkl.dump(all_data,f)
f.close()