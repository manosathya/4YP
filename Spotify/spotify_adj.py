import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pandas as  pd
from os.path import join
splits = ["train","val","test"]

for split in splits:
    f = open(join("Spotify", "all_data_" + split + ".pkl"),'rb')
    all_data = pkl.load(f)
    
    headers = ['track_id', 'genre', 'album_id', 'set', 'artist_id']
    dataset = pd.read_csv(join("msdi-data","splits", "MSD-I_dataset.tsv"), header = 0, names = headers + [''] , sep = '\t', usecols = headers)
    data_split = dataset[dataset.set == split]
    
    genre_df = dict()
    tid_hash = dict()
    i=0
    x = []
    count = 0
    for k, v in data_split.groupby('genre'):
        for idx in range(len(v)):
            tid_hash[v.index[idx]] = count
            count+=1
    c = 0
    length = len([val for sublist in all_data['artist_tracks'] for val in sublist])
    A = np.zeros((length,length))
    for artist in all_data['name']:
        
        try:
            artist_tracks = all_data['artist_tracks'][c]
            artist_tracks = [val for sublist in artist_tracks for val in sublist]
        except:
            pass
        try:
            related_idx = all_data['related_track_idx'][artist]
            related_idx = [val for sublist in related_idx for val in sublist]
        except:
            pass
        for i in artist_tracks:
            for j in related_idx:
                A[tid_hash[i],tid_hash[j]] = 1
                A[tid_hash[j],tid_hash[i]] = 1
        c+=1 
    np.save(join("spotify_adj_" + split + ".npy"), np.int8(A))