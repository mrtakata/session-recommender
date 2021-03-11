'''
Created on 10.10.2017

@author: ludewig
'''
import scipy.sparse as sparse
import pandas as pd
import numpy as np
import time
import gensim
import sys
import json
sys.path.append('../')
from evaluation import loader as loader

# data_path = '../../data/retailrocket/single/'
# file_prefix = 'events'
# FOLDER = '../../data/retailrocket/prepared2d/'

data_path = '../../data/rsc15/single/'
file_prefix = 'rsc15-clicks'
FOLDER = '../../data/rsc15/prepared2d/'
# FILE = 'yoochoose-clicks-full'

def create_w2v_features( train, size=10, pos=False ):
    
    start = time.time()
    
    train['ItemId'] = train['ItemId'].astype('str')
    print( train['ItemId'].min() )
    
    sequences = train.groupby('SessionId')['ItemId'].apply(list)

    print('prepared features in ',(time.time() - start))
    
    # Learn decompositon ----------------------------------------------------------------
    print('ITEM2VEC FEATURES')
    start = time.time()
    
    model = gensim.models.Word2Vec(sequences, size=size, window=5, min_count=1, workers=4, iter=50)
    print('Model trained in ',(time.time() - start))
    weights = model.wv.vectors
    np.save(open(FOLDER+'w2v.'+str(size)+'.wght', 'wb'), weights)
    
    vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    with open(FOLDER+'w2v.'+str(size)+'.voc', 'w') as f:
        f.write(json.dumps(vocab))
  
if __name__ == '__main__':
    
    train, test = loader.load_data(data_path, file_prefix,
                                slice_num=None, rows_train=None,
                                rows_test=None, density=1)
    # train, test = ld.load_data(FOLDER, FILE)
    # create_latent_factors( combi, size=32, pos=False )
    create_w2v_features( train, size=64 )
    