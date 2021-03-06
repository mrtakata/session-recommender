# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:57:27 2015

@author: mludewig
"""

import numpy as np
import pandas as pd
from math import log10
import collections as col
from datetime import datetime as dt
from datetime import timedelta as td

class SequentialRules: 
    '''
    SequentialRules( steps = 10, weighting='div', last_n_days=None, session_key='SessionId', item_key='ItemId', time_key='Time' )
    
    Additional Version that does not support pruning, but instead extends the rule set while giving predictions. 
    
    Parameters
    --------
    steps : int
        Number of steps to walk back from the currently viewed item. (Default value: 10)
    weighting : string
        Weighting function for the previous items (linear, same, div, log, qudratic). (Default value: div)
    last_n_days : int
        Only use the last N days of the data for the training process. (Default value: None)
    session_key : string
        The data frame key for the session identifier. (Default value: SessionId)
    item_key : string
        The data frame key for the item identifier. (Default value: ItemId)
    time_key : string
        The data frame key for the timestamp. (Default value: Time)
    
    '''
    
    def __init__( self, steps = 10, weighting='div', last_n_days=None, session_key='SessionId', item_key='ItemId', time_key='Time' ):
        self.steps = steps
        self.weighting = weighting
        self.last_n_days = last_n_days
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.session = -1
        self.session_items = []
            
    def fit( self, data ):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        
            
        '''
        
        if self.last_n_days != None:
            
            max_time = dt.fromtimestamp( data[self.time_key].max() )
            date_threshold = max_time.date() - td( self.last_n_days )
            stamp = dt.combine(date_threshold, dt.min.time()).timestamp()
            train = data[ data[self.time_key] >= stamp ]
        
        else: 
            train = data
            
        cur_session = -1
        last_items = []
        rules = dict()
        
        index_session = train.columns.get_loc( self.session_key )
        index_item = train.columns.get_loc( self.item_key )
        
        for row in train.itertuples( index=False ):
            
            session_id, item_id = row[index_session], row[index_item]
            
            if session_id != cur_session:
                cur_session = session_id
                last_items = []
            else: 
                for i in range( 1, self.steps+1 if len(last_items) >= self.steps else len(last_items)+1 ):
                    prev_item = last_items[-i]
                    
                    if not prev_item in rules :
                        rules[prev_item] = dict()
                    
                    if not item_id in rules[prev_item]:
                        rules[prev_item][item_id] = 0
                    
                    rules[prev_item][item_id] += getattr(self, self.weighting)( i )
                    
            last_items.append(item_id)
            
        self.rules = rules
    
    def linear(self, i):
        return 1 - (0.1*i) if i <= 100 else 0
    
    def same(self, i):
        return 1
    
    def div(self, i):
        return 1/i
    
    def log(self, i):
        return 1/(log10(i+1.7))
    
    def quadratic(self, i):
        return 1/(i*i)
    
    def predict_next(self, session_id, input_item_id, predict_for_item_ids, skip=False, type='view', timestamp=0):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''
        if session_id != self.session:
            self.session_items = []
            self.session = session_id
        
        if type == 'view':
            self.session_items.append( input_item_id )
            
        if skip:
            return
        
        preds = np.zeros( len(predict_for_item_ids) ) 
             
        if input_item_id in self.rules:
            for key in self.rules[input_item_id]:
                preds[ predict_for_item_ids == key ] = self.rules[input_item_id][key]
        
        series = pd.Series(data=preds, index=predict_for_item_ids)
        
        series = series / series.max()
        
        #extend
        last_items = self.session_items[:-1]
        for i in range( 1, self.steps+1 if len(last_items) >= self.steps else len(last_items)+1 ):
            prev_item = last_items[-i]
            
            if not prev_item in self.rules :
                self.rules[prev_item] = dict()
            
            if not input_item_id in self.rules[prev_item]:
                self.rules[prev_item][input_item_id] = 0
            
            self.rules[prev_item][input_item_id] += getattr(self, self.weighting)( i )
        
        
        return series      