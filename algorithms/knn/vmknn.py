from _operator import itemgetter
from math import sqrt
import random
import time
from .cknn import ContextKNN as sknn

from pympler import asizeof
import numpy as np
import pandas as pd
from math import log10
from datetime import datetime as dt
from datetime import timedelta as td
import math


class VMContextKNN(sknn):
    '''
    VMContextKNN( k, sample_size=1000, sampling='recent', similarity='cosine', weighting='div', dwelling_time=True, last_n_days=None, last_n_clicks=None, extend=False, weighting_score='div_score', weighting_time=False, normalize=True, session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time')

    Parameters
    -----------
    k : int
        Number of neighboring session to calculate the item scores from. (Default value: 100)
    sample_size : int
        Defines the length of a subset of all training sessions to calculate the nearest neighbors from. (Default value: 500)
    sampling : string
        String to define the sampling method for sessions (recent, random). (default: recent)
    similarity : string
        String to define the method for the similarity calculation (jaccard, cosine, binary, tanimoto). (default: jaccard)
    weighting : string
        Decay function to determine the importance/weight of individual actions in the current session (linear, same, div, log, quadratic). (default: div)
    weighting_score : string
        Decay function to lower the score of candidate items from a neighboring sessions that were selected by less recently clicked items in the current session. (linear, same, div, log, quadratic). (default: div_score)
    weighting_time : boolean
        Experimental function to give less weight to items from older sessions (default: True)
    dwelling_time : boolean
        Experimental function to use the dwelling time for item view actions as a weight in the similarity calculation. (default: False)
    last_n_days : int
        Use only data from the last N days. (default: None)
    last_n_clicks : int
        Use only the last N clicks of the current session when recommending. (default: None)
    extend : bool
        Add evaluated sessions to the maps.
    normalize : bool
        Normalize the scores in the end.
    session_key : string
        Header of the session ID column in the input file. (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file. (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file. (default: 'Time')
    '''

    def __init__( self, k, sample_size=1000, sampling='recent', similarity='cosine',
                  weighting='div', dwelling_time=False, last_n_days=None, last_n_clicks=None,
                  extend=False, weighting_score='div_score', weighting_time=False, normalize=True,
                  session_key='SessionId', item_key='ItemId', time_key='Time' ):

        super().__init__(
            k=k,
            sample_size=sample_size,
            sampling=sampling,
            similarity=similarity,
            # remind=remind,
            # pop_boost=pop_boost,
            extend=extend,
            normalize=normalize,
            session_key=session_key,
            item_key=item_key,
            time_key=time_key,
        )

        self.last_n_days = last_n_days
        self.last_n_clicks = last_n_clicks

        self.weighting = weighting
        self.dwelling_time = dwelling_time
        self.weighting_score = weighting_score
        self.weighting_time = weighting_time


    def predict_next( self, session_id, input_item_id, predict_for_item_ids, skip=False, type='view', timestamp=0):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.

        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.

        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.

        '''
        self.num_recommendations += 1
        if( self.session != session_id ): #new session

            if( self.extend ):
                item_set = set( self.session_items )
                self.session_item_map[self.session] = item_set
                for item in item_set:
                    map_is = self.item_session_map.get( item )
                    if map_is is None:
                        map_is = set()
                        self.item_session_map.update({item : map_is})
                    map_is.add(self.session)

                ts = time.time()
                self.session_time.update({self.session : ts})

            self.last_ts = -1
            self.session = session_id
            self.session_items = list()
            self.dwelling_times = list()
            self.relevant_sessions = set()

        if type == 'view':
            self.session_items.append( input_item_id )
            if self.dwelling_time:
                if self.last_ts > 0:
                    self.dwelling_times.append( timestamp - self.last_ts )
            self.last_ts = timestamp
        if skip:
            return

        items = self.session_items if self.last_n_clicks is None else self.session_items[-self.last_n_clicks:]
        neighbors = self.find_neighbors( items, input_item_id, session_id, self.dwelling_times, timestamp )
        scores = self.score_items( neighbors, items, timestamp )

        # Create things in the format ..
        predictions = np.zeros(len(predict_for_item_ids))
        mask = np.in1d( predict_for_item_ids, list(scores.keys()) )

        items = predict_for_item_ids[mask]
        values = [scores[x] for x in items]
        predictions[mask] = values
        series = pd.Series(data=predictions, index=predict_for_item_ids)

        if self.normalize:
            series = series / series.max()

        return series


    def vec(self, first, second, map):
        '''
        Calculates the ? for 2 sessions

        Parameters
        --------
        first: Id of a session
        second: Id of a session

        Returns
        --------
        out : float value
        '''
        a = first & second
        sum = 0
        for i in a:
            sum += map[i]

        result = sum / len(map)

        return result


    def vec_for_session(self, session):
        '''
        Returns all items in the session

        Parameters
        --------
        session: Id of a session

        Returns
        --------
        out : set
        '''
        return self.session_vec_map.get(session);


    def calc_similarity(self, session_items, sessions, dwelling_times, timestamp ):
        '''
        Calculates the configured similarity for the items in session_items and each session in sessions.

        Parameters
        --------
        session_items: set of item ids
        sessions: list of session ids

        Returns
        --------
        out : list of tuple (session_id,similarity)
        '''

        pos_map = {}
        length = len(session_items)
        count = 1
        for item in session_items:
            if self.weighting is not None:
                pos_map[item] = getattr(self, self.weighting)( count, length )
                count += 1
            else:
                pos_map[item] = 1

        items = set(session_items)
        neighbors = []
        for session in sessions:
            # get items of the session, look up the cache first
            n_items = self.session_item_map[session]

            similarity = self.vec(items, n_items, pos_map)
            if similarity > 0:
                neighbors.append((session, similarity))


        return neighbors


    def find_neighbors( self, session_items, input_item_id, session_id, dwelling_times, timestamp ):
        '''
        Finds the k nearest neighbors for the given session_id and the current item input_item_id.

        Parameters
        --------
        session_items: set of item ids
        input_item_id: int
        session_id: int

        Returns
        --------
        out : list of tuple (session_id, similarity)
        '''
        possible_neighbors = self.possible_neighbor_sessions( session_items, input_item_id, session_id )
        possible_neighbors = self.calc_similarity( session_items, possible_neighbors, dwelling_times, timestamp )

        possible_neighbors = sorted( possible_neighbors, reverse=True, key=lambda x: x[1] )
        possible_neighbors = possible_neighbors[:self.k]

        return possible_neighbors


    def score_items(self, neighbors, current_session, timestamp):
        '''
        Compute a set of scores for all items given a set of neighbors.

        Parameters
        --------
        neighbors: set of session ids

        Returns
        --------
        out : list of tuple (item, score)
        '''
        # now we have the set of relevant items to make predictions
        scores = dict()
        # iterate over the sessions
        for session_id, item_score in neighbors:
            # get the items in this session
            items = self.session_item_map[session_id]
            # step = 1

            # for item in reversed(current_session):
            #     if item in items:
            #         decay = getattr(self, self.weighting_score)(step)
            #         break
            #     step += 1

            for item in items:
                score = scores.get(item, 0)
                scores[item] = score + item_score

                # old_score = scores.get(item)
                # similarity = session[1]

                # if old_score is None:
                #     scores.update({item : (similarity * decay)})
                # else:
                #     new_score = old_score + (similarity * decay)
                #     scores.update({item : new_score})

        return scores


    def linear_score(self, i):
        return 1 - (0.1*i) if i <= 100 else 0

    def same_score(self, i):
        return 1

    def div_score(self, i):
        return 1/i

    def log_score(self, i):
        return 1/(log10(i+1.7))

    def quadratic_score(self, i):
        return 1/(i*i)

    def linear(self, i, length):
        return 1 - (0.1*(length-i)) if i <= 10 else 0

    def same(self, i, length):
        return 1

    def div(self, i, length):
        return i/length

    def log(self, i, length):
        return 1/(log10((length-i)+1.7))

    def quadratic(self, i, length):
        return (i/length)**2
