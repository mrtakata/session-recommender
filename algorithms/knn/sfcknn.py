from _operator import itemgetter
from math import sqrt
from .cknn import ContextKNN
# from .wcknn import WindowContextKNN
import random
import time

import numpy as np
import pandas as pd


class SeqFilterContextKNN(ContextKNN):
    '''
    ContextKNN(k, sample_size=500, sampling='recent',  similarity='cosine', remind=False, pop_boost=0, session_key='SessionId', item_key= 'ItemId')

    Parameters
    -----------
    k : int
        Number of neighboring session to calculate the item scores from. (Default value: 100)
    sample_size : int
        Defines the length of a subset of all training sessions to calculate the nearest neighbors from. (Default value: 500)
    sampling : string
        String to define the sampling method for sessions (recent, random). (default: recent)
    similarity : string
        String to define the method for the similarity calculation (jaccard, cosine, binary, tanimoto). (default: cosine)
    remind : bool
        Should the last items of the current session be boosted to the top as reminders
    pop_boost : int
        Push popular items in the neighbor sessions by this factor. (default: 0 to leave out)
    extend : bool
        Add evaluated sessions to the maps
    normalize : bool
        Normalize the scores in the end
    session_key : string
        Header of the session ID column in the input file. (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file. (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file. (default: 'Time')
    '''

    def __init__(self, k, sample_size=1000, sampling='recent', similarity='cosine',
                 remind=False, pop_boost=0, extend=False, normalize=True,
                 session_key='SessionId', item_key='ItemId', time_key='Time',
                 window_type='hour'
                 ):
        super().__init__(
            k=k,
            sample_size=sample_size,
            sampling=sampling,
            similarity=similarity,
            remind=remind,
            pop_boost=pop_boost,
            extend=extend,
            normalize=normalize,
            session_key=session_key,
            item_key=item_key,
            time_key=time_key,
            # window_type=window_type
        )
        self.followed_by = {}
        self.sim_time = 0

    def fit(self, train, items=None):
        '''
        Trains the predictor.

        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).

        '''
        super().fit(train, items)
        index_session = train.columns.get_loc(self.session_key)
        index_item = train.columns.get_loc(self.item_key)

        session = -1
        last_item = -1
        for row in train.itertuples(index=False):
            if row[index_session] != session:
                session = row[index_session]
                last_item = -1
            if last_item != -1:  # fill followed by map for filtering of candidate items
                if not last_item in self.followed_by:
                    self.followed_by[last_item] = set()
                self.followed_by[last_item].add(row[index_item])
            last_item = row[index_item]

    def predict_next(self, session_id, input_item_id, predict_for_item_ids, skip=False, type='view', timestamp=0):
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

        # gc.collect()
        # process = psutil.Process(os.getpid())
        # print('cknn.predict_next: ', process.memory_info().rss, ' memory used')
        self.num_recommendations += 1
        if(self.session != session_id):  # new session

            if(self.extend):
                item_set = set(self.session_items)
                self.session_item_map[self.session] = item_set;
                for item in item_set:
                    map_is = self.item_session_map.get(item)
                    if map_is is None:
                        map_is = set()
                        self.item_session_map.update({item : map_is})
                    map_is.add(self.session)

                ts = time.time()
                self.session_time.update({self.session : ts})

                last_item = -1
                for item in self.session_items:
                    if last_item != -1:
                        if not last_item in self.followed_by:
                            self.followed_by[last_item] = set()
                        self.followed_by[last_item].add(item)
                    last_item = item


            self.session = session_id
            self.session_items = list()
            self.relevant_sessions = set()

        if type == 'view':
            self.session_items.append(input_item_id)

        if skip:
            return

        neighbors = self.find_neighbors(set(self.session_items), input_item_id, session_id)
        scores = self.score_items(neighbors, input_item_id)

        # add some reminders
        if self.remind:

            reminderScore = 5
            takeLastN = 3

            cnt = 0
            for elem in self.session_items[-takeLastN:]:
                cnt = cnt + 1
                #reminderScore = reminderScore + (cnt/100)

                oldScore = scores.get(elem)
                newScore = 0
                if oldScore is None:
                    newScore = reminderScore
                else:
                    newScore = oldScore + reminderScore
                #print 'old score ', oldScore
                # update the score and add a small number for the position
                newScore = (newScore * reminderScore) + (cnt/100)

                scores.update({elem : newScore})

        #push popular ones
        if self.pop_boost > 0:

            pop = self.item_pop(neighbors)
            # Iterate over the item neighbors
            #print itemScores
            for key in scores:
                item_pop = pop.get(key)
                # Gives some minimal MRR boost?
                scores.update({key : (scores[key] + (self.pop_boost * item_pop))})


        # Create things in the format ..
        predictions = np.zeros(len(predict_for_item_ids))
        mask = np.in1d(predict_for_item_ids, list(scores.keys()))

        items = predict_for_item_ids[mask]
        values = [scores[x] for x in items]
        predictions[mask] = values
        series = pd.Series(data=predictions, index=predict_for_item_ids)

        if self.normalize:
            series = series / series.max()

        return series


    def score_items(self, neighbors, input_item_id):
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
        for session in neighbors:
            # get the items in this session
            items = self.session_item_map[session[0]]

            for item in items:

                if input_item_id in self.followed_by and item in self.followed_by[input_item_id]:  # hard filter the candidates

                    old_score = scores.get(item)
                    new_score = session[1]

                    if old_score is None:
                        scores.update({item : new_score})
                    else:
                        new_score = old_score + new_score
                        scores.update({item: new_score})

        return scores
