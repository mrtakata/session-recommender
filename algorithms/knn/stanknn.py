from math import sqrt
import random
import time
import numpy as np
import pandas as pd
from .cknn import ContextKNN as sknn
from .helpers import similarities
from .helpers.contexts import *
from math import exp


class STANContextKNN(sknn):
    """
    STANContextKNN(k, sample_size=500, sampling='recent',
               similarity='jaccard', remind=False, pop_boost=0,
               session_key='SessionId', item_key='ItemId')

    This is the STAN-KNN implemented by Matheus Takata based on the paper
    (https://dl.acm.org/doi/abs/10.1145/3331184.3331322).

    Parameters
    -----------
    k : int
        Number of neighboring session to calculate the item scores from.
        (Default value: 100)
    sample_size : int
        Defines the length of a subset of all training sessions to calculate
        the nearest neighbors from. (Default value: 500)
    sampling : string
        String to define the sampling method for sessions (recent, random).
        (default: recent)
    similarity : string
        String to define the method for the similarity calculation
        (jaccard, cosine, binary, tanimoto). (default: jaccard)
    remind : bool
        Should the last items of the current session be boosted to the top
        as reminders
    pop_boost : int
        Push popular items in the neighbor sessions by this factor.
        (default: 0 to leave out)
    extend : bool
        Add evaluated sessions to the maps
    normalize : bool
        Normalize the scores in the end
    session_key : string
        Header of the session ID column in the input file.
        (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file. (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file. (default: 'Time')
    """
    def __init__(
        self,
        k,
        sample_size=1000,
        sampling="recent",
        similarity="cosine",
        remind=False,
        pop_boost=0,
        extend=False,
        normalize=True,
        session_key="SessionId",
        item_key="ItemId",
        time_key="Time",
        lmbd1=1,
        lmbd2=1,
        lmbd3=1,
    ):
        super().__init__(
            k,
            sample_size,
            sampling,
            similarity,
            remind,
            pop_boost,
            extend,
            normalize,
            session_key,
            item_key,
            time_key,
        )
        self.lmbd1 = lmbd1
        self.lmbd2 = lmbd2
        self.lmbd3 = lmbd3

    def fit(self, train, items=None):
        """
        Trains the predictor.

        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions.
            It has one column for session IDs, one for item IDs and one for
            the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must
            correspond to the ones you set during the initialization of
            the network (session_key, item_key, time_key properties).

        """

        index_session = train.columns.get_loc(self.session_key)
        index_item = train.columns.get_loc(self.item_key)
        index_time = train.columns.get_loc(self.time_key)

        session = -1
        session_items = []
        time = -1

        for row in train.itertuples(index=False):
            # cache items of sessions
            if row[index_session] != session:
                if len(session_items) > 0:
                    self.session_item_map.update({session: session_items})
                    self.session_time.update({session: time})
                session = row[index_session]
                session_items = []
            time = row[index_time]
            session_items.append(row[index_item])

            # cache sessions involving an item
            self.update_item_session_map(row[index_item], row[index_session])

        # Add the last tuple
        self.session_item_map.update({session: session_items})
        self.session_time.update({session: time})

    def weight_recency(self, position, session_length):
        return exp((position - session_length) / self.lmbd1)

    def weight_timestamp(self, time1, time2):
        # print(time1, time2, -(time1 - time2)/20/86400)
        return exp(-(time1 - time2)/20/86400)

    def weight_item_proximity(self, pos, ideal_pos):
        return exp(-abs(pos - ideal_pos)/self.lmbd3)

    def similarity_recency(self, session1, session2):
        numerator = 0
        session1_len = len(self.session_items)
        for item in session2:
            if item in self.session_items:
                numerator += self.weight_recency(self.session_items.index(item) + 1, session1_len)
        denominator = sqrt(session1_len * len(session2))
        return numerator/denominator

    def calc_similarity(self, session_items, sessions):
        """
        Calculates the configured similarity for the items in session_items
        and each session in sessions.

        Parameters
        --------
        session_items: set of item ids
        sessions: list of session ids

        Returns
        --------
        out : list of tuple (session_id,similarity)
        """

        neighbors = []
        for session in sessions:
            # get items of the session, look up the cache first
            session_items_test = self.session_item_map[session]
            session_items_test = set(session_items_test)
            session_time = self.session_time[session]
            similarity = getattr(similarities, self.similarity)(
                session_items_test, session_items
            )
            # similarity = self.similarity_recency(self.session_items, session_items_test)
            similarity *= self.weight_timestamp(self.time, session_time)
            if similarity > 0:
                neighbors.append((session, similarity))

        return neighbors

    def score_items(self, neighbors):
        """
        Compute a set of scores for all items given a set of neighbors.

        Parameters
        --------
        neighbors: set of session ids

        Returns
        --------
        out : list of tuple (item, score)
        """
        # now we have the set of relevant items to make predictions
        scores = {}
        last_item = self.session_items[-1]
        # iterate over the sessions
        for session_id, item_score in neighbors:
            # get the items in this session
            items = self.session_item_map[session_id]

            for item in items:
                score = scores.get(item, 0)
                index_item = items.index(item)
                if last_item in items:
                    index_last_item = items.index(last_item)
                    w3 = self.weight_item_proximity(index_item, index_last_item)
                    # print(w3)
                    scores[item] = score + item_score * w3

        return scores
