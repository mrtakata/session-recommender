from _operator import itemgetter
from math import sqrt
from .cknn import ContextKNN
import random
import time

import numpy as np
import pandas as pd
from math import log10


class SeqContextKNN(ContextKNN):
    '''
    SeqContextKNN(k, sample_size=500, sampling='recent', similarity='cosine',
                  remind=False, pop_boost=0, session_key='SessionId',
                  item_key='ItemId')

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
        String to define the method for the similarity calculation (jaccard,
        cosine, binary, tanimoto). (default: cosine)
    weighting: string
        weighting function used to calculate score. Default: div
    remind : bool
        Should the last items of the current session be boosted to the top as
        reminders.
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
    '''

    def __init__(self, k, sample_size=1000, sampling='recent',
                 similarity='cosine', weighting='div', remind=False,
                 pop_boost=0, extend=False, normalize=True,
                 session_key='SessionId', item_key='ItemId', time_key='Time',
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
        )
        self.weighting = weighting

    def score_items(self, neighbors):
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
            step = 1

            for item in reversed(self.session_items):
                if item in items:
                    decay = getattr(self, self.weighting)(step)
                    break
                step += 1

            for item in items:
                old_score = scores.get(item, 0)
                similarity = session[1]
                new_score = old_score + (similarity * decay)
                scores.update({item: new_score})

        return scores


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
