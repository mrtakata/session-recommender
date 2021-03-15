import random
import numpy as np
import pandas as pd
from datetime import datetime as dt
import math

from .cknn import ContextKNN as sknn


class SessionKNN(sknn):
    """
    SessionKNN(
        k=500,
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
        window_type="hour",
        score_type="standard",
        weighting="div",
        lmbd=0.5,
        beta=0.5,
        use_of_context=None
    )

    This class extends the original S-KNN model by allowing different pre-filtering
    and post-filtering strategies using contextual information.

    Parameters
    -----------
    k : int
        Number of neighboring session to calculate the item scores from.
        (Default value: 500)
    sample_size : int
        Defines the length of a subset of all training sessions to calculate
        the nearest neighbors from. (Default value: 1000)
    sampling : string
        String to define the sampling method for sessions (recent, random, epcsr).
        (default: recent)
    similarity : string
        String to define the method for the similarity calculation
        (jaccard, cosine, binary, tanimoto). (default: cosine)
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
    window_type : string
        Type of temporal window to be used
        ('weekday', 'hour', 'period', 'weekend'). (default: 'weekday')
    score_type : string
        scoring method used ('standard', 'sequence', 'sequence_filter', 'context_double') (default: 'standard')
    weighting="div",
    lmbd : float
        if similarity is 'dsm', setups lambda value of function. (default: 0.5)
    beta : float
        if similarity is 'dsm', setups beta value of function. (default: 0.5)
    use_of_context : string
        usage of context. 'mandatory' uses context as a prefiltering strategy in all cases,
        'prefilter' uses context only if sample is greater than sample_size,
        None doesn't use context at all. (default: None)
    context_function: string
        Determines the score function when score_type is 'context'. (default: 'cficf')
        values: 'context_filtering', 'inverse_context_frequency', 'cooc'
    """

    def __init__(
        self,
        k=500,
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
        window_type="hour",
        score_type="standard",
        weighting="div",
        lmbd=0.5,
        beta=0.5,
        use_of_context=None,
        context_function="inverse_context_frequency"
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
        self.use_of_context = use_of_context
        self.setup_context_variables(window_type)
        self.get_context = {
            "weekday": self.get_weekday,
            "hour": self.get_hour,
            "period": self.get_period,
            "weekend": self.is_weekend,
            "even_hours": self.is_even_hour,
            "season": self.get_northern_season,
            "is_morning": self.is_morning,
            "even_days": self.is_even_day
        }
        self.total_training_sessions = 0
        self.score_type = score_type
        self.weighting = weighting
        self.score_items = self.set_score_function(score_type)
        self.followed_by = {}
        self.total_sampled_sessions = 0
        self.num_recommendations = 0
        self.lmbd = lmbd
        self.beta = beta
        self.current_context = None
        self.context_function = context_function


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
        super().fit(train, items)
        self.total_training_sessions = train[self.session_key].nunique()
        index_session = train.columns.get_loc(self.session_key)
        index_item = train.columns.get_loc(self.item_key)
        index_time = train.columns.get_loc(self.time_key)

        session = -1
        session_items = set()
        time = -1
        last_item = -1

        # hash setup
        self.setup_context_map()

        for row in train.itertuples(index=False):
            # cache items of sessions
            item = row[index_item]
            if row[index_session] != session:
                self.update_context_map(session, time, item)
                session = row[index_session]
                last_item = -1
            time = row[index_time]
            if self.score_type == 'sequence_filter':
                if last_item != -1:
                    if not last_item in self.followed_by:
                        self.followed_by[last_item] = set()
                    self.followed_by[last_item].add(item)
            last_item = item
        # add last row
        self.update_context_map(session, time, item)


    def possible_neighbor_sessions(self, session_items, input_item_id, session_id):
        """
        Find a set of session to later on find neighbors in.
        A self.sample_size of 0 uses all sessions in which any item of
        the current session appears.
        self.sampling can be performed with the options "recent" or "random".
        "recent" selects the self.sample_size most recent sessions;
        "random" just choses sessions randomly.

        Parameters
        --------
        sessions: set of session ids

        Returns
        --------
        out : set
        """

        # TODO: Changes location of this code to predict_next
        if self.similarity == "vec_div":
            self.hashmap = {}
            for index, item in enumerate(self.session_items):
                self.hashmap[item] = index + 1 / len(self.session_items)

        self.current_context = self.get_context[self.window_type](self.time)
        self.relevant_sessions = self.relevant_sessions | self.item_session_map.get(input_item_id, set())
        # context = self.sessions_for_context[
        #     self.get_context[self.window_type](self.time)
        # ]
        # sample = self.relevant_sessions & context
        sample = self.relevant_sessions
        if self.use_of_context == "mandatory":
            sessions_in_context = self.sessions_for_context[self.current_context]
            sample = self.relevant_sessions & sessions_in_context
        if self.sample_size <= 0:  # use all session as possible neighbors
            print("!!!!! running KNN without a sample size (check config)")

        if len(sample) > self.sample_size:
            if self.use_of_context == "prefilter":
                sessions_in_context = self.sessions_for_context[self.current_context]
                sample = self.relevant_sessions & sessions_in_context
            # if len(sample) > self.sample_size:
            if self.sampling == "recent":
                sample = self.most_recent_sessions(sample)
            elif self.sampling == "random":
                sample = random.sample(sample, self.sample_size)
            elif self.sampling == "epcsr":
                sample = self.sample_epcsr()
                # sample = set()
                # ratio = math.floor(self.sample_size / len(self.session_items))
                # rc = self.item_session_map[self.session_items[-1]]
                # if len(rc) > ratio:
                #     rc = random.sample(rc, ratio)
                #     rc = self.most_recent_sessions(rc)
                # sample = sample | set(rc)
                # if len(self.session_items) > 1:
                #     sample_rest = self.sample_size - ratio
                #     other_items = self.session_items[:-1]
                #     ratio_others = math.floor(sample_rest / len(other_items))
                #     for index, item in enumerate(other_items):
                #         try:
                #             rc = self.item_session_map[item]
                #             if len(rc) > ratio_others:
                #                 rc = random.sample(rc, ratio)
                #             sample = sample | set(rc)
                #         except Exception as e:
                #             print(e)
                #             print("Value of ratio: {}".format(ratio))
            else:
                sample = sample
                # sample = random.sample(sample, self.sample_size)

        self.total_sampled_sessions += len(sample)
        return sample

    def set_score_function(self, score_type):
        if score_type == 'sequence':
            return self.score_items_sequence
        elif score_type == 'sequence_filter':
            return self.score_items_sequence_filter
        elif score_type == 'context':
            return self.score_items_context
        else:
            print("Using standard sknn score function.")
            return super().score_items

    def setup_context_variables(self, context):
        self.window_type = context
        self.sessions_for_context = {}
        self.items_for_context = {}
        self.setup_context_map()

    def sample_epcsr(self):
        sample = set()
        original_sample_size = self.sample_size
        # sample sessions from last item by most_recent
        ratio = self.sample_size = math.floor(self.sample_size / len(self.session_items))
        rc = self.item_session_map[self.session_items[-1]]
        if len(rc) > ratio:
            # rc = random.sample(rc, ratio)
            rc = self.most_recent_sessions(rc)
        sample = sample | set(rc)

        self.sample_size = original_sample_size
        # sample randomly from other items
        other_items = self.session_items[:-1]
        if other_items:
            sample_size = self.sample_size - ratio
            ratio = math.floor(sample_size / len(other_items))
            for item in other_items:
                rc = self.item_session_map[item]
                if len(rc) > ratio:
                    rc = random.sample(rc, ratio)
                sample = sample | set(rc)
        return sample

    def setup_context_map(self):
        contexts_partitions = {
            "weekday": 7,
            "hour": 24,
            "weekend": 2,
            "period": 4,
            "even_hours": 2,
            "season": 4,
            "is_morning": 2,
            "even_days": 2
        }
        num_contexts = contexts_partitions[self.window_type]
        self.total_contexts = 0
        for i in range(num_contexts):
            self.sessions_for_context[i] = set()
            self.items_for_context[i] = {}

    def update_context_map(self, session, timestamp, item):
        context = self.get_context[self.window_type](timestamp)
        self.sessions_for_context[context].add(session)
        self.total_contexts += 1
        self.items_for_context[context][item] = self.items_for_context[context].get(item, 0) + 1


    # Context functions
    def get_weekday(self, timestamp):
        return dt.fromtimestamp(timestamp).weekday()

    def get_hour(self, timestamp):
        return dt.fromtimestamp(timestamp).hour

    def get_period(self, timestamp):
        hour = dt.fromtimestamp(timestamp).hour
        if hour < 6:
            return 0
        elif hour < 12:
            return 1
        elif hour < 18:
            return 2
        else:
            return 3

    def is_weekend(self, timestamp):
        weekday = dt.fromtimestamp(timestamp).weekday()
        return int(weekday >= 5)

    def is_even_hour(self, timestamp):
        hour = dt.fromtimestamp(timestamp).hour
        return hour % 2 == 0

    def is_morning(self, timestamp):
        hour = dt.fromtimestamp(timestamp).hour
        return int(6 <= hour < 18)

    def get_northern_season(self, timestamp):
        month = dt.fromtimestamp(timestamp).weekday()
        # 0: winter, 1: spring, 2: summer, 3: fall
        if month < 3 or month > 11:
            return 0
        elif month < 6:
            return 1
        elif month < 9:
            return 2
        else:
            return 3

    def is_even_day(self, timestamp):
        weekday = dt.fromtimestamp(timestamp).weekday()
        return weekday % 2 == 0

    # Scoring functions
    def score_items_sequence_filter(self, neighbors):
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
        input_item_id = self.session_items[-1]
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

    def score_items_sequence(self, neighbors):
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

    def score_items_context(self, neighbors):
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
        context = self.current_context
        neighbor_ids = set([neighbor[0] for neighbor in neighbors])
        # iterate over the sessions
        for session_id, item_score in neighbors:
            # get the items in this session
            items = self.session_item_map[session_id]
            sessions_in_context = self.sessions_for_context[context]
            neighbors_in_context = sessions_in_context & neighbor_ids
            weight = 1
            if self.context_function == "context_filter":
                weight = math.sqrt(len(neighbors_in_context)/len(sessions_in_context)) if session_id in self.sessions_for_context[context] else 0
            elif self.context_function == "inverse_context_frequency":
                weight = math.sqrt(self.total_contexts/len(sessions_in_context)) if session_id in self.sessions_for_context[context] else 1
            else:
                raise TypeError("'context_function' not defined.")
            for item in items:
                sessions_for_item = self.item_session_map[item]
                item_context_relevance = len(sessions_for_item & sessions_in_context)
                score = scores.get(item, 0)
                scores[item] = (score + weight * item_score)
        return scores

    # decay functions for scknn
    def linear(self, i):
        return 1 - (0.1*i) if i <= 100 else 0

    def same(self, i):
        return 1

    def div(self, i):
        return 1/i

    def log(self, i):
        return 1/(math.log10(i+1.7))

    def quadratic(self, i):
        return 1/(i*i)

    # different similarity measures
    def dsm(self, first, second):
        '''
        Calculates the Diffusion-based Similarity Method for two sessions
        described on (https://link.springer.com/chapter/10.1007/978-3-030-16145-3_30)
        For lmbd=0.5 and beta=0, it's the same as calculating the cosine similarity.
        For lmbd=1 and beta=1, it's the same as calculating mass_diffusion
        lmbd: regularize sessions' degrees
        beta: regularize item degrees

        Sim_{DSM}(x,j,\lambda, \beta) =
            \frac{1}{d^{\lambda}_{x} \times d^{1-\lambda}_{j}} \sum^{n}_{i=0}\frac{a_{xi} a_{ji}}{d^{\beta}_{i}}

        Parameters
        --------
        first: set of items of a session
        second: set of items of a session
        Returns
        --------
        out : float value
        '''
        numerator = 0
        for item in (first & second):
            numerator += 1/(len(self.item_session_map[item]) ** self.beta)
        denominator = (len(first) ** self.lmbd) * (len(second) ** (1-self.lmbd))

        return numerator/denominator

    def vec_div(self, first, second):
        inter = first & second
        sum = 0
        for item in inter:
            sum += self.hashmap[item]
        return sum / len(self.hashmap)

    # context score functions
    def cooc(self, session, context):
        """
        Compute a set of scores for all items given a set of neighbors.

        Parameters
        --------
        neighbors: set of session ids

        Returns
        --------
        out : list of tuple (item, score)
        """
        sum_sessions = self.sessions_for_context[session].sum()
        sum_contexts = self.total_contexts[context]
        return self.sessions_for_context[session][context] / sum_contexts
