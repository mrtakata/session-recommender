from _operator import itemgetter
# from .helpers import similarities
from math import sqrt
from datetime import datetime, timedelta
import random
import time
import numpy as np
import pandas as pd


class ContextKNN:
    """
    ContextKNN(k, sample_size=500, sampling='recent',
               similarity='jaccard', remind=False, pop_boost=0,
               session_key='SessionId', item_key='ItemId')

    This is the Session-Based KNN described in the paper
    (https://arxiv.org/pdf/1803.09587.pdf).

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
    ):

        self.k = k
        self.sample_size = sample_size
        self.sampling = sampling
        self.similarity = similarity

        # extensions
        self.remind = remind
        self.pop_boost = pop_boost
        self.extend = extend
        self.normalize = normalize

        # pandas columns
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key

        # updated while recommending
        self.session = -1
        self.session_items = set()
        self.time = 0
        self.relevant_sessions = set()

        # cache relations once at startup
        self.session_item_map = dict()  # items for session
        self.item_session_map = dict()  # sessions for item
        self.session_time = dict()

        self.sim_time = 0
        self.total_sampled_sessions = 0
        self.num_recommendations = 0
        self.last_n_days = None

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
        if self.last_n_days != None:

            max_time = datetime.fromtimestamp(train[self.time_key].max())
            date_threshold = max_time.date() - timedelta(self.last_n_days)
            stamp = datetime.combine(date_threshold, datetime.min.time()).timestamp()
            train = train[train[self.time_key] >= stamp]

        index_session = train.columns.get_loc(self.session_key)
        index_item = train.columns.get_loc(self.item_key)
        index_time = train.columns.get_loc(self.time_key)

        session = -1
        session_items = set()
        time = -1

        for row in train.itertuples(index=False):
            # cache items of sessions
            if row[index_session] != session:
                if len(session_items) > 0:
                    self.session_item_map.update({session: session_items})
                    self.session_time.update({session: time})
                session = row[index_session]
                session_items = set()
            time = row[index_time]
            session_items.add(row[index_item])

            # cache sessions involving an item
            self.update_item_session_map(row[index_item], row[index_session])

        # Add the last tuple
        self.session_item_map.update({session: session_items})
        self.session_time.update({session: time})

    def update_item_session_map(self, item_id, session_id):
        if item_id in self.item_session_map:
            self.item_session_map[item_id].add(session_id)
        else:
            self.item_session_map[item_id] = {session_id}

    def predict_next(
        self,
        session_id,
        input_item_id,
        predict_for_item_ids,
        skip=False,
        type="view",
        timestamp=0,
    ):
        """
        Gives predicton scores for a selected set of items on how likely they
        be the next item in the session.

        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs
            of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores.
            Every ID must be in the set of item IDs of the training set.

        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely
            to be the next item of this session. Indexed by the item IDs.

        """

        # gc.collect()
        # process = psutil.Process(os.getpid())
        # print('cknn.predict_next: ', process.memory_info().rss,
        #       ' memory used')
        self.num_recommendations += 1

        if self.session != session_id:  # new session

            if self.extend:
                item_set = set(self.session_items)
                self.session_item_map[self.session] = item_set
                for item in item_set:
                    self.update_item_session_map(item, self.session)

                ts = time.time()
                self.session_time.update({self.session: ts})

            self.session = session_id
            self.time = timestamp
            self.session_items = []
            self.relevant_sessions = set()

        if type == "view":
            self.session_items.append(input_item_id)

        if skip:
            return

        # items = self.session_items if self.last_n_clicks is None else self.session_items[-self.last_n_clicks:]
        item_set = set(self.session_items)
        neighbors = self.find_neighbors(item_set, input_item_id, session_id)
        scores = self.score_items(neighbors)

        # add some reminders
        if self.remind:
            reminderScore = 5
            takeLastN = 3

            cnt = 0
            for elem in self.session_items[-takeLastN:]:
                cnt = cnt + 1
                # reminderScore = reminderScore + (cnt/100)

                oldScore = scores.get(elem)
                newScore = 0
                if oldScore is None:
                    newScore = reminderScore
                else:
                    newScore = oldScore + reminderScore
                # print 'old score ', oldScore
                # update the score and add a small number for the position
                newScore = (newScore * reminderScore) + (cnt / 100)

                scores.update({elem: newScore})

        # push popular ones
        if self.pop_boost > 0:

            pop = self.item_pop(neighbors)
            # Iterate over the item neighbors
            # print itemScores
            for key in scores:
                item_pop = pop.get(key)
                # Gives some minimal MRR boost?
                scores.update({key: (scores[key] + (self.pop_boost * item_pop))})

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

    def item_pop(self, sessions):
        """
        Returns a dict(item,score) of the item popularity for the given list of
        sessions (only a set of ids)

        Parameters
        --------
        sessions: set

        Returns
        --------
        out : dict
        """
        result = {}
        max_pop = 0
        for session, weight in sessions:
            items = self.session_item_map[session]
            for item in items:
                count = result.get(item, 0)
                count += 1
                result[item] = count
                if count > max_pop:
                    max_pop = count

        for key in result:
            result[key] /= max_pop

        return result

    def most_recent_sessions(self, sessions):
        """
        Find the most recent sessions in the given set

        Parameters
        --------
        sessions: set of session ids

        Returns
        --------
        out : set
        """
        sample = set()
        tuples = list()
        for session in sessions:
            time = self.session_time[session]
            tuples.append((session, time))

        tuples = sorted(tuples, key=itemgetter(1), reverse=True)
        for element in tuples[: self.sample_size]:
            sample.add(element[0])

        return sample

    def possible_neighbor_sessions(self, session_items, input_item_id, session_id):
        """
        Find a set of session to later on find neighbors in.
        A self.sample_size of 0 uses all sessions in which any item of the current session appears.
        self.sampling can be performed with the options "recent" or "random".
        "recent" selects the self.sample_size most recent sessions while "random" just choses randomly.

        Parameters
        --------
        sessions: set of session ids

        Returns
        --------
        out : set
        """

        # Adds sets of relevant_sessions for last item
        self.relevant_sessions = (
            self.relevant_sessions | self.item_session_map.get(input_item_id, set())
        )
        sample = self.relevant_sessions

        if self.sample_size <= 0:  # use all session as possible neighbors
            print("!!!!! running KNN without a sample size (check config)")

        if len(sample) > self.sample_size:

            if self.sampling == "recent":
                sample = self.most_recent_sessions(sample)
            elif self.sampling == "random":
                sample = random.sample(sample, self.sample_size)
            else:
                sample = random.sample(sample, self.sample_size)

        self.total_sampled_sessions += len(sample)
        return sample

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

            similarity = getattr(self, self.similarity)(session_items_test, session_items)
            if similarity > 0:
                neighbors.append((session, similarity))

        return neighbors

    def find_neighbors(self, session_items, input_item_id, session_id):
        """
        Finds the k nearest neighbors for the given session_id and the current
        item input_item_id.

        Parameters
        --------
        session_items: set of item ids
        input_item_id: int
        session_id: int

        Returns
        --------
        out : list of tuple (session_id, similarity)
        """
        possible_neighbors = self.possible_neighbor_sessions(
            session_items, input_item_id, session_id
        )
        possible_neighbors = self.calc_similarity(session_items, possible_neighbors)

        possible_neighbors = sorted(
            possible_neighbors, reverse=True, key=lambda x: x[1]
        )
        possible_neighbors = possible_neighbors[: self.k]

        return possible_neighbors

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
        # iterate over the sessions
        for session_id, item_score in neighbors:
            # get the items in this session
            items = self.session_item_map[session_id]

            for item in items:
                score = scores.get(item, 0)
                scores[item] = score + item_score

        return scores

    def cosine(self, first, second):
        '''
        Calculates the cosine similarity for two sessions

        Parameters
        --------
        first: set of items of a session
        second: set of items of a session

        Returns
        --------
        out : float value
        '''
        li = len(first & second)
        la = len(first)
        lb = len(second)
        result = li / sqrt(la) * sqrt(lb)

        return result

    def jaccard(self, first, second):
        '''
        Calculates the jaccard index for two sessions

        Parameters
        --------
        first: set of items of a session
        second: set of items of a session

        Returns
        --------
        out : float value
        '''
        intersection = len(first & second)
        union = len(first | second)
        res = intersection / union

        return res

    def tanimoto(self, first, second):
        '''
        Calculates the cosine tanimoto similarity for two sessions

        Parameters
        --------
        first: set of items of a session
        second: set of items of a session

        Returns
        --------
        out : float value
        '''
        li = len(first & second)
        la = len(first)
        lb = len(second)
        result = li / (la + lb - li)

        return result

    def binary(self, first, second):
        '''
        Calculates the ? for 2 sessions

        Parameters
        --------
        first: set of items of a session
        second: set of items of a session

        Returns
        --------
        out : float value
        '''
        a = len(first & second)
        b = len(first)
        c = len(second)

        result = (2 * a) / ((2 * a) + b + c)

        return result

    def random(self, first, second):
        '''
        Calculates the ? for 2 sessions

        Parameters
        --------
        first: set of items of a session
        second: set of items of a session

        Returns
        --------
        out : float value
        '''
        return random.random()
