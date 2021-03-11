from _operator import itemgetter
from math import sqrt
import random
import time
import numpy as np
import pandas as pd
import os
import psutil
import gc
from gensim.models import Word2Vec as w2v


data_path = '../../data/retailrocket/slices/'
file_prefix = 'events'
data_trained = '../../data/retailrocket/prepared2d/'


class Session2Vec:
    '''
    Session2Vec(k, sample_size=500, sampling='recent',
               similarity = 'jaccard', remind=False, pop_boost=0,
               session_key = 'SessionId', item_key= 'ItemId')

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
    use_embeddings : boolean
        Use w2v model if true, becomes regular S-KNN model otherwise
    method : string
        Determines the method used to combine item vectors to form a session
        representation. (default: "average")
        options: "average", "concatenate", "reciprocal", "weighted","quadratic", "linear"
    last_n : integer
        Determines the last n items of the ongoing session to be considered in
        the recommendation process. (default: 0 (use all))
    '''

    def __init__(self, k=500, sample_size=1000, sampling='recent',
                 similarity='cosine', remind=False, pop_boost=0,
                 extend=False, normalize=True, session_key='SessionId',
                 item_key='ItemId', time_key='Time', lmbd=1, beta=1,
                 path_trained=data_trained, file_prefix=file_prefix, slice=-1,
                 factors=100, epochs=30, window=5, sg=1, hs=0, threshold=0,
                 use_embeddings=True, method="average", last_n=0):

        self.remind = remind
        self.k = k
        self.sample_size = sample_size
        self.sampling = sampling
        self.similarity = similarity
        self.pop_boost = pop_boost
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.extend = extend
        self.normalize = normalize
        self.lmbd = lmbd
        self.beta = beta
        self.threshold = threshold

        # updated while recommending
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        # cache relations once at startup
        self.session_item_map = {}
        self.item_session_map = {}
        self.session_time = {}

        # file-related variables
        self.path_trained = path_trained
        self.file_prefix = file_prefix
        self.slice = slice

        # word2vec-related variables
        # self.wv = {}
        self.factors = factors
        self.epochs = epochs
        self.window = window
        self.sg = sg
        self.hs = hs

        # session2vec-related variables
        self.use_embeddings = use_embeddings
        self.method = method
        self.last_n = last_n
        self.all_session_items = {}
        self.all_session_vectors = {}

    def fit(self, train, items=None):
        '''
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

        '''

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
                    # cache the last time stamp of the session
                    self.session_time.update({session: time})
                session = row[index_session]
                session_items = set()
            time = row[index_time]
            session_items.add(row[index_item])

            # cache sessions involving an item
            map_is = self.item_session_map.get(row[index_item])
            if map_is is None:
                map_is = set()
                self.item_session_map.update({row[index_item]: map_is})
            map_is.add(row[index_session])

        # Add the last tuple
        self.session_item_map.update({session: session_items})
        self.session_time.update({session: time})

        if self.use_embeddings:
            self.fit_word_vectors(train)
            last_n = self.last_n
            self.last_n = 0
            self.fit_session_vectors()
            self.last_n = last_n

    def predict_next(self, session_id, input_item_id, predict_for_item_ids,
                     skip=False, type='view', timestamp=0):
        '''
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

        '''

        # gc.collect()
        # process = psutil.Process(os.getpid())
        # print('cknn.predict_next: ', process.memory_info().rss,
        #       ' memory used')

        if(self.session != session_id):  # new session

            if(self.extend):
                item_set = set(self.session_items)
                self.session_item_map[self.session] = item_set
                for item in item_set:
                    map_is = self.item_session_map.get(item)
                    if map_is is None:
                        map_is = set()
                        self.item_session_map.update({item: map_is})
                    map_is.add(self.session)

                ts = time.time()
                self.session_time.update({self.session: ts})

            self.session = session_id
            self.session_items = list()
            self.relevant_sessions = set()

        if type == 'view':
            self.session_items.append(input_item_id)

        if skip:
            return

        # finding neighbors and score
        if self.use_embeddings:
            neighbors = self.find_neighbors_vectors(self.session_items[-self.last_n:],
                                                    input_item_id, session_id)

        else:
            neighbors = self.find_neighbors(set(self.session_items[-self.last_n:]),
                                            input_item_id, session_id)

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
                # print 'old score', oldScore
                # update the score and add a small number for the position
                newScore = (newScore * reminderScore) + (cnt/100)

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

    def items_for_session(self, session):
        '''
        Returns all items in the session

        Parameters
        --------
        session: Id of a session

        Returns
        --------
        out : set
        '''
        return self.session_item_map.get(session)

    def sessions_for_item(self, item_id):
        '''
        Returns all session for an item

        Parameters
        --------
        item: Id of the item session

        Returns
        --------
        out : set
        '''
        return self.item_session_map.get(item_id)

    def most_recent_sessions(self, sessions, number):
        '''
        Find the most recent sessions in the given set

        Parameters
        --------
        sessions: set of session ids

        Returns
        --------
        out : set
        '''
        sample = set()

        tuples = list()
        for session in sessions:
            time = self.session_time.get(session)
            if time is None:
                print(' EMPTY TIMESTAMP!! ', session)
            tuples.append((session, time))

        tuples = sorted(tuples, key=itemgetter(1), reverse=True)
        # print 'sorted list ', sortedList
        cnt = 0
        for element in tuples:
            cnt = cnt + 1
            if cnt > number:
                break
            sample.add(element[0])
        # print 'returning sample of size ', len(sample)
        return sample

    def possible_neighbor_sessions(self, session_items, input_item_id,
                                   session_id):
        '''
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
        '''

        self.relevant_sessions = self.relevant_sessions | self.sessions_for_item(input_item_id)

        if self.sample_size == 0:  # use all session as possible neighbors

            print('!!!!! running KNN without a sample size (check config)')
            return self.relevant_sessions

        else:  # sample some sessions

            self.relevant_sessions = self.relevant_sessions | self.sessions_for_item(input_item_id)

            if len(self.relevant_sessions) > self.sample_size:

                if self.sampling == 'recent':
                    sample = self.most_recent_sessions(self.relevant_sessions, self.sample_size)
                elif self.sampling == 'random':
                    sample = random.sample(self.relevant_sessions, self.sample_size)
                else:
                    sample = self.relevant_sessions[:self.sample_size]

                return sample
            else:
                return self.relevant_sessions

    def calc_similarity(self, session_items, sessions):
        '''
        Calculates the configured similarity for the items in session_items
        and each session in sessions.

        Parameters
        --------
        session_items: set of item ids
        sessions: list of session ids

        Returns
        --------
        out : list of tuple (session_id,similarity)
        '''

        neighbors = []
        cnt = 0
        for session in sessions:
            cnt = cnt + 1
            # get items of the session, look up the cache first
            session_items_test = self.items_for_session(session)

            similarity = getattr(self, self.similarity)(session_items_test,
                                 session_items)
            if similarity > self.threshold:
                neighbors.append((session, similarity))
        return neighbors

    def find_neighbors(self, session_items, input_item_id, session_id):
        '''
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
        '''
        possible_neighbors = self.possible_neighbor_sessions(
                                  session_items, input_item_id, session_id)
        possible_neighbors = self.calc_similarity(
                                  session_items, possible_neighbors)

        possible_neighbors = sorted(possible_neighbors, reverse=True,
                                    key=lambda x: x[1])
        return possible_neighbors[:self.k]

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
            items = self.items_for_session(session[0])

            for item in items:
                old_score = scores.get(item)
                new_score = session[1]

                if old_score is None:
                    scores.update({item: new_score})
                else:
                    new_score = old_score + new_score
                    scores.update({item: new_score})

        return scores

    def item_pop(self, sessions):
        '''
        Returns a dict(item,score) of the item popularity for the given list of
        sessions (only a set of ids)

        Parameters
        --------
        sessions: set

        Returns
        --------
        out : dict
        '''
        result = dict()
        max_pop = 0
        for session, weight in sessions:
            items = self.items_for_session(session)
            for item in items:

                count = result.get(item)
                if count is None:
                    result.update({item: 1})
                else:
                    result.update({item: count + 1})

                if(result.get(item) > max_pop):
                    max_pop = result.get(item)

        for key in result:
            result.update({key: (result[key] / max_pop)})

        return result

    # similarity functions
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
        sc = time.clock()
        intersection = len(first & second)
        union = len(first | second)
        res = intersection / union
        return res

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
            numerator += 1/(len(self.sessions_for_item(item)) ** self.beta)
        denominator = (len(first) ** self.extendlmbd) * (len(second) ** (1-self.extendlmbd))

        return numerator/denominator

    # embeddings functions

    def fit_session_vocabulary(self, train):
        train[self.item_key] = train[self.item_key].astype(str)
        self.all_session_items = train.groupby(self.session_key)[self.item_key].apply(list)
        train[self.item_key] = train[self.item_key].astype(int)

    def fit_word_vectors(self, train, save_model=True):
        # Load word vectors if model already exists
        self.fit_session_vocabulary(train)
        if os.path.isfile(self.path_trained + self.file_prefix +
                          self.file_suffix()):
            print("Model already trained! Loading...", end="")
            self.load_w2v_model()
            print("done.")
        else:
            # Generate word vectors
            print("Generating word vectors...", end="")
            self.model = w2v(self.all_session_items.values, size=self.factors,
                             window=self.window, sg=self.sg, workers=4,
                             hs=self.hs, iter=self.epochs, min_count=1)
            print("done.")

            if save_model:
                self.save_w2v_model()
                print("Model saved!")
        self.wv = self.model.wv
        del(self.model)  # Discards wv model

    def fit_session_vectors(self):
        for session, items in self.all_session_items.items():
            items = items[-self.last_n:]
            self.all_session_vectors[session] = self.get_session_vector(items)

    def get_session_vector(self, session_items):
        vector = np.zeros(self.factors)
        if self.method == "average":
            for item in session_items:
                try:
                    vector = vector + self.wv[str(item)]
                except:
                    print("Item not found!")
            return vector/len(session_items)

        elif self.method == "reciprocal":
            for i, item in enumerate(session_items):
                try:
                    vector = vector + self.wv[str(item)]/(len(session_items)-i)
                except:
                    print("Item not found!")
            return vector
        elif self.method == "weighted":
            total_weights = (len(session_items) * (len(session_items)+1))/2
            for i, item in enumerate(session_items):
                try:
                    vector = vector + (self.wv[str(item)] * (len(session_items)-i))/total_weights
                except:
                    print("Item not found!")

            return vector
        elif self.method == "quadratic":
            for i, item in enumerate(session_items):
                try:
                    vector = vector + self.wv[str(item)]/((len(session_items)-i) ** 2)
                except:
                    print("Item not found!")

            return vector
        elif self.method == "linear":
            for i, item in enumerate(reversed(session_items)):
                try:
                    div = 1 - 0.1 * i
                    vector = vector + self.wv[str(item)] / div
                except:
                    print("Item not found!")

            return vector
        elif self.method == "concatenate":
            vector = []
            for i, item in enumerate(reversed(session_items)):
                try:
                    vector = np.concatenate([vector, self.wv[str(item)]])
                except:
                    print("Item not found!")

        else:
            raise RuntimeError("Method for session vectors not implemented yet!")
        return vector

    def cosine_embeddings(self, first, second):
        '''
        Calculates the cosine similarity for two session vectors

        Parameters
        --------
        first: list of floats
        second: list of floats

        Returns
        --------
        out : float value
        '''
        if self.method == 'concat':
            diff = abs(len(first) - len(second))
            if len(first) < len(second):
                first = np.concatenate([first, np.zeros(diff)])
            else:
                second = np.concatenate([second, np.zeros(diff)])

        dot_product = np.dot(first, second)
        norm_a = np.linalg.norm(first)
        norm_b = np.linalg.norm(second)
        return dot_product/(norm_a * norm_b)

    def calc_similarity_embeddings(self, session_items, sessions):
        '''
        Calculates the configured similarity for the items in session_items
        and each session in sessions.

        Parameters
        --------
        session_items: set of item ids
        sessions: list of session ids

        Returns
        --------
        out : list of tuple (session_id,similarity)
        '''

        neighbors = []
        cnt = 0
        curr_session_vector = self.get_session_vector(session_items)
        for session in sessions:
            cnt = cnt + 1
            # get items of the session, look up the cache first
            session_vector = self.all_session_vectors[session]
            similarity = self.cosine_embeddings(curr_session_vector, session_vector)
            if similarity > 0:
                neighbors.append((session, similarity))
        return neighbors

    def find_neighbors_vectors(self, session_items, input_item_id, session_id):
        '''
        Finds the k nearest neighbors for the given session_id and the current
        item input_item_id.

        Parameters
        --------
        session_items: list of item ids
        input_item_id: int
        session_id: int

        Returns
        --------
        out : list of tuple (session_id, similarity)
        '''
        possible_neighbors = self.possible_neighbor_sessions(
                                  set(session_items), input_item_id, session_id)
        len(possible_neighbors)
        possible_neighbors = self.calc_similarity_embeddings(session_items,
                                  possible_neighbors)
        len(possible_neighbors)
        possible_neighbors = sorted(possible_neighbors, reverse=True,
                                    key=lambda x: x[1])
        possible_neighbors = possible_neighbors[:self.k]
        return possible_neighbors

    # file-related variables
    def load_w2v_model(self, path=None, filename=None):
        if(path and filename):
            self.model = w2v.load(path + filename + self.file_suffix())
        #     else:
        #         w2v.load(path + file_prefix+".w2v")
        else:
            self.model = w2v.load(self.path_trained + self.file_prefix +
                                  self.file_suffix())

    def save_w2v_model(self, path=None, filename=None):
        if(path):
            if(filename):
                self.model.save(path + filename + self.file_suffix())
            else:
                self.model.save(path + file_prefix + self.file_suffix())
        else:
            self.model.save(self.path_trained + self.file_prefix +
                            self.file_suffix())

    def file_suffix(self):
        suff = "-f{}-sg{}-w{}-iter{}-hs{}" \
                .format(self.factors, self.sg, self.window,
                        self.epochs, self.hs)
        if self.slice > -1:
            return suff + "-slice-" + str(self.slice) + ".w2v"
        else:
            return suff + ".w2v"

    def algorithm_name(self):
        return "s2v-f{}-sg{}-w{}-iter{}-hs{}" \
                .format(self.factors, self.sg, self.window,
                        self.epochs, self.hs)


if __name__ == '__main__':

    # for testing in main
    import sys
    sys.path.append('../../')
    from evaluation import loader as loader
    data_path = '../../data/retailrocket/slices/'
    file_prefix = 'events'
    data_trained = '../../data/retailrocket/prepared2d/'

    train, test = loader.load_data(data_path, file_prefix,
                                   slice_num=0, rows_train=None,
                                   rows_test=None, density=1)
    items_to_predict = test['ItemId'].unique()

    model = Session2Vec(k=100, use_embeddings=True, method="average", last_n=0,
                        path_trained=data_trained, file_prefix=file_prefix,
                        slice=0)
    model.fit(train)
    print("Predicting...")
    model.predict_next(session_id=12034919041, input_item_id=67045,
                       predict_for_item_ids=items_to_predict)
    model.predict_next(session_id=12034919041, input_item_id=285930,
                       predict_for_item_ids=items_to_predict)
    model.predict_next(session_id=12034919041, input_item_id=67045,
                       predict_for_item_ids=items_to_predict)
    print(model.predict_next(session_id=12034919041, input_item_id=325215,
                             predict_for_item_ids=items_to_predict)
                             .sort_values(ascending=False))
