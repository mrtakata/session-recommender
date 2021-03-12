import random
import math

from .cknn import ContextKNN as sknn


class HeatConductionKNN(sknn):
    """
    ContextKNN(k, sample_size=1000, sampling='recent',
               similarity='dsm', remind=False, pop_boost=0,
               session_key='SessionId', item_key='ItemId')

    This is the implementation of the extension of the S-KNN model presented in
    (https://link.springer.com/chapter/10.1007/978-3-030-16145-3_30)

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
        (jaccard, cosine, binary, tanimoto). (default: dsm)
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
        k=500,
        sample_size=1000,
        sampling="recent",
        similarity="dsm",
        remind=False,
        pop_boost=0,
        extend=False,
        normalize=True,
        session_key="SessionId",
        item_key="ItemId",
        time_key="Time",
        lmbd=0.5,
        beta=1,
        window_type="weekend",
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
            window_type=window_type,

        )
        self.lmbd = lmbd
        self.beta = beta

    def possible_neighbor_sessions(self, session_items, input_item_id, session_id):
        """
        Find a set of session to later on find neighbors in.
        A self.sample_size of 0 uses all sessions in which any item of the
        current session appears.
        self.sampling is done with Equal Probability Candidate Selection at Random (EPCSR).

        Parameters
        --------
        session_items: set of session ids
        input_item_id: id of the item used to get all sessions that accessed the item
        session_id: id of the session that will receive the recommendation list
        Returns
        --------
        out : set
        """

        self.relevant_sessions = self.relevant_sessions | self.item_session_map[input_item_id]
        sample = self.relevant_sessions
        if self.sample_size == 0:  # use all session as possible neighbors
            print("!!!!! running KNN without a sample size (check config)")

        # sample if rc is too big
        if len(self.relevant_sessions) > self.sample_size:
            sample = set()
            for item in session_items:
                try:
                    ratio = math.floor(self.sample_size / len(session_items))
                    rc = self.item_session_map[item]
                    if len(rc) > ratio:
                        rc = self.most_recent_sessions(rc)
                        rc = random.sample(rc, ratio)
                    sample = sample | set(rc)
                except Exception as e:
                    print(e)
                    print("Value of ratio: {}".format(ratio))

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
        cnt = 0
        for session in sessions:
            cnt = cnt + 1
            # get items of the session, look up the cache first
            session_items_test = self.session_item_map[session]

            similarity = self.dsm(
                session_items_test, session_items
            )
            if similarity > 0:
                neighbors.append((session, similarity))

        return neighbors

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
                scores[item] = score + item_score * weight


        return scores