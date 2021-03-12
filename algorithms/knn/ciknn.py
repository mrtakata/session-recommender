import numpy as np
import pandas as pd
from datetime import datetime as dt


class ContextItemKNN:
    '''
    ItemKNN(k = 100, lmbd = 20, alpha = 0.5, session_key = 'SessionId', item_key = 'ItemId', time_key = 'Time')

    Item-to-item predictor that computes the the similarity to all items to the given item.

    Similarity of two items is given by:

    .. math::
        s_{i,j}=\sum_{s}I\{(s,i)\in D & (s,j)\in D\} / (supp_i+\\lambda)^{\\alpha}(supp_j+\\lambda)^{1-\\alpha}

    Parameters
    --------
    k : int
        Only give back non-zero scores to the N most similar items. Should be higher or equal than the cut-off of your evaluation. (Default value: 100)
    lmbd : float
        Regularization. Discounts the similarity of rare items (incidental co-occurrences). (Default value: 20)
    alpha : float
        Balance between normalizing with the supports of the two items. 0.5 gives cosine similarity, 1.0 gives confidence (as in association rules).
    session_key : string
        header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        header of the timestamp column in the input file (default: 'Time')
    context : string
        context used to make recommendations. (default: 'weekend')
    '''

    def __init__(self, k = 100, lmbd = 20, alpha = 0.5, session_key = 'SessionId', item_key = 'ItemId', time_key = 'Time', context='weekend'):
        self.k = k
        self.lmbd = lmbd
        self.alpha = alpha
        self.item_key = item_key
        self.session_key = session_key
        self.time_key = time_key
        self.sessions_for_item = {}
        self.sims = {}
        self.context = context
        self.setup_context_variables()

    def fit(self, data):
        '''
        Trains the predictor.

        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).

        '''
        data['context'] = data['Time'].apply(self.get_context)
        data_all = data
        for context in range(self.num_contexts):
            sims = {}
            data = data_all[data_all['context'] == context]
            data.set_index(np.arange(len(data)), inplace=True)
            itemids = data[self.item_key].unique()
            n_items = len(itemids)
            data = pd.merge(data, pd.DataFrame({self.item_key:itemids, 'ItemIdx':np.arange(len(itemids))}), on=self.item_key, how='inner')
            sessionids = data[self.session_key].unique()
            data = pd.merge(data, pd.DataFrame({self.session_key:sessionids, 'SessionIdx':np.arange(len(sessionids))}), on=self.session_key, how='inner')
            supp = data.groupby('SessionIdx').size()
            session_offsets = np.zeros(len(supp)+1, dtype=np.int32)
            session_offsets[1:] = supp.cumsum()
            index_by_sessions = data.sort_values(['SessionIdx', self.time_key]).index.values
            supp = data.groupby('ItemIdx').size()
            item_offsets = np.zeros(n_items+1, dtype=np.int32)
            item_offsets[1:] = supp.cumsum()
            index_by_items = data.sort_values(['ItemIdx', self.time_key]).index.values
            for i in range(n_items):
                iarray = np.zeros(n_items)
                start = item_offsets[i]
                end = item_offsets[i+1]
                for e in index_by_items[start:end]:
                    uidx = data.SessionIdx.values[e]
                    ustart = session_offsets[uidx]
                    uend = session_offsets[uidx+1]
                    user_events = index_by_sessions[ustart:uend]
                    iarray[data.ItemIdx.values[user_events]] += 1
                iarray[i] = 0
                norm = np.power((supp[i] + self.lmbd), self.alpha) * np.power((supp.values + self.lmbd), (1.0 - self.alpha))
                norm[norm == 0] = 1
                iarray = iarray / norm
                indices = np.argsort(iarray)[-1:-1-self.k:-1]
                sims[itemids[i]] = pd.Series(data=iarray[indices], index=itemids[indices])

            self.sims[context] = sims
        del(data_all)
        del(data)


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
        context = self.get_context(timestamp)
        preds = np.zeros(len(predict_for_item_ids))

        sim_list = self.sims[context].get(input_item_id, None)
        if sim_list is not None:
            mask = np.in1d(predict_for_item_ids, sim_list.index)
            preds[mask] = sim_list[predict_for_item_ids[mask]]
        return pd.Series(data=preds, index=predict_for_item_ids)

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
        la = len(first) + self.lmbd
        lb = len(second) + self.lmbd
        result = li / pow(la, self.alpha) * pow(lb, 1 - self.alpha)

        return result

    # context-related variables
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


    def setup_context_variables(self):
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
        context_functions = {
            "weekday": self.get_weekday,
            "hour": self.get_hour,
            "period": self.get_period,
            "weekend": self.is_weekend,
            "even_hours": self.is_even_hour,
            "season": self.get_northern_season,
            "is_morning": self.is_morning,
            "even_days": self.is_even_day
        }
        self.get_context = context_functions[self.context]
        self.num_contexts = contexts_partitions[self.context]
        self.sims = {}
        for i in range(self.num_contexts):
            self.sims[i] = {}