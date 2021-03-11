import numpy as np
import pandas as pd
from gensim.models import Word2Vec as w2v
import os

data_path = '../../data/retailrocket/slices/'
file_prefix = 'events'
data_trained = '../../data/retailrocket/prepared2d/'


class Item2Vec:
    '''
    Item2vec(factors=100, epochs=10, window=5, session_key='SessionId',
             sg=1, item_key='ItemId', time_key='Time')

    Parameters
    -----------
    item_key : string
        Header of the item ID column in the input file. (default: 'ItemId')
    session_key : string
        Header of the session ID column in the input file.
        (default: 'SessionId')
    time_key : string
        Header of the time column in the input file. (default: 'Time')
    factors : int
        Number of latent factors. (Default value: 100)
    epochs : int
        Number of training epochs. (default: 10)
    window : int
        Size of sliding window. (default: 5)
    sg : int
        Word2vec model training method used. (default: 1)
        1: Skip-gram
        0: CBOW
    hs : int
        Loss function of the WordVec model. (default: 1)
        1: Hierarchical Softmax
        0: Negative Sampling
    workers : int
        number of threads to train the model. (default : 4)
    seed : int
        If seed > 0, uses seed to train the model and sets # of workers to 1,
        in order to make it deterministic. (default : -1)
    '''

    def __init__(self, factors=150, epochs=30, window=5, workers=4, sg=1, hs=1,
                 session_key='SessionId', item_key='ItemId', time_key='Time',
                 slice=-1, path_trained=data_trained, file_prefix=file_prefix,
                 seed=-1):

        self.factors = factors
        self.epochs = epochs
        self.window = window
        self.sg = sg
        self.hs = hs
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.workers = workers
        self.seed = seed

        if seed > 0:
            self.workers = 1

        # file-related variables
        self.path_trained = path_trained
        self.file_prefix = file_prefix
        self.slice = slice
        self.not_found_words = 0

    def fit(self, train, save=True, load=True):
        '''
        Trains the predictor.

        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions.
            It has one column for session IDs, one for item IDs and one for the
            timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must
            correspond to the ones you set during the initialization of the
            network (session_key, item_key, time_key properties).

        save : bool
            Flag to save the model in file after training. (default: True)
        load : bool
            Flag to load trained model if exists. (default: True)
        '''

        if load and os.path.isfile(self.path_trained + self.file_prefix +
                                   self.file_suffix()):
            print("Model already trained! Loading...", end="")
            self.load_w2v_model()
            print("done.")
        else:
            print("Creating session vocabulary...", end="")
            train[self.item_key] = train[self.item_key].astype(str)
            sequences = train.groupby(self.session_key)[self.item_key] \
                             .apply(list)
            print("done.")
            print("Training model...", end="")
            if self.seed > 0:
                self.model = w2v(sequences, size=self.factors, sg=self.sg,
                                 window=self.window, workers=1, hs=self.hs,
                                 iter=self.epochs, min_count=1, seed=self.seed)
            else:
                self.model = w2v(sequences, size=self.factors, sg=self.sg,
                                 window=self.window, workers=self.workers,
                                 hs=self.hs, iter=self.epochs, min_count=1)

            print("done.")
            if(save):
                self.save_w2v_model()
                print("Model saved!")

            train[self.item_key] = train[self.item_key].astype(int)

    def predict_next(self, session_id, input_item_id, predict_for_item_ids,
                     timestamp=0, cut_off=20):
        '''
        Gives predicton scores for a selected set of items on how likely they
        be the next item in the session.

        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the
            training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores.
            Every ID must be in the set of item IDs of the training set.

        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next
            item of this session. Indexed by the item IDs.
        '''
        # predict_for_item_ids = predict_for_item_ids.astype(int)
        num_items = len(predict_for_item_ids)
        preds = np.zeros(num_items)
        input_item_id = str(input_item_id)
        try:
            sim_list_tuple = self.model.wv.most_similar(input_item_id,
                                                        topn=cut_off)
            # relative_cosine_similarity
            sim_index = []
            sim_val = []
            for score in sim_list_tuple:
                sim_index.append(int(score[0]))
                sim_val.append(score[1])
            sim_list = pd.Series(data=sim_val, index=sim_index)
            mask = np.in1d(predict_for_item_ids, sim_list.index)
            preds[mask] = sim_list[predict_for_item_ids[mask]]
        except:
            self.not_found_words = self.not_found_words + 1

        return pd.Series(data=preds, index=predict_for_item_ids)

    def load_w2v_model(self, path=None, filename=None):
        if(path and filename):
            self.model = w2v.load(path + filename + self.file_suffix())
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
        suff = "-f{}-sg{}-w{}-iter{}-hs{}".format(self.factors, self.sg,
                                                  self.window, self.epochs,
                                                  self.hs)
        if self.slice > -1:
            return suff + "-slice-" + str(self.slice) + ".w2v"
        else:
            return suff + ".w2v"


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

    factors = 100
    window = 5
    sg = 1
    epochs = 10
    model = Item2Vec(factors=factors, window=window, sg=sg, workers=4, hs=1,
                     epochs=epochs)
    model.fit(train)
    print(model.predict_next(1, 67045, items_to_predict)
               .sort_values(ascending=False))
