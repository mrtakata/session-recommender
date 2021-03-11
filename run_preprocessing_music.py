import preprocessing.preprocess_music as pp
import time
import sys

'''
preprocessing method ["info","org","days_test","slice"]
    info: just load and show info
    org: from gru4rec (last day => test set)
    days_test: adapted from gru4rec (last N days => test set)
    slice: new (create multiple train-test-combinations with a sliding window approach
'''
METHOD = "days_test"

'''
data config (all methods) // change dataset here
'''
PATH = 'data/30music/raw/'  # 30music,nowplaying,aotm,8tracks
PATH_PROCESSED = 'data/30music/single/'  # 30music,nowplaying,aotm,8tracks
# PATH_PROCESSED = 'data/nowplaying/slices/'  # 30music,nowplaying,aotm,8tracks
FILE = '30music-200ks'  # 30music-200ks,nowplaying,playlists-aotm,8tracks

'''
filtering config (all methods)
'''
MIN_SESSION_LENGTH = 5
MAX_SESSION_LENGTH = 999999
MIN_ITEM_SUPPORT = 2

'''
days test default config
'''
DAYS_FOR_TEST = 7
METHOD = "days_test"

if __name__ == '__main__':
    '''
    Run the preprocessing configured above.
    '''
    if(len(sys.argv) > 1):
        METHOD = sys.argv[1]

    print( "START preprocessing ", METHOD )
    sc, st = time.clock(), time.time()

    if METHOD == "info":
        pp.preprocess_info( PATH, FILE, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, MAX_SESSION_LENGTH )

    elif METHOD == "org":
        pp.preprocess_org( PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, MAX_SESSION_LENGTH )

    elif METHOD == "days_test":
        pp.preprocess_days_test( PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, MAX_SESSION_LENGTH, DAYS_FOR_TEST )

    elif METHOD == "slice":
        PATH_PROCESSED = 'data/nowplaying/slices/'
        '''
        slicing default config
        '''
        NUM_SLICES = 5  # offset in days from the first date in the data set
        DAYS_OFFSET = 0  # number of days the training start date is shifted after creating one slice
        DAYS_SHIFT = 63
        # each slice consists of...
        DAYS_TRAIN = 56
        DAYS_TEST = 7
        pp.preprocess_slices(PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, MAX_SESSION_LENGTH, NUM_SLICES, DAYS_OFFSET, DAYS_SHIFT, DAYS_TRAIN, DAYS_TEST)

    else:
        print("Invalid method ", METHOD)

    print("END preproccessing ", (time.clock() - sc), "c ", (time.time() - st), "s")
