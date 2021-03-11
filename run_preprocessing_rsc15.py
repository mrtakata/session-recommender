import preprocessing.preprocess_rsc15 as pp
import time
import sys

'''
preprocessing method ["info","org","days_test","slice"]
    info: just load and show info
    org: from gru4rec (last day => test set)
    org_min_date: from gru4rec (last day => test set) but from a minimal date onwards
    days_test: adapted from gru4rec (last N days => test set)
    slice: new (create multiple train-test-combinations with a sliding window approach
    buys: load buys and safe file to prepared
'''
METHOD = "slice"

'''
data config (all methods)
'''
PATH = 'data/rsc15/raw/'
PATH_PROCESSED = 'data/rsc15/single/'
FILE = 'rsc15-clicks'

'''
org_min_date config
'''
MIN_DATE = '2014-04-01'

'''
filtering config (all methods)
'''
MIN_SESSION_LENGTH = 2
MAX_SESSION_LENGTH = 999999
MIN_ITEM_SUPPORT = 5

'''
days test default config
'''
DAYS_TEST = 7
METHOD = "days_test"


if __name__ == '__main__':
    '''
    Run the preprocessing configured above.
    '''
    if(len(sys.argv) > 1):
        METHOD = sys.argv[1]

    print("START preprocessing ", METHOD)
    sc, st = time.clock(), time.time()

    if METHOD == "info":
        pp.preprocess_info(PATH, FILE, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, MAX_SESSION_LENGTH)

    elif METHOD == "org":
        pp.preprocess_org(PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, MAX_SESSION_LENGTH)

    elif METHOD == "org_min_date":
        pp.preprocess_org_min_date(PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, MAX_SESSION_LENGTH, MIN_DATE)

    elif METHOD == "day_test":
        pp.preprocess_days_test(PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, MAX_SESSION_LENGTH, DAYS_TEST)

    elif METHOD == "slice":
        PATH_PROCESSED = 'data/rsc15/slices/'
        '''
        slicing default config
        '''
        NUM_SLICES = 5   # offset in days from the first date in the data set
        DAYS_OFFSET = 0  # number of days the training start date is shifted after creating one slice
        DAYS_SHIFT = 30
        #each slice consists of...
        DAYS_TRAIN = 28
        DAYS_TEST = 2
        pp.preprocess_slices(PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, MAX_SESSION_LENGTH, NUM_SLICES, DAYS_OFFSET, DAYS_SHIFT, DAYS_TRAIN, DAYS_TEST)

    elif METHOD == "buys":
        pp.preprocess_buys(PATH, FILE, PATH_PROCESSED)

    elif METHOD == "save":
        pp.preprocess_save(PATH, FILE, PATH_PROCESSED, MIN_ITEM_SUPPORT, MIN_SESSION_LENGTH, MAX_SESSION_LENGTH)

    else:
        print("Invalid method ", METHOD)

    print("END preproccessing ", (time.clock() - sc), "c ", (time.time() - st), "s")
