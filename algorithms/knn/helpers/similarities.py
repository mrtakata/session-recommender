from math import sqrt
import random


def cosine(first, second):
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


def jaccard(first, second):
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


def tanimoto(first, second):
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


def binary(first, second):
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


def random(first, second):
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


contexts_partitions = {
    "weekday": 7,
    "hour": 24,
    "weekend": 2,
    "period": 4,
    "even_hours": 2,
    "season": 4,
}