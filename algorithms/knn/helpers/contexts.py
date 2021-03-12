from datetime import datetime as dt


def get_northern_season(timestamp):
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


def is_morning(timestamp):
    hour = dt.fromtimestamp(timestamp).hour
    return int(5 <= hour < 18)


def is_even_hour(timestamp):
    hour = dt.fromtimestamp(timestamp).hour
    return int(hour % 2 == 0)


def is_weekend(timestamp):
    weekday = dt.fromtimestamp(timestamp).weekday()
    return int(weekday >= 5)


def get_period(timestamp):
    hour = dt.fromtimestamp(timestamp).hour
    if hour < 6:
        return 0
    elif hour < 12:
        return 1
    elif hour < 18:
        return 2
    else:
        return 3


def get_hour(timestamp):
    return int(dt.fromtimestamp(timestamp).hour)


def get_weekday(timestamp):
    return int(dt.fromtimestamp(timestamp).weekday())
