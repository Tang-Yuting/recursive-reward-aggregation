import numpy as np

def find_closest_date_before(date, date_list):
    date_list_smaller = np.array(date_list)
    date_list_smaller = date_list_smaller[date_list_smaller <= date]
    return date_list_smaller[-1]

def find_closest_date_after(date, date_list):
    date_list_smaller = np.array(date_list)
    date_list_smaller = date_list_smaller[date_list_smaller < date]
    return date_list[len(date_list_smaller)]


def str_to_bool(s):
    if s == "True":
        return True
    elif s == "False":
        return False
    else:
        raise ValueError("Invalid string")