import pickle

def data_to_file(data, filepath):
    """
    Save a value of a variable data to filepath

    Usage:
    data_to_file(data, filepath)
    """
    with open(filepath, "wb") as fp:   #Pickling
        result = pickle.dump(data, fp)
    return result

def data_from_file(filepath):
    """
    Loads the data from filepath to a variable var

    Usage:
    var = data_from_file(filepath)
    """
    with open(filepath, "rb") as fp:   # Unpickling
        b = pickle.load(fp)
    return b
