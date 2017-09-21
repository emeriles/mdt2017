import pickle
import numpy as np

def data_to_file(data, filepath):
    with open(filepath, "wb") as fp:   #Pickling
        result = pickle.dump(data, fp)
    return result

def data_from_file(filepath):
    with open(filepath, "rb") as fp:   # Unpickling
        b = pickle.load(fp)
    return b

arr = np.array(range(0,10000))

res = data_to_file(arr, "./PICKLERICK.pickle")

print(res)

print("---")

arr2 = data_from_file("./PICKLERICK.pickle")

print(type(arr2))
