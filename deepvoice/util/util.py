import numpy as np

def sparse_labels(labels):
    return np.expand_dims(np.argmax(labels, -1), -1)