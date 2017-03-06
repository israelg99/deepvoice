import numpy as np
import itertools
import re

# Based on https://github.com/fchollet/keras/blob/master/examples/addition_rnn.py
class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    '''

    def __init__(self, chars='', maxlen=None, null_char=' ', left_pad=False):
        self.chars = sorted(set([null_char] + list(chars)))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen
        self.left_pad = left_pad
        self.null_char = null_char

    def fit(self, Cs, null_char=' '):
        """Determine chars and maxlen by fitting to data"""
        self.chars = sorted(set(itertools.chain([null_char], *Cs)))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = max(len(c) for c in Cs)
        self.null_char = null_char

    def encode(self, Cs, maxlen=None):
        """Pass in an array of arrays to convert to integers"""
        maxlen = maxlen if maxlen else self.maxlen
        n = len(Cs)
        X = np.zeros((n, maxlen, len(self.chars)), dtype=np.bool)
        for j, C in enumerate(Cs):
            if self.left_pad:
                C = [self.null_char] * (maxlen - len(C)) + list(C)
            else:
                C = list(C) + [self.null_char] * (maxlen - len(C))
                for i, c in enumerate(C):
                    X[j, i, self.char_indices[c]] = True
        return X

    def decode(self, Xs, calc_argmax=True):
        if calc_argmax:
            Xs = Xs.argmax(axis=-1)
        return np.array(list([self.indices_char[x] for x in X] for X in Xs))
