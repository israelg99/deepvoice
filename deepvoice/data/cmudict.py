import numpy as np
import itertools
import re

from keras.utils.data_utils import get_file
from sklearn.model_selection import train_test_split

from deepvoice.data.character_table import CharacterTable

def get_cmudict(origin='https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict', test_size=0.33, verbose=False, maxlen_x=None, maxlen_y=None, blacklist='().0123456789', max_phonemes=np.inf, max_chars=np.inf, seed=42):
    """
    Process CMU pronounciation dictionary as one-hot encoded grapheme and phoneme data
    # Arguments
        seed: Random seed for data split and shuffle
        test_size: Fraction of data to set aside for testing, 0 will return empty test data.
        verbose: Print messages about data processing
        maxlen_x: Crop and pad grapheme sequences to this length
        maxlen_y: Crop and pad phoneme sequences to this length
        max_phonemes: Restrict data to this <=max_phonemes
        max_chars: Restrict data to this <=max_charectors
        blacklist: Remove words with these charectors e.g. HOUSE(2) for the second varient of house
    # Output
        (X_train, y_train): One-hot encoded graphemes and phonemes
        (X_test, y_test): Test data
        (xtable, ytable): Charecter en/decoding tables
    # Example
        ```
        (X_train, y_train), (X_test, y_test), (xtable, ytable) = get_cmudict(
            verbose=1,
            test_size=0
        )

        # Preview 5 random samples of data.
        rand_samples = np.random.randint(X_train.shape[0], size=5)
        [''.join(i) for i in xtable.decode(X_train[rand_samples])]
        [''.join(i) for i in ytable.decode(y_train[rand_samples])]
        ```
    """

    cmudict_path = get_file("cmudict-py", origin=origin, untar=False)

    # load data
    X, y= [], []
    for line in open(cmudict_path,'r').readlines():
        word, pron = line.strip().split(' ',1)
        X.append(list(word))
        y.append(pron.split(' '))
    X = np.array(X)
    y = np.array(y)
    if verbose:  print('loaded {} entries from cmu_dict'.format(len(X)))

    # compile blacklist
    p=re.compile('[%s]'%(blacklist))

    # filter out duplicate entries like 'HOUSE(2)'
    X, y = zip(*[(x,y) for x,y in zip(X,y) if not bool(p.findall(''.join(x)))])
    if verbose:  print('removed duplicate entries leaving {}'.format(len(X)))

    # filter out complex entries
    X, y = zip(*[(x,y) for x,y in zip(X,y) if len(y)<=max_phonemes and len(x)<=max_chars])
    if verbose:  print('restricted to less than {} phonemes leaving {} entries'.format(max_phonemes, len(X)))

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # encode x and y and pad them
    xtable, X_train, X_test = _encode_chartable(X_train, X_test, maxlen_x)
    ytable, y_train, y_test = _encode_chartable(y_train, y_test, maxlen_y)

    if verbose:
        print('X_train shape:', X_train.shape)
        print('y_train shape:', y_train.shape)

        print('X_test shape:', X_test.shape)
        print('y_test shape:', y_test.shape)

    return (X_train, y_train), (X_test, y_test), (xtable, ytable)

def _encode_chartable(train, test, maxlen=None):
    load = train

    test_empty = len(test) == 0
    if not test_empty: load = test + train

    table = CharacterTable()
    table.fit(load)

    if maxlen: table.maxlen = maxlen

    return table, table.encode(train), np.empty(0) if test_empty else table.encode(test)

def test_dataset_cmudict():
    (X_train, y_train), (X_test, y_test), (xtable, ytable) = get_cmudict()

    # Assert lengths.
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    # Assert one-hot vectors.
    assert len(X_train.shape) == 3, 'Should be one-hot'
    assert len(y_train.shape) == 3, 'Should be one-hot'
    assert y_test.reshape((-1, y_test.shape[-1])).sum(-1).all(), 'Should be one-hot'
    assert X_test.reshape((-1, X_test.shape[-1])).sum(-1).all(), 'Should be one-hot'
    assert y_train.reshape((-1, y_train.shape[-1])).sum(-1).all(), 'Should be one-hot'
    assert X_train.reshape((-1, X_train.shape[-1])).sum(-1).all(), 'Should be one-hot'

    # Assert that test and train do not have overlap.
    dx_train = [' '.join(xx) for xx in xtable.decode(X_train)]
    dx_test = [' '.join(xx) for xx in xtable.decode(X_test)]
    x = dx_train + dx_test
    assert len(x) == len(set(x)), 'Should be no overlap between test and train'

    # Preview random samples of data.
    assert X_train.shape[0] == y_train.shape[0]
    rand_samples = np.random.randint(X_train.shape[0], size=5)
    print([''.join(i) for i in xtable.decode(X_train[rand_samples])])
    print([''.join(i) for i in ytable.decode(y_train[rand_samples])])

    # Assert test data is 0 when the split is 0.
    (X_train, y_train), (X_test, y_test), (xtable, ytable) = get_cmudict(test_size=0)
    assert X_test.size == 0 and y_test.size == 0, 'When test size is 0, test data must be empty.'

if __name__ == "__main__":
    test_dataset_cmudict()
