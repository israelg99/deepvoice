import numpy as np

from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Bidirectional, LSTM, GRU, Dense, InputLayer
from keras.optimizers import Nadam
from keras.utils.visualize_util import plot

from deepvoice.data.cmudict import get_cmudict, test_dataset_cmudict
from deepvoice.util.util import sparse_labels

def G2P(layers, tables, RNN=GRU, build=True):
    # TODO: Teacher-forcing.
    # TODO: Beam search.

    nb_chars = len(tables[0].chars)
    nb_phons = len(tables[1].chars)
    word_length = tables[0].maxlen
    phon_length = tables[1].maxlen

    model = Sequential()

    model.add(InputLayer((word_length, nb_chars)))

    for _ in range(layers):
        model.add(Bidirectional(RNN(nb_phons, return_sequences=True, consume_less='gpu')))

    for _ in range(layers):
        model.add(RNN(nb_phons, return_sequences=True, consume_less='gpu'))

    model.add(TimeDistributed(Dense(nb_phons)))
    model.add(Activation('softmax'))

    if build:
        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=Nadam(),
                    metrics=['accuracy'])

    return model

def test_fit_G2P():
    #%% Test CMUDict.
    test_dataset_cmudict()

    #%% Get CMUDict data.
    (X_train, y_train), (_, _), (xtable, ytable) = get_cmudict(
        verbose=1,
        test_size=0.
    )

    #%% Examine features and labels (check if they are aligned).
    assert X_train.shape[0] == y_train.shape[0]
    rand_samples = np.random.randint(X_train.shape[0], size=5)
    print([''.join(i) for i in xtable.decode(X_train[rand_samples])])
    print([''.join(i) for i in ytable.decode(y_train[rand_samples])])

    # Sparse labels.
    y_train = sparse_labels(y_train)

    # Define model.
    model = G2P(3, (xtable, ytable), GRU, True)

    #%% Summerize and plot model.
    model.summary()
    plot(model)

    #%% Fit model.
    model.fit(X_train, y_train, batch_size=128, nb_epoch=20, verbose=1)

if __name__ == "__main__":
    test_fit_G2P()
