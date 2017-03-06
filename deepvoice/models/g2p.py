import numpy as np

from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Bidirectional, LSTM, GRU, Dense, InputLayer, RepeatVector
from keras.optimizers import Nadam
from keras.utils.visualize_util import plot

from recurrentshop.engine import RecurrentContainer
from recurrentshop.cells import GRUCell

from deepvoice.data.cmudict import get_cmudict, test_dataset_cmudict
from deepvoice.util.util import sparse_labels

def G2P(layers, tables, recurrentshop=False, RNN=None, feed_seq=False, build=True):
    # TODO: Teacher-forcing.
    # TODO: Beam search.

    RNN = GRU
    if recurrentshop:
        RNN = GRUCell

    nb_chars = len(tables[0].chars)
    nb_phons = len(tables[1].chars)
    word_length = tables[0].maxlen
    phon_length = tables[1].maxlen

    model = Sequential()

    model.add(InputLayer((word_length, nb_chars)))

    # TODO: Decide if the RecurrentShop extension is worth it.
    if recurrentshop:
        # Use the RecurrentShop extension.
        encoder = RecurrentContainer(return_sequences=feed_seq)
        encoder.add(RNN(nb_phons, input_dim=nb_chars))
        for _ in range(layers-1):
            encoder.add(RNN(nb_phons))

        decoder = RecurrentContainer(return_sequences=True)
        decoder.add(RNN(nb_phons, input_dim=nb_phons*2))
        for _ in range(layers-1):
            decoder.add(RNN(nb_phons))

        model.add(Bidirectional(encoder))
        if not feed_seq:
            model.add(RepeatVector(word_length))
        model.add(decoder)
    else:
        # Use vanilla Keras.
        for _ in range(layers-1):
            model.add(Bidirectional(RNN(nb_phons, return_sequences=True, consume_less='gpu')))
        model.add(Bidirectional(RNN(nb_phons, return_sequences=feed_seq, consume_less='gpu')))

        if not feed_seq:
            model.add(RepeatVector(word_length))

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
    # Test CMUDict.
    test_dataset_cmudict()

    # Get CMUDict data.
    (X_train, y_train), (_, _), (xtable, ytable) = get_cmudict(
        verbose=1,
        test_size=0.
    )

    # Sparse labels.
    y_train = sparse_labels(y_train)

    def test_recurrentshop(recurrentshop):
        # Define model.
        model = G2P(3, (xtable, ytable), recurrentshop=recurrentshop)

        # Summerize and plot model.
        model.summary()
        plot(model)

        # Fit model.
        model.fit(X_train, y_train, batch_size=1024, nb_epoch=1, verbose=1)

    test_recurrentshop(True) # Test model with RecurrentShop.
    test_recurrentshop(False) # Test model with vanilla keras.

if __name__ == "__main__":
    test_fit_G2P()
