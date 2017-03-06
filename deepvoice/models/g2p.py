import numpy as np

from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Bidirectional, LSTM, GRU, Dense, InputLayer, RepeatVector
from keras.optimizers import Nadam
from keras.utils.visualize_util import plot

from recurrentshop.engine import RecurrentContainer
from recurrentshop.cells import GRUCell

from deepvoice.data.cmudict import get_cmudict, test_dataset_cmudict
from deepvoice.util.util import sparse_labels

def G2P(layers, tables, recurrentshop=False, RNN=None, feed_seq=True, build=True):
    """
    Grapheme-to-phoneme converter; RNN encoder-decoder model.
    # Arguments
        layers: Amount of layers for the encoder and decoder.
        tables: Charecter en/decoding tables, can be retrieved by `get_cmudict()`.
        recurrentshop: If to use the RecurrentShop extension, or to use vanilla Keras.
        RNN: Type of RNN cell, in case of vanilla Keras: GRU, LSTM.. in case of RecurrentShop: GRUCell, LSTMCell..
        feed_seq: If to feed the decoder the sequence of states from the encoder, or to feed the latest encoder state only.
        build: If to compile the model in Keras (the model will expect sprase labels).

    # Output
        A Keras model.
        Input:  `(word_length, nb_chars)` shaped one-hot vectors.
        Output: `(word_length, nb_phons)` shaped one-hot vectors.

    # Example
        ```
        (X_train, y_train), (X_test, y_test), (xtable, ytable) = get_cmudict()
        y_train = sparse_labels(y_train)

        model = G2P(3, (xtable, ytable), recurrentshop=True)
        model.fit(X_train, y_train, batch_size=64, nb_epoch=20)
        ```
    """
    # TODO: Teacher-forcing.
    # TODO: Beam search.
    # TODO: Decoder to output `phon_length` and not `word_length`.
    # TODO: Decide to either only use the RecurrentShop extension or vanilla Keras.

    if RNN is None:
        # Considering our RNN backend, pick an appropriate default RNN cell.
        # Vanilla Keras = GRU
        # RecurrentShop = GRUCell
        RNN = GRU
        if recurrentshop:
            RNN = GRUCell

    # Decode data into neat named variables.
    nb_chars = len(tables[0].chars)
    nb_phons = len(tables[1].chars)
    word_length = tables[0].maxlen
    phon_length = tables[1].maxlen

    model = Sequential()

    model.add(InputLayer((word_length, nb_chars)))

    if recurrentshop:
        # Use the RecurrentShop extension.

        # ENCODER:
        # Multi-layer bidirectional RNN.
        encoder = RecurrentContainer(return_sequences=feed_seq)
        encoder.add(RNN(nb_phons, input_dim=nb_chars))
        for _ in range(layers-1):
            encoder.add(RNN(nb_phons))

        model.add(Bidirectional(encoder))

        # If we don't feed the decoder with sequential states from the encoder.
        if not feed_seq:
            # Duplicate the last state from the encoder word-length times.
            model.add(RepeatVector(word_length))

        # DECODER:
        # Multi-layer unidirectional RNN.
        decoder = RecurrentContainer(return_sequences=True)
        decoder.add(RNN(nb_phons, input_dim=nb_phons*2))
        for _ in range(layers-1):
            decoder.add(RNN(nb_phons))

        model.add(decoder)
    else:
        # Use vanilla Keras.

        # ENCODER:
        # Multi-layer bidirectional RNN.
        for _ in range(layers-1):
            model.add(Bidirectional(RNN(nb_phons, return_sequences=True, consume_less='gpu')))
        model.add(Bidirectional(RNN(nb_phons, return_sequences=feed_seq, consume_less='gpu')))

        # If we don't feed the decoder with sequential states from the encoder.
        if not feed_seq:
            # Duplicate the last state from the encoder word-length times.
            model.add(RepeatVector(word_length))

        # DECODER:
        # Multi-layer unidirectional RNN.
        for _ in range(layers):
            model.add(RNN(nb_phons, return_sequences=True, consume_less='gpu'))

    # Add a dense layer at each timestep.
    # It will result in `(timesteps, number_of_phonemes)` shaped output values.
    model.add(TimeDistributed(Dense(nb_phons)))

    # Softmax to result in `(timesteps, number_of_phonemes)` shaped output probabilities.
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
    test_recurrentshop(False) # Test model with vanilla Keras.

if __name__ == "__main__":
    test_fit_G2P()
