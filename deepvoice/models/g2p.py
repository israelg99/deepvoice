import numpy as np

from keras.models import Sequential, Model
from keras.layers import Activation, TimeDistributed, Bidirectional, LSTM, GRU, Dense, InputLayer, RepeatVector, Input
from keras.optimizers import Nadam
from keras.utils.vis_utils import plot_model

from deepvoice.data.cmudict import get_cmudict, test_dataset_cmudict
from deepvoice.util.util import sparse_labels

def G2P(layers, chars=29, phons=75, word_len=28, phon_len=28, tables=None,
        build=True, build_args=None, optimization=2):
    """
    Grapheme-to-phoneme converter; RNN GRU encoder-decoder model.
    # Arguments
        layers: Amount of layers for the encoder and decoder.
        chars: The amount of characters (English has 29).
        phons: The amount of phonemes (CMUDict uses 75).
        word_len: The length of the input word (CMUDict uses 28).
        phon_len: The length of the output phoneme (CMUDict uses 28).
        tables: Charecter en/decoding tables, can be retrieved by `get_cmudict()`.
        build: If to compile the model in Keras (the model will expect sprase labels).

    # Output
        A Keras model.
        Input:  `(word_length, chars)` shaped one-hot vectors.
        Output: `(word_length, phons)` shaped one-hot vectors.

    # Example
        ```
        (X_train, y_train), (X_test, y_test), (xtable, ytable) = get_cmudict()
        y_train = sparse_labels(y_train)

        model = G2P(layers=3, tables=(xtable, ytable))
        model.fit(X_train, y_train, batch_size=1024, epochs=20)
        ```
    """
    # V TODO: Engineer encoder model.
    # V TODO: Engineer decoder model.
    # V TODO: Get the state of an encoder's layer symbolically.
    # V TODO: Initialize a decoder layer state with the corresponding encoder layer state.
    # V TODO: Engineer the initial decoder input token (the output of encoder?).
    # V TODO: Feed the output of the decoder at t-1 as input at t.
    #   TODO: Engineer teacher forcing.

    # Decode data into neat named variables.
    if tables is not None:
        chars = len(tables[0].chars)
        phons = len(tables[1].chars)
        word_length = tables[0].maxlen
        phon_length = tables[1].maxlen

    # Define general RNN config.
    rnn_conf = {'units': phons,
                'return_sequences': True,
                'implementation': optimization}

    # Define our model's input.
    input_seq = Input((word_length, chars))

    ''' ENCODER '''
    # Multi-layer bidirectional GRU.
    # Keep an array of the encoder forward layer states to later (symbolically) initialize the decoder layers.
    # Define and add the encoders into the graph.
    encoder_conf = {**rnn_conf,
                    'return_state': True}
    encoder_bi_merge = 'sum'
    encoder_forward_states = [None]*layers

    encoded, encoder_forward_states[0], _ = Bidirectional(GRU(**encoder_conf), encoder_bi_merge)(input_seq)
    for layer in range(layers-2):
        encoded, encoder_forward_states[layer+1], _ = Bidirectional(GRU(**encoder_conf), encoder_bi_merge)(encoded)
    encoder_conf['return_sequences'] = False
    encoded, encoder_forward_states[-1], _ = Bidirectional(GRU(**encoder_conf), encoder_bi_merge)(encoded)
    assert not (None in encoder_forward_states), 'All encoder layer states haves to be assigned.'

    # Assign the encoder's final output as the decoder's initial input.
    # The encoder's output is of shape: `(phones)`.
    # The decoder expects input of shape: `(timestep, phones)`.
    # Use RV to add one timstep dimension to the encoder's output shape.
    input_decoder = RepeatVector(1)(encoded)

    # Teacher forcing.
    # ground_truth = Input((phon_len, phons))

    ''' DECODER '''
    # Multi-layer unidirectional GRU.
    # Initialize the decoder's layer states with the corresponding forward layer states from the encoder.
    # Define and add the decoders into the graph.
    decoder_conf = {**rnn_conf,
                    'unroll': True,
                    'output_length': phon_length}

    decoded = GRU(**decoder_conf)(input_decoder, initial_states=encoder_forward_states[0])
    for layer in range(layers-1):
        decoded = GRU(**decoder_conf)(decoded, initial_states=encoder_forward_states[layer+1])

    # Add a dense layer at each timestep to determine the output phonemes.
    # It will result in `(timesteps, number_of_phonemes)` shaped output values.
    output_densed = TimeDistributed(Dense(phons))(decoded)

    # Softmax to result in `(timesteps, number_of_phonemes)` shaped output probabilities.
    output_softmax = Activation('softmax')(output_densed)

    # Finalize the G2P model.
    g2p = Model(input_seq, output_softmax)

    if build:
        if build_args is None:
            build_args = {}
        if 'loss' not in build_args:
            build_args['loss'] = 'sparse_categorical_crossentropy'
        if 'optimizer' not in build_args:
            build_args['optimizer'] = Nadam()
        if 'metrics' not in build_args:
            build_args['metrics'] = ['accuracy']
        g2p.compile(**build_args)

    return g2p

def test_fit_G2P():
    # Test CMUDict.
    # test_dataset_cmudict()

    # Get CMUDict data.
    (X_train, y_train), (_, _), (xtable, ytable) = get_cmudict(
        verbose=1,
        test_size=0.
    )

    # Sparse labels.
    sparse_y_train = sparse_labels(y_train)

    # Define the G2P model.
    g2p = G2P(layers=3, tables=(xtable, ytable))

    # Summerize and plot model.
    g2p.summary()
    plot_model(g2p)

    # Crop the training data so it fits the batch size.
    batch = 1024
    X_batched = X_train[:X_train.shape[0]//batch*batch]
    y_batched = y_train[:y_train.shape[0]//batch*batch]

    y_sparse_batched = sparse_y_train[:sparse_y_train.shape[0]//batch*batch]

    # Fit the G2P model.
    g2p.fit(X_batched, y_sparse_batched, batch_size=batch, epochs=1, verbose=1)

if __name__ == "__main__":
    test_fit_G2P()
