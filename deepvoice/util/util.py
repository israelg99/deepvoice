import itertools
import numpy as np
from six.moves import zip_longest
from collections import namedtuple
from keras import backend as K


def sparse_labels(labels):
    return np.expand_dims(np.argmax(labels, -1), -1)


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks."""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def flatten(list_of_lists):
    """Flatten one level of nesting."""
    return itertools.chain.from_iterable(list_of_lists)


def beamsearch(
    step, states0, batch_size,
    start_symbol,  stop_symbol,
    max_length,    beam_size=8,   min_length=0):
    """
    Beam search algorithm taken from BNAS and adjusted to Keras

    Parameters
    -------
    step : function
        Function which takes the following arguments:
        (iteration number, list of states, output history,
        output history mask)
        It is expected to return a tuple (new_states, outputs),
        where `new_states` is of the same form as `states0` and outputs
        is a matrix of shape (batch_size, n_symbols) containing probability
        distributions for the next symbol in each sequence of the batch.
    states0 : list
        Initial states, this will be passed to the `step` function on the first iteration.
    batch_size : int
        Batch size, this should be derivable from states0 but is included as
        an extra parameter for clarity.
    start_symbol : int
        Start symbol in the encoding used.
    stop_symbol : int
        Stop symbol in the encoding used.
    max_length : int
        Maximum length of sequence, but shorter sequences may also be returned.
    beam_size: int (default: 8)
        Size of the beam used.
    min_length: int (default: 0)
        Minimum length of the sequence.

    Returns
    ----------
    outputs : numpy.ndarray(int64)
        Array of shape ``(n_beams, length, batch_size)`` with the output
        sequences. `n_beams` is less than or equal to `beam_size`.
    outputs_mask : numpy.ndarray(K.floatx())
        Array of shape ``(n_beams, length, batch_size)``, containing the mask for `outputs`.
    scores : numpy.ndarray(float64)
        Array of shape ``(n_beams, batch_size)``.
        Log-probability of the sequences in `outputs`.
    """

    n_states = len(states0)

    # (beam, position, batch)
    sequence = np.full((1, 1, batch_size), start_symbol, dtype=np.int64)
    mask = np.ones((1, 1, batch_size), dtype=K.floatx())

    # (beam, batch, dims)
    states = [s[None, :, :] for s in states0]

    # (beam, batch)
    scores = np.zeros((1, batch_size))

    for i in range(max_length - 2):

        # Current size of beam
        n_beams = sequence.shape[0]

        all_states = []
        all_dists = []
        for j in range(n_beams):
            (part_states, part_dists) = step(i, [s[j, ...] for s in
                    states], sequence[j, ...], mask[j, ...])
            if i <= min_length:
                part_dists[:, stop_symbol] = 1e-30

            # Hard constraint: </S> must always be followed by </S>
            finished = (sequence[j, -1, :] == stop_symbol)[..., None]
            finished_dists = np.full_like(part_dists, 1e-30)
            finished_dists[:, stop_symbol] = 1.0
            part_dists = part_dists * (1 - finished) + finished_dists \
                * finished
            all_states.append(part_states)
            all_dists.append(part_dists)

        # list of (n_beams, batch_size, dims)
        all_states = [np.array(x) for x in zip(*all_states)]

        # (n_beams, batch_size, n_symbols)
        all_dists = np.log(np.array(all_dists)) + scores[:, :, None]
        n_symbols = all_dists.shape[-1]

        # (batch_size, n_beams*n_symbols)
        all_dists = np.concatenate(list(all_dists), axis=-1)

        # (beam_size, batch_size)
        best = np.argsort(all_dists.T, axis=0)[-beam_size:, :]
        best_beam = np.floor_divide(best, n_symbols)
        best_symbol = best - n_symbols * best_beam

        # TODO: optimize by allocating sequence/mask in the beginning,
        #       then shrink if necessary before returning

        sequence = np.concatenate([np.swapaxes(sequence[best_beam, :,
                                  np.arange(batch_size)[None, :]], 1,
                                  2), best_symbol[:, None, :]], axis=1)
        last_active = sequence[:, -2, :] != stop_symbol
        mask = np.concatenate([np.swapaxes(mask[best_beam, :,
                              np.arange(batch_size)[None, :]], 1, 2),
                              last_active[:, None, :]], axis=1)
        states = [s[best_beam, np.arange(batch_size)[None, :], :]
                  for s in all_states]
        scores = all_dists.T[best, np.arange(batch_size)[None, :]]

        if not mask[:, -1, :].any():
            return (sequence, mask, scores)

    n_beams = sequence.shape[0]
    sequence = np.concatenate([sequence, np.full((n_beams, 1,
                              batch_size), stop_symbol,
                              dtype=np.int64)], axis=1)
    mask = np.concatenate([mask, mask[:, -1:, :]], axis=1)

    return (sequence, mask, scores)
