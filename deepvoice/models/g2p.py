#%% Import.
import numpy as np

from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Bidirectional, LSTM, GRU, Dense, InputLayer
from keras.optimizers import Nadam
from keras.utils.visualize_util import plot

from deepvoice.data.cmudict import get_cmudict, test_dataset_cmudict
from deepvoice.util.util import sparse_labels

#%% Test CMUDict.
test_dataset_cmudict()

#%% Get CMUDict data.
(X_train, y_train), (_, _), (xtable, ytable) = get_cmudict(
    verbose=1,
    test_size=0.
)

# Sparse labels.
y_train = sparse_labels(y_train)

#%% Examine features and labels (check if they are aligned).
rand_samples = np.random.randint(X_train.shape[0], size=5)
[''.join(i) for i in xtable.decode(X_train[rand_samples])]
[''.join(i) for i in ytable.decode(y_train[rand_samples])]

#%% Assign data-dependent model config.
nb_chars = len(xtable.chars)
nb_phons = len(ytable.chars)
word_length = xtable.maxlen
phon_length = ytable.maxlen

#%% Define model config.
RNN=GRU
LAYERS = 3

#%% Define training config.
batch_size = 100
epochs = 20

#%% Define model.
model = Sequential()

model.add(InputLayer((word_length, nb_chars)))

for _ in range(LAYERS):
    model.add(Bidirectional(RNN(nb_phons, return_sequences=True, consume_less='gpu')))

for _ in range(LAYERS):
    model.add(RNN(nb_phons, return_sequences=True, consume_less='gpu'))

model.add(TimeDistributed(Dense(nb_phons)))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Nadam(),
              metrics=['accuracy'])

#%% Summerize and plot model.
model.summary()
plot(model)

#%% Fit model.
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1)
