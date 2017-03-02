# Deep Voice
*Based on the [Deep Voice paper](https://arxiv.org/pdf/1702.07825.pdf).*

Deep Voice is a text-to-speech system based entirely on deep neural networks.

Deep Voice comprises five models:

- Grapheme-to-phoneme converter.
- Phoneme Segmentation.
- Phoneme duration predictor.
- Frequency predictor.
- Audio synthesis.

## Grapheme-to-phoneme
##### Abstract
The grapheme-to-phoneme converter converts from written text (e.g English characters) to phonemes (encoded using a phonemic alphabet such as ARPABET).

#### Architecture
Based on [this architecture](https://arxiv.org/pdf/1506.00196.pdf) but with some changes.

The Grapheme-to-phoneme converter is an encoder-decoder:

- **Encoder**: multi-layer, bidirectional encoder, with a gated recurrent unit (GRU) nonlinearity.  
- **Decoder**: identical to the encoder but unidirectional.

It takes written text as input.

#### Setup
- **Initialization**: every decoder layer is initialized to the final hidden state of the corresponding encoder forward layer.  
- **Training**: the architecture is trained with teacher forcing.  
- **Decoding**: performed using beam search.

#### Hyperparameters
- **Encoder**: 3 bidirectional layers with 1024 units each.  
- **Decoder**: 3 unidirectional layers of the same size as the encoder.  
- **Beam Search**: width of 5 candidates.  
- **Dropout**: 0.95 rate after each recurrent layer.

## Phoneme Segmentation
##### Abstract
- The phoneme segmentation model locates phoneme boundaries in the voice dataset.  
- Given an audio file and a phoneme-by-phoneme transcription of the audio, the segmentation model identifies where in the audio each phoneme begins and ends.  
- The phoneme segmentation model is trained to output the alignment between a given utterance and a sequence of target phonemes. This task is similar to the problem of aligning speech to written output in speech recognition.

### Architecture
The segmentation model uses the convolutional recurrent neural network based on [Deep Speech 2](https://arxiv.org/pdf/1512.02595.pdf).

***The architecture graph***

1. Audio vector.
2. 20 MFCCs with 10ms stride.
2. Double 2D convolutions (frequency bins ***** time).
3. Triple bidirectional recurrent GRUs.
4. Softmax.
5. Output sequence of pairs.

### Hyperparameters
***Convolutions***
- **Stride**: (9, 5).
- **Dropout**: 0.95 rate after last convolution.

***Recurrent layers***
- **Dimensionality**: 512 GRU cells for each direction.
- **Dropout**: 0.95 rate after the last recurrent layer.

### Training
The segmentation model uses the [connectionist temporal classification (CTC) loss](ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf).

## Phoneme Duration + Frequency Predictor
#### Abstract
A single architecture is used to jointly predict phoneme duration and time-dependent fundamental frequency.

#### *Phoneme Duration* Abstract
The phoneme duration predictor predicts the temporal duration of every phoneme in a phoneme sequence (an utterance).

#### *Frequency Predictor* Abstract
The frequency predictor predicts whether a phoneme is voiced. If it is, the model predicts the fundamental frequency (F0) throughout the phoneme’s duration.

### Architecture
1. A sequence of phonemes with stresses, encoded in one-hot vector.
2. Double fully-connected layers.
3. Double unidirectional recurrent layers.
4. Fully-connected layer.

### Hyperparameters
**Double fully-connected layers**
- **Dimensionality**: 256.
- **Dropout**: 0.8 rate after last layer.

**Double unidirectional recurrent layers**
- **Dimensionality**: 128 GRUs.
- **Dropout**: 0.8 rate after last layer.

## Audio Synthesis
#### Abstract
* Combines the outputs of the grapheme-to-phoneme, phoneme duration, and  frequency predictor models.
* Synthesizes audio at a high sampling rate, corresponding to the desired text.
* Uses a WaveNet variant which requires less parameters and is faster to train.

### Architecture
The architecture is based on [WaveNet](https://arxiv.org/pdf/1609.03499.pdf) but with some changes.

***Will be updated soon.***