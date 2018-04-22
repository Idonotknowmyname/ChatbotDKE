import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from keras.models import Model, Sequential
from keras.layers import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Params on input sequences
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000

# Word embedding size
EMBEDDING_DIM = 100

FILE_PATH = ''

GLOVE_DIR = ''

def load_txt_file(path, ):
    pass

def load_embeddings_index(path=GLOVE_DIR):
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    return embeddings_index


def encoder_decoder_RNN(in_timesteps, embedding_dim, dict_size, out_timesteps=1, enc_depth=2, dec_depth=2,
                        rnn_type='LSTM', dropout=0.2, linear_reg_output=True, rnn_output_dim=16, attention=False,
                        optimizer='RMSprop', loss='mse', **kwargs):

    types = {'GRU': GRU, 'LSTM': LSTM, 'RNN': SimpleRNN}
    rnn = types[rnn_type]

    inputs = Input(shape=(in_timesteps, embedding_dim))


    # Encoder
    if enc_depth >= 2:
        encoded = rnn(rnn_output_dim, return_sequences=True)(inputs)

        if dropout > 0:
            encoded = Dropout(dropout)(encoded)

        for i in range(enc_depth - 2):
            encoded = rnn(rnn_output_dim, return_sequences=True)(encoded)

            if dropout > 0:
                encoded = Dropout(dropout)(encoded)


    enc_out, state_h, state_c = rnn(rnn_output_dim, return_state=True)(encoded if enc_depth >= 2 else inputs)


    encoder_states = [state_h, state_c]


    # Decoder

    decoder_inputs = Input(shape=(out_timesteps, dict_size))

    decoded = rnn(rnn_output_dim, return_sequences=True)(decoder_inputs, initial_state=encoder_states)

    if dropout > 0:
        decoded = Dropout(dropout)(decoded)

    for i in range(dec_depth - 1):
        decoded = rnn(rnn_output_dim, return_sequences=True)(decoded)

        if dropout > 0:
            decoded = Dropout(dropout)(decoded)

    activation = 'linear' if linear_reg_output else 'sigmoid'

    output = TimeDistributed(Dense(dict_size, activation=activation))(decoded)

    model = Model(inputs=[inputs, decoder_inputs], outputs=output)

    model.compile(optimizer=optimizer, loss=loss)

    return model


