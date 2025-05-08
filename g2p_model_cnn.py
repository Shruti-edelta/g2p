from nltk.corpus import cmudict
import re
from collections import Counter
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense,Conv1D, BatchNormalization, Dropout, TimeDistributed, Activation
from tensorflow.keras.optimizers import Adam
import string

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

cmu = cmudict.dict()
# print(list(cmu.items())[:5])
print(len(cmu))

def remove_stress(phonemes):
    return [re.sub(r'\d', '', p) for p in phonemes]

words = []
phonemes = []

seen_punct = set()
for word, prons in cmu.items():

    # Handle entries like ")close-paren"
    if (word[0] in string.punctuation):
        if word[0] in seen_punct: # skip second time punctuation
            continue
        punct = word[0]
        # print(punct)
        seen_punct.add(punct)  # avoid duplicates
        words.append([punct])  # punctuation as a grapheme token
        phonemes.append(prons[0])
    else:    
        words.append(list(word.lower()))
        phonemes.append(prons[0])  # keep stress

# for word, prons in cmu.items():
#     # Skip words with non-alphabetic characters
#     # if not word.isalpha():
#     #     continue
#     # phn = remove_stress(prons[0])
#     phn = prons[0]
#     words.append(list(word.lower()))
#     phonemes.append(phn)

print(words[1])
graphemes = sorted(set(ch for w in words for ch in w))
char2idx = {c: i + 1 for i, c in enumerate(graphemes)}
char2idx['<pad>'] = 0
char2idx['<sos>'] = len(char2idx)
char2idx['<eos>'] = len(char2idx)
print("grapheme: ",char2idx)
idx2char = {i: c for c, i in char2idx.items()}

# Phoneme vocab
phoneme_set = sorted(set(p for ph in phonemes for p in ph))
# print(phoneme_set,len(phoneme_set))
phn2idx = {p: i + 1 for i, p in enumerate(phoneme_set)}
phn2idx['<pad>'] = 0
phn2idx['<sos>'] = len(phn2idx)
phn2idx['<eos>'] = len(phn2idx)
# print("phoneme: ",phn2idx)
idx2phn = {i: p for p, i in phn2idx.items()}
print(idx2phn)

def encode_sequences(data, token2idx, add_sos=False, add_eos=False, maxlen=None):
    encoded = []
    # print(data)
    for seq in data:
        s = []
        if add_sos: 
            s.append(token2idx['<sos>'])
        s += [token2idx[c] for c in seq]
        if add_eos: 
            s.append(token2idx['<eos>'])
        encoded.append(s)
        # print(s)zywicki
    # print(encoded)
    max_len = maxlen or max(len(s) for s in encoded)
    padded = [s + [token2idx['<pad>']] * (max_len - len(s)) for s in encoded]
    return np.array(padded), max_len

X, x_len = encode_sequences(words, char2idx)
print(X[19934])
y_in, y_len = encode_sequences(phonemes, phn2idx, add_sos=True)
print(y_in.shape)
y_out, _ = encode_sequences(phonemes, phn2idx, add_eos=True, maxlen=y_len)
y_out = np.expand_dims(y_out, -1)
print(y_out.shape)

vocab_size = len(char2idx)
phoneme_size = len(phn2idx)
embedding_dim = 128

def g2p_model(x_len,vocab_size,embedding_dim=128):
    # Input layer
    encoder_inputs = Input(shape=(x_len,), name="encoder_input")

    # Embedding layer
    x_emb = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(encoder_inputs)

    # CNN layers
    x = x_emb
    for _ in range(4):
        x = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

    # Output projection
    output = TimeDistributed(Dense(phoneme_size, activation='softmax'))(x)

    # Model
    model = Model(encoder_inputs, output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

model=g2p_model(x_len,vocab_size)
model.summary()

# Fit the model
history=model.fit(X, y_out, batch_size=32, epochs=25, validation_split=0.1)

model.save('model/model_cnn.keras')
model.save_weights('model/model_cnn_w.weights.h5')

history_df = pd.DataFrame(history.history)
history_df['epoch'] = range(1, len(history_df) + 1)
history_df.to_csv('model/model_cnn_metrics.csv', index=False)


