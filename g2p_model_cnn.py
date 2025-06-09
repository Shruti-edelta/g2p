import re
from collections import Counter
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense,Conv1D, BatchNormalization, Dropout, TimeDistributed, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import string
from sklearn.model_selection import train_test_split
import ast
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

df = pd.read_csv("dataset/cmu_dict_with_stress.csv")

df['word'] = df['word'].astype(str).apply(list)
words = df['word'].values
phonemes = df['phonemes'].apply(ast.literal_eval).values.tolist()


# word_train, word_val, phoneme_train, phoneme_val = train_test_split(words, phonemes, test_size=0.10, random_state=42)
# pd.DataFrame({'words': word_train, 'Phoneme_text': phoneme_train}).to_csv('dataset/train.csv', index=False)
# pd.DataFrame({'words': word_val, 'Phoneme_text': phoneme_val}).to_csv('dataset/val.csv', index=False)

print(words)
# print(words[1])
graphemes = sorted(set(ch for w in words for ch in w))
char2idx = {c: i + 1 for i, c in enumerate(graphemes)}
char2idx['<pad>'] = 0
char2idx['<sos>'] = len(char2idx)
char2idx['<eos>'] = len(char2idx)
print("grapheme: ",char2idx)
idx2char = {i: c for c, i in char2idx.items()}

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
        # print(type(seq))
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
print(X)
y_in, y_len = encode_sequences(phonemes, phn2idx, add_sos=True)
print(y_in.shape)
y_out, _ = encode_sequences(phonemes, phn2idx, add_eos=True, maxlen=y_len)
y_out = np.expand_dims(y_out, -1)
print(y_out.shape)

word_train, word_val, phoneme_train, phoneme_val = train_test_split(X, y_out, test_size=0.05, random_state=42)

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
# print(word_train)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint('model/1/best_model_cnn.keras', monitor='val_loss', save_best_only=True, verbose=1),
]
# Fit the model
history=model.fit(word_train, phoneme_train, batch_size=32, epochs=100,validation_data=[word_val,phoneme_val],callbacks=callbacks)
# history=model.fit(X, y_out, batch_size=32, epochs=100,validation_split=0.1,callbacks=callbacks)
model.save('model/1/model_cnn.keras')
model.save_weights('model/1/model_cnn_w.weights.h5')

history_df = pd.DataFrame(history.history)
history_df['epoch'] = range(1, len(history_df) + 1)
history_df.to_csv('model/1/model_cnn_metrics.csv', index=False)


# print(data_dict)``
# words=[]
# phonemes=[]

# seen_punct = set()
# for word, prons in cmu.items():

#     # Handle entries like ")close-paren"
#     if (word[0] in string.punctuation):
#         if word[0] in seen_punct: # skip second time punctuation
#             continue
#         punct = word[0]
#         # print(punct)
#         seen_punct.add(punct)  # avoid duplicates
#         words.append([punct])  # punctuation as a grapheme token
#         phonemes.append(prons[0])
#     else:    
#         words.append(list(word.lower()))
#         phonemes.append(prons[0])  # keep stress

# # for word, prons in cmu.items():
# #     # Skip words with non-alphabetic characters
# #     # if not word.isalpha():
# #     #     continue
#     # phn = remove_stress(prons[0])
#     phn = prons[0]
#     words.append(list(word.lower()))
#     phonemes.append(phn)