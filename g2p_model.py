from nltk.corpus import cmudict
import re
from collections import Counter
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

cmu = cmudict.dict()

def remove_stress(phonemes):
    return [re.sub(r'\d', '', p) for p in phonemes]

words = []
phonemes = []

for word, prons in cmu.items():
    # Skip words with non-alphabetic characters
    if not word.isalpha():
        continue
    phn = remove_stress(prons[0])
    words.append(list(word.lower()))
    phonemes.append(phn)

graphemes = sorted(set(ch for w in words for ch in w))
char2idx = {c: i + 2 for i, c in enumerate(graphemes)}
char2idx['<pad>'] = 0
char2idx['<sos>'] = len(char2idx)
char2idx['<eos>'] = len(char2idx)
idx2char = {i: c for c, i in char2idx.items()}

# Phoneme vocab
phoneme_set = sorted(set(p for ph in phonemes for p in ph))
phn2idx = {p: i + 2 for i, p in enumerate(phoneme_set)}
phn2idx['<pad>'] = 0
phn2idx['<sos>'] = len(phn2idx)
phn2idx['<eos>'] = len(phn2idx)
idx2phn = {i: p for p, i in phn2idx.items()}

def encode_sequences(data, token2idx, add_sos=False, add_eos=False, maxlen=None):
    encoded = []
    for seq in data:
        s = []
        if add_sos: s.append(token2idx['<sos>'])
        s += [token2idx[c] for c in seq]
        if add_eos: s.append(token2idx['<eos>'])
        encoded.append(s)
    
    max_len = maxlen or max(len(s) for s in encoded)
    padded = [s + [token2idx['<pad>']] * (max_len - len(s)) for s in encoded]
    return np.array(padded), max_len

X, x_len = encode_sequences(words, char2idx)
Y_in, y_len = encode_sequences(phonemes, phn2idx, add_sos=True)
Y_out, _ = encode_sequences(phonemes, phn2idx, add_eos=True, maxlen=y_len)
Y_out = np.expand_dims(Y_out, -1)  # required for sparse categorical loss

vocab_size = len(char2idx)
phoneme_size = len(phn2idx)
embedding_dim = 128
latent_dim = 256

# Encoder
# encoder_inputs = Input(shape=(x_len,))
# enc_emb = Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
# _, state_h, state_c = LSTM(latent_dim, return_state=True)(enc_emb)

# # Decoder
# decoder_inputs = Input(shape=(y_len,))
# dec_emb = Embedding(phoneme_size, embedding_dim, mask_zero=True)(decoder_inputs)
# dec_lstm, _, _ = LSTM(latent_dim, return_sequences=True, return_state=True)(dec_emb, initial_state=[state_h, state_c])
# decoder_dense = Dense(phoneme_size, activation='softmax')
# decoder_outputs = decoder_dense(dec_lstm)

# # Model
# model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# model.summary()
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train
# history=model.fit([X, Y_in], Y_out, batch_size=64, epochs=10, validation_split=0.1)

# model.save('model.keras')
# model.save_weights('model_weight.weights.h5')

# history_df = pd.DataFrame(history.history)
# history_df['epoch'] = range(1, len(history_df) + 1)
# history_df.to_csv('training_metrics.csv', index=False)





# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

# # Encoder model (same as training encoder)
# encoder_inputs = Input(shape=(None,))
# enc_emb = Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
# encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(enc_emb)
# encoder_model = Model(encoder_inputs, [state_h, state_c])
# encoder_model.summary()

# # Decoder model (for inference)
# decoder_inputs = Input(shape=(None,))
# dec_emb = Embedding(phoneme_size, embedding_dim, mask_zero=True)(decoder_inputs)
# dec_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
# decoder_lstm_out, _, _ = dec_lstm(dec_emb, initial_state=[state_h, state_c])
# decoder_dense = Dense(phoneme_size, activation='softmax')
# decoder_outputs = decoder_dense(decoder_lstm_out)
# decoder_model = Model(decoder_inputs, decoder_outputs)
# decoder_model.summary()

# def decode_sequence(input_seq):
#     # Encode the input word (graphemes)
#     print(input_seq[0])
#     states_value = encoder_model.predict(input_seq[0])
    
#     # Generate the first input to the decoder, which is '<sos>'
#     target_seq = np.array([[phn2idx['<sos>']]])
    
#     # Initialize an empty list to store the predicted phonemes
#     decoded_phonemes = []
    
#     while True:
#         # Predict the next phoneme (step-by-step)
#         output_tokens = decoder_model.predict([target_seq] + states_value)
        
#         # Get the most likely next phoneme
#         sampled_token_index = np.argmax(output_tokens[0, -1, :])
#         sampled_phoneme = idx2phn[sampled_token_index]
        
#         # If we predict the '<eos>' token, stop the decoding
#         if sampled_phoneme == '<eos>' or len(decoded_phonemes) >= 50:
#             break
        
#         decoded_phonemes.append(sampled_phoneme)
        
#         # Update the target sequence with the predicted phoneme
#         target_seq = np.array([[sampled_token_index]])
        
#         # Update the states
#         states_value = [state_h, state_c]  # Use the last states from the LSTM
        
#     return decoded_phonemes

# def predict_phonemes(word):
#     # Convert the word to integer sequence
#     word_seq = [char2idx[c] for c in word.lower()]
#     word_seq = np.expand_dims(word_seq, axis=0)
    
#     # Get the predicted phonemes
#     phonemes = decode_sequence(word_seq)
    
#     return phonemes



# # Test the model with a new word
# test_word = "example"
# predicted_phonemes = predict_phonemes(test_word)
# print(f"Predicted phonemes for '{test_word}': {predicted_phonemes}")


def predict_phonemes(word,model):
    # Convert the word to integer sequence
    word_seq = [char2idx[c] for c in word.lower()]
    word_seq = np.expand_dims(word_seq, axis=0)
    print(word_seq.shape)
    phonemes=model.predict(word_seq)
    return phonemes

# model = tf.keras.models.load_model('model.keras')  # Load your trained mode
# test_word = "example"
# predicted_phonemes = predict_phonemes(test_word,model)
# print(predict_phonemes)
