import tensorflow as tf
import numpy as np
import re
import numpy as np
import pandas as pd

df=pd.read_csv("dataset/cmu_dict_no_stress.csv")
# df=pd.read_csv("dataset/cmu_dict_with_punctu_stress.csv")

words = df["word"].tolist()
phonemes = df["phonemes"].tolist()

def phoneme_string_to_list(phoneme_strs):
    return [p.split() for p in phoneme_strs]
phonemes = phoneme_string_to_list(phonemes)

graphemes = sorted(set(ch for w in words for ch in str(w)))
char2idx = {c: i + 1 for i, c in enumerate(graphemes)}
char2idx['<pad>'] = 0
char2idx['<sos>'] = len(char2idx)
char2idx['<eos>'] = len(char2idx)
idx2char = {i: c for c, i in char2idx.items()}
print(idx2char)
# Phoneme vocab
phoneme_set = sorted(set(p for ph in phonemes for p in ph))
phn2idx = {p: i + 1 for i, p in enumerate(phoneme_set)}
phn2idx['<pad>'] = 0
phn2idx['<sos>'] = len(phn2idx)
phn2idx['<eos>'] = len(phn2idx)
idx2phn = {i: p for p, i in phn2idx.items()}

def encode_sequences(data, token2idx, add_sos=False, add_eos=False, maxlen=None):
    encoded = []
    for seq in data:
        s = []
        s += [token2idx[c] for c in seq]
        encoded.append(s)
    max_len = maxlen or max(len(s) for s in encoded)
    padded = [s + [token2idx['<pad>']] * (max_len - len(s)) for s in encoded]
    return np.array(padded), max_len

def preprocess_input(text, char2idx, maxlen=None):
    text = text.lower() # Convert the text to a list of characters
    words_1=text.split()
    encoded_word = []
    for w in words_1:
        encoded_text, _ = encode_sequences([w], char2idx, maxlen=maxlen)
        encoded_word.append(encoded_text)   
    return encoded_word

x_len=33
# Example usage:
new_text="Example"
new_text="Unfolding File"
new_text = "Shree Shruti"
# new_text = "excision"
# new_text="orange"
# # new_text="examplecom"
# new_text="oneall"
preprocessed_input = preprocess_input(new_text, char2idx, maxlen=x_len)  # Use x_len from your training
print("prerprocessed_input: ",preprocessed_input)

def predict_phonemes(model, preprocessed_input):
    pre_phon_l=[]
    for w in preprocessed_input:
        predictions = model.predict(w)
        # print(predictions)
        predicted_phonemes = np.argmax(predictions, axis=-1)  # Get the index of the highest probability phoneme
        # print("======",predicted_phonemes)
        pre_phon_l.append(predicted_phonemes[0])
    return pre_phon_l

# Example usage:
model = tf.keras.models.load_model('model/1/1model_cnn.keras') 
predicted_phonemes = predict_phonemes(model, preprocessed_input)
print("predicted tokenizer phoneme:",predicted_phonemes)

# print(idx2phn)
def decode_predictions(predictions, idx2phn):
    decoded_preds = []
    for w_p in predictions:
        # print(w_p)
        decoded_seq = [idx2phn.get(i, "<unk>") for i in w_p if i != char2idx['<pad>']]  # Ignore <pad> token
        # print(decoded_seq)
        decoded_preds.append(decoded_seq)
    return decoded_preds

# Example usage:
decoded_phonemes = decode_predictions(predicted_phonemes, idx2phn)
print(decoded_phonemes)



