from nltk.corpus import cmudict
import re
import numpy as np
import pandas as pd
import string

cmu = cmudict.dict()

# def remove_stress(phonemes):
#     return [re.sub(r'\d', '', p) for p in phonemes]

words = []
phonemes = []

seen_punct = set()
for word, prons in cmu.items():
    # Handle pure alphabetic words

    # Handle entries like ")close-paren"
    if (word[0] in string.punctuation):
        if word[0] in seen_punct:
            continue
        punct = word[0]
        print(punct)
        seen_punct.add(punct)  # avoid duplicates
        words.append([punct])  # punctuation as a grapheme token
        phonemes.append(prons[0])
    else:    
        words.append(list(word.lower()))
        phonemes.append(prons[0])  # keep stress
# print(len(seen_punct))

# for word, prons in cmu.items():
#     # Skip words with non-alphabetic characters
#     if not word.isalpha():
#         continue
#     phn = remove_stress(prons[0])
#     words.append(list(word.lower()))
#     phonemes.append(phn)

graphemes = sorted(set(ch for w in words for ch in w))
# print(graphemes)
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

def encode_sequences(data, token2idx, add_sos=False, add_eos=False, maxlen=None):
    encoded = []
    # print(data)
    for seq in data:
        s = []
        # print("seq: ",seq)
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

x, x_len = encode_sequences(words, char2idx)
print(x.shape)
y_in, y_len = encode_sequences(phonemes[:], phn2idx, add_sos=True)
print(y_in.shape)
y_out, _ = encode_sequences(phonemes[:], phn2idx, add_eos=True, maxlen=y_len)
print(y_out.shape)
# print("y_out: ",y_out)
y_out = np.expand_dims(y_out, -1)  # required for sparse categorical loss
print(y_out.shape)

import csv

with open("dataset/cmu_dict_pun_stress.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["word", "phonemes"])  # Write header row

    for w, p in zip(words, phonemes):
        word_str = ''.join(w)              # Convert list of chars to a string
        phoneme_str = ' '.join(p)          # Convert list of phonemes to space-separated string
        writer.writerow([word_str, phoneme_str])

# with open("dataset/cmu_dict.csv", "w", newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(["word", "phonemes"])  # Write header row

#     for word, prons in cmu.items():
#         # Use the first pronunciation for each word (you can modify this to include more if needed)
#         phoneme_str = ' '.join(prons[0])  # Join phonemes in the first pronunciation
#         writer.writerow([word, phoneme_str])


