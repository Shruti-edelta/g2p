import pandas as pd
import numpy as np
import re
import tensorflow
import unicodedata
import string
import contractions
from word2number import w2n
import tensorflow as tf
from num2words import num2words

class TextNormalizer:
    def __init__(self):
        self.abbreviations = {"Mr.": "Mister",
                        "Mrs.": "Misses",
                        "Dr.": "Doctor",
                        "No.": "Number",
                        "St.": "Saint",
                        "Co.": "Company",
                        "Jr.": "Junior",
                        "Sr.": "Senior",
                        "Maj.": "Major",
                        "Gen.": "General",
                        "Drs.": "Doctors",
                        "Rev.": "Reverend",
                        "Lt.": "Lieutenant",
                        "Hon.": "Honorable",
                        "Sgt.": "Sergeant",
                        "Capt.": "Captain",
                        "Esq.": "Esquire",
                        "Ltd.": "Limited",
                        "Col.": "Colonel",
                        "Ft.": "Fort",
                        "Ave.": "Avenue",
                        "etc.": "et cetera",
                        "i.e.": "that is",
                        "e.g.": "for example",
                        "$": "dollars",
                        "M": "million"}

    def expand_abbreviations(self, text):
        """Expands known abbreviations in the text."""
        for abbr, expansion in self.abbreviations.items():
            text = re.sub(r'\b'+abbr, expansion+" ", text)
        return text

    def number_to_words(self,text):
        def replace(match):
            num = int(match.group())
            return num2words(num)
        return re.sub(r'\b\d+\b', replace, text)

    # def normalize_year(self,year):
    #     """Convert 2023 → twenty twenty-three"""
    #     if 2000 <= year <= 2099:
    #         return f"twenty {num2words(year % 100)}"
    #     return num2words(year)

    # def normalize_text(self,text):
    #     for abbr, full in self.abbreviations.items():
    #         text = text.replace(abbr, full)
    #     # Step 2: Convert money (e.g., 5.6M)
    #     money_pattern = r'(\d+(\.\d+)?)(\s*)million|M|m'
    #     text = re.sub(r'\$(\d+(\.\d+)?)([Mm])', lambda m: f"{num2words(float(m.group(1)))} million dollars", text)
    #     # Step 3: Convert year (4-digit number)
    #     text = re.sub(r'\b(20\d{2})\b', lambda m: self.normalize_year(int(m.group())), text)
    #     return text

    def remove_punctuation(self, text):
        """Remove punctuation marks from the text.(!"#$%&'()*+,-./:;<=>?@[\]^_`{|})"""
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    def remove_extra_spaces(self, text):
        """Remove multiple spaces and trim leading/trailing spaces."""
        text = ' '.join(text.split())
        return text

    def expand_contractions(self, text):
        """Expand contractions like 'I'm' to 'I am'."""
        text = contractions.fix(text)
        return text

    def normalize_unicode(self, text):
        """Normalize unicode characters to a consistent form."""
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        return text

    def remove_urls_and_emails(self, text):
        """Remove URLs and email addresses from the text."""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        return text
    
    def normalize_time(self,text):
        # Match time in 12-hour format (e.g., "10 am", "3:00 pm")
        text = re.sub(r'(\d{1,2}):(\d{2})\s*(am|pm)', r'\1:\2 \3', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d{1,2})\s*(am|pm)', r'\1:00 \2', text, flags=re.IGNORECASE)  # Standardize to HH:MM AM/PM
        return text

    def normalize_text(self, text):
        # text = text.lower()  # Lowercase the text
        text = self.expand_contractions(text)  # Expand contractions
        text = self.expand_abbreviations(text)  # Expand abbreviations
        text = self.remove_extra_spaces(text)  # Remove extra spaces
        text = self.normalize_unicode(text)  # Normalize unicode
        text = self.normalize_time(text)  # Normalize time
        # text = self.remove_urls_and_emails(text)  # Remove URLs and emails
        text = self.number_to_words(text)  # Convert numbers to words
        text = self.remove_punctuation(text)  # Remove punctuation
        return text
    
    # def text_to_phonemes(self, text):
    #     """Convert text to phonemes."""
    #     text = self.normalize_text(text)  # Normalize the text first
    #     phonemes = []
    #     # Use phonemizer for ARPAbet or CMU Dictionary for word-level phonemes
    #     words = text.split()
    #     for word in words:
    #         if word_phonemes:
    #             phonemes.append(word_phonemes)
    #         else:
    #             # If phonemizer doesn't return a result, check CMU Pronouncing Dictionary
    #             cmu_phonemes = pronouncing.phones_for_word(word)
    #             if cmu_phonemes:
    #                 phonemes.append(cmu_phonemes[0])  # Take the first pronunciation variant
    #             else:
    #                 phonemes.append(word)
    #     return phonemes


class G2PConverter:
    def __init__(self, model_path, vocab_path="dataset/cmu_dict_no_stress.csv", max_len=33):
        self.max_len = max_len
        self.model = tf.keras.models.load_model(model_path)
        self._load_vocab(vocab_path)

    def _load_vocab(self, vocab_path):
        df = pd.read_csv(vocab_path)
        self.words = df["word"].tolist()
        self.phonemes = self._phoneme_string_to_list(df["phonemes"].tolist())

        # Build grapheme vocab
        graphemes = sorted(set(ch for w in self.words for ch in str(w)))
        self.char2idx = {c: i + 1 for i, c in enumerate(graphemes)}
        self.char2idx['<pad>'] = 0
        self.char2idx['<sos>'] = len(self.char2idx)
        self.char2idx['<eos>'] = len(self.char2idx)
        self.idx2char = {i: c for c, i in self.char2idx.items()}

        # Build phoneme vocab
        phoneme_set = sorted(set(p for ph in self.phonemes for p in ph))
        self.phn2idx = {p: i + 1 for i, p in enumerate(phoneme_set)}
        self.phn2idx['<pad>'] = 0
        self.phn2idx['<sos>'] = len(self.phn2idx)
        self.phn2idx['<eos>'] = len(self.phn2idx)
        self.idx2phn = {i: p for p, i in self.phn2idx.items()}

    def _phoneme_string_to_list(self, phoneme_strs):
        return [p.split() for p in phoneme_strs]

    def _encode_sequences(self, data, token2idx, maxlen=None):
        encoded = []
        for seq in data:
            s = [token2idx.get(c, token2idx['<pad>']) for c in seq]
            encoded.append(s)
        max_len = maxlen or max(len(s) for s in encoded)
        padded = [s + [token2idx['<pad>']] * (max_len - len(s)) for s in encoded]
        return np.array(padded)

    def preprocess_input(self, text):
        text = text.lower()
        words = text.split()
        encoded_words = [self._encode_sequences([w], self.char2idx, maxlen=self.max_len) for w in words]
        return encoded_words

    def predict(self, text):
        encoded_words = self.preprocess_input(text)
        predicted_phonemes = []
        for w in encoded_words:
            preds = self.model.predict(w, verbose=0)        # (1, 33, 42)
            phoneme_token = np.argmax(preds, axis=-1)[0]      # (33,) /(1, 33)
            phoneme_seq = [self.idx2phn.get(i, "<unk>") for i in phoneme_token if i != self.phn2idx['<pad>']]
            predicted_phonemes.append(phoneme_seq)
        return predicted_phonemes

if __name__ == "__main__":

    normalizer = TextNormalizer()
    raw_text = "Dr. Smith is going to the store. Visit https://example.com or email me at test@example.com! I'm excited, No. 1 fan!!!"
    normalized_text = normalizer.normalize_text(raw_text)
    print("Original Text:", raw_text)
    print("Normalized Text:", normalized_text)

    g2p=G2PConverter("model/1/model_cnn.keras")
    phonemes=g2p.predict(normalized_text)
    print(phonemes)
    for word, phon in zip(normalized_text.split(), phonemes):
        print(f"{word}: {' '.join(phon)}")


    # def ordinal_day(day):
    #     suffix = ['th', 'st', 'nd', 'rd', 'th', 'th', 'th', 'st', 'nd', 'rd', 'th', 'th', 'th', 'st', 'nd', 'rd', 'th', 'th', 'th']
    #     if 4 <= day <= 20 or 24 <= day <= 30:
    #         return str(day) + suffix[0]
    #     else:
    #         return str(day) + suffix[day % 10]

    # # Function to normalize dates (convert to "5th March 2025" format)
    # def normalize_date(text):
    #     # Match dates in formats like "March 5th, 2025", "5th March 2025", "2025-03-05", etc.
    #     # We assume a date format like "5th March 2025" or "March 5, 2025"
    #     text = re.sub(r'(\d{1,2})\s([A-Za-z]+)\s(\d{4})', " dfd", text)
    #     return text





