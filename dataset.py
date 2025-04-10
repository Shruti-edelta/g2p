import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
from text_preprocess import TextNormalizer

def pad_or_truncate(mel_spectrogram, max_time_frames=512):
    if mel_spectrogram.shape[1] < max_time_frames:
        pad_width = max_time_frames - mel_spectrogram.shape[1]
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    elif mel_spectrogram.shape[1] > max_time_frames:
        mel_spectrogram = mel_spectrogram[:, :max_time_frames]
    return mel_spectrogram

def audio_to_mel_spectrogram(audio_file,max_time_frames=512):     
    y, sr = librosa.load(audio_file,sr=22050)  # load at 22050 Hz consistant SR
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)     #range in 0.00000 somthing 7.4612461e-03
    mel_spectrogram_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB (decibels)unit scale 
    mel_spectrogram_db = (mel_spectrogram_db - np.mean(mel_spectrogram_db)) / np.std(mel_spectrogram_db)       # Normalize the Mel spectrogram to a fixed range (e.g., -1 to 1)
    mel_spectrogram_db = pad_or_truncate(mel_spectrogram_db, max_time_frames)  # Pad or truncate
    # print(mel_spectrogram_db)
    print(mel_spectrogram_db.T)       #(512, 128)
    return mel_spectrogram_db.T

def save_npyfile():
    for file in tqdm(file_names):
        audio_path = os.path.join(folder_path, "wavs", file)
        npy_path = audio_path.replace('.wav', '.npy')
        try:
            if not os.path.exists(npy_path):
                mel_spec = audio_to_mel_spectrogram(audio_path)
                np.save(npy_path, mel_spec)
                # print(f"Created {npy_path}")
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
                
folder_path='dataset/LJSpeech/'
file_names = os.listdir(folder_path+"wavs/")

df=pd.read_csv(folder_path+"metadata.csv",sep='|')

# rows_all_null = df[df.isnull().any(axis=1)]
# print(rows_all_null)

normalizer = TextNormalizer()
df.dropna(inplace=True,ignore_index=True)
df.drop(columns=['Original_text'],inplace=True)

print(type(df['Normalize_text'].iloc[0]))

df['Phoneme_text'] = df['Normalize_text'].apply(normalizer.text_to_phonemes)

save_npyfile()
# audio_to_mel_spectrogram("dataset/LJSpeech/wavs/LJ048-0016.wav")

for i in range(len(df)):
    file_id = df.iloc[i, 0]
    path = os.path.join(folder_path, "wavs", f"{file_id}.npy")
    df.at[i, 'Read_npy'] = path


print(df)

rows_all_null = df[df.isnull().any(axis=1)]
print(rows_all_null)
df.to_csv('tts_data_LJ.csv',index=False)







# Input shape: (16, 512)
# Target shape: (16, 512, 128)
# first training data
# LJ048-0016,He likewise indicated he was disenchanted with Russia.
# ['HH IY1', 'L AY1 K W AY2 Z', 'IH1 N D AH0 K EY2 T AH0 D', 'HH IY1', 'W AA1 Z', 'D IH0 S IH0 N CH AE1 N T IH0 D', 'W IH1 DH', 'R AH1 SH AH0'] 
# [19 52 61 45 18 65 99 10 60  2 34  1 18 92  7  1  4 19 52 16 20 10 49 15
#  14 15  2 89 17  2  7 15  4 16  9 85 50 27 51 12  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0]
# [[-1.0548493  -0.91554594 -0.67141706 ... -0.12701225 -0.10908517
#    0.00408208]
#  [-1.0438508  -0.744871   -0.41393754 ...  0.53789884  0.5130554
#    0.5846023 ]
#  [-1.3826112  -1.0201247  -0.58847576 ...  0.6733828   0.59189814
#    0.63817024]
#  ...
#  [ 0.          0.          0.         ...  0.          0.
#    0.        ]
#  [ 0.          0.          0.         ...  0.          0.
#    0.        ]
#  [ 0.          0.          0.         ...  0.          0.
#    0.        ]]


# import librosa
# import numpy as np
# import pandas as pd
# import os
# import pronouncing
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import Tokenizer

# def text_to_phonemes(text):
#     words = text.split()
#     phonemes = []
#     for word in words:
#         # Get phonemes for each word from the CMU Pronouncing Dictionary
#         word_phonemes = pronouncing.phones_for_word(word)
#         if word_phonemes:
#             phonemes.append(word_phonemes[0])  # Take the first pronunciation variant
#         else:
#             phonemes.append(word)  # If no phoneme found, keep the word as it is
#     return phonemes

# def audio_to_mel_spectrogram(audio_file):     
#     y, sr = librosa.load(audio_file,sr=None)  # load at 22050 Hz consistant SR
#     mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80,fmax=8000)     #range in 0.00000 somthing 7.4612461e-03
#     mel_spectrogram_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB (decibels)unit scale 
#     mel_spectrogram_db = (mel_spectrogram_db - np.mean(mel_spectrogram_db)) / np.std(mel_spectrogram_db)       # Normalize the Mel spectrogram to a fixed range (e.g., -1 to 1)
#     return mel_spectrogram_db

# def save_npyfile():
#     for k,v in dic_w.items():
#         # print("speaker_id: ",k,len(v),v[0])
#         for file in v:
#             if not os.path.exists(file.replace('.wav','.npy')):
#                 mel_spec = audio_to_mel_spectrogram(file)
#                 # print(mel_spec.shape)
#                 np.save(file.replace('.wav','.npy'), mel_spec)  # save as .npy    
#                 print(f"create {file.replace('.wav','.npy')} npy file... ")

# def concat_speaker():
#     global final_df
#     for k,v in dic.items():
#         text_df = pd.read_csv(v, sep='\t')
#         text_df.dropna(inplace=True) 
#         text_df['Phoneme_text'] = text_df['Normalize_text'].apply(lambda x: text_to_phonemes(x))
#         text_df.drop(columns=["Original_text","Normalize_text"],inplace=True) 
#         temp_df=text_df
#         final_df=pd.concat([final_df, temp_df],ignore_index = True)

# # folder_path = 'dataset/libri_dataset/'
# dic_w={}
# dic={}
# # dic_t={}

# for speaker_id in os.listdir(folder_path):
#     if speaker_id!='.DS_Store':
#         # print(speaker_id)
#         for file in os.listdir(os.path.join(folder_path, speaker_id)):
#             # if file.endswith(".txt"):
#             #     file_path=os.path.join(os.path.join(folder_path, speaker_id),file)
#             #     dic_t.setdefault(speaker_id, []).append(file_path)
#             if file.endswith(".wav"):
#                 file_path=os.path.join(os.path.join(folder_path, speaker_id),file)
#                 dic_w.setdefault(speaker_id, []).append(file_path)
#             elif file.endswith(".trans.tsv"):
#                 file_path=os.path.join(os.path.join(folder_path, speaker_id),file)
#                 dic[speaker_id]=file_path

# # for speaker_id in os.listdir(folder_path):
# #     if speaker_id!='.DS_Store':
# #         # print(f'{speaker_id} = {len(dic_t[speaker_id])}')
# #         print(f'{speaker_id} = {len(dic_w[speaker_id])}')

# save_npyfile()
# final_df = pd.DataFrame(columns=['File_Name', 'Phoneme_text','Read_npy'])
# concat_speaker()

# for i in range(len(final_df)):
#     file_id=final_df.iloc[i, 0]
#     path=folder_path+file_id.split('_')[0]+'/'+file_id+'.npy'
#     final_df.at[i, 'Read_npy'] = path

# final_df.to_csv('train.csv',index=False)


# # texts=final_df['Phoneme_text'].values
# # token=tokenizer_text_word().tokenizer_text()
# # sequences = token.texts_to_sequences(texts)
# # padded_text = pad_sequences(sequences, maxlen=600, padding='post') 
# # padded_text=padded_text.tolist()
# # final_df['padded_texts'] = final_df['padded_texts'].astype(object)
# # for i in range(len(final_df)):
# #     final_df.at[i, 'padded_texts'] = padded_text[i]

# # print(final_df)
# # s_row=final_df.iloc[1075]
# # print(s_row)

# # path="dataset/libri_dataset/32/32_4137_000002_000000.npy"
# # arr=np.load(path)
# # print(type(arr))
# # print(arr)
# # print(arr.shape)  # (1, 32, 32)



