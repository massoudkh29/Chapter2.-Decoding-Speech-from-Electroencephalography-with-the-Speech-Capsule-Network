import scipy.io as spio
import numpy as np
import pickle
import mne
import os
from preprocessing.utils import load_pickle

# dir = ('..//imagined_speech/')
dir = ('C:/Users/SB00763936/Desktop/EEG/imagined_speech_capsule/')
# filename = "imagined_words.pickle"
filename = "imagined_vowels.pickle"
for folder in os.listdir(dir):
    if not folder.endswith(".txt") and not folder.endswith(".xlsx") and not folder.endswith("ica"):
        new_folder = folder + '/pre_ica/'
        data_folder = dir + new_folder

        with open(data_folder + filename, 'rb') as f:
            file = pickle.load(f)
        labels = file['imagined_labels']
        eeg_3d = file['imagined_eeg_3d']
        eeg_2d = file['imagined_eeg_2d']
        del file

        sfreq = 1024
        ch_names = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4']
        channel_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']
        Montage = 'standard_1020'
        line_freq = 50
        info = mne.create_info(ch_names, sfreq, channel_types, Montage)
        info['line_freq'] = line_freq
        info['highpass'] = 2.0
        info['lowpass'] = 40.0

        custom_raw = mne.io.RawArray(eeg_2d, info)
        print(custom_raw)

        # save_file = dir + new_folder + '/' + 'raw_array.pickle'
        save_file = dir + new_folder + '/' + 'raw_array_vowels.pickle'
        f = open(save_file, 'wb')
        save = {'raw_array': custom_raw,
                'labels': labels}
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()