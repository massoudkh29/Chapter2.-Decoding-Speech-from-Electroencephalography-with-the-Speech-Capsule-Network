import pickle
import numpy as np

def load_subject_eeg(subject_id, vowels):
    # data_folder = 'C:/Users/SB00763936/Desktop/EEG/imagined_speech_capsule/S{}/'.format(subject_id)
    data_folder = 'C:/Users/SB00763936/Desktop/EEG/imagined_speech_capsule/S{}/pre_ica/'.format(subject_id)

    words_file = 'imagined_words.pickle'
    # words_file = 'raw_array_ica.pickle'
    vowels_file = 'imagined_vowels.pickle'
    # vowels_file = 'raw_array_vowels_ica.pickle'



    with open(data_folder + words_file, 'rb') as f:
        file = pickle.load(f)
    # w_data = file['raw_arrays_ica']
    w_data = file['imagined_words']
    w_labels = file['imagined_labels']

    if vowels == False:
        return w_data, w_labels

    elif vowels:
        try:
            with open(data_folder + vowels_file, 'rb') as f:
                file = pickle.load(f)
        except:
            with open(data_folder + vowels_file, 'rb') as f:
                file = pickle.load(f)

        v_data = file['imagined_vowels']
        # v_data = file['imagined_eeg_3d']
        v_labels = file['imagined_labels']
        # v_data = file['raw_array'][:][0]
        # v_labels = file['labels']

    return w_data, v_data, w_labels, v_labels


def eeg_to_3d(data, epoch_size, n_events, n_chan):
    idx = []
    eeg_format = []
    speech_index = 24576
    [idx.append(i) for i in range(0, speech_index, epoch_size)]
    for j in data:
        eeg_idx = []
        [eeg_idx.append(j[idx[k]:idx[k] + epoch_size]) for k in range(len(idx))]
        eeg_format.append(eeg_idx)
    eeg_format = np.array(eeg_format)

    return eeg_format

# ------------------------------------------------------------
# idx, a, x = ([] for i in range(3))
# [idx.append(i) for i in range(0,data.shape[1],epoch_size)]
# #print(idx)
# for j in data:
#     [a.append([j[idx[k]:idx[k]+epoch_size]]) for k in range(len(idx))]
# return np.reshape(np.array(a),(n_events,n_chan,epoch_size))
# --------------------------------------------------------------------

def format_data(data_type, subject_id, epoch):
    """
    Returns data into format required for inputting to the CNNs.
    Parameters:
        data_type: str()
        subject_id: str()
        epoch: length of single trials, int
    """

    if data_type == 'words':
        data, labels = load_subject_eeg(subject_id, vowels=False)
        n_chan = len(data)
        data = eeg_to_3d(data, epoch, int(data.shape[1] / epoch), n_chan).astype(np.float32)
        labels = labels.astype(np.int64)
        labels[:] = [x - 6 for x in labels]  # zero-index the labels
    elif data_type == 'vowels':
        _, data, _, labels = load_subject_eeg(subject_id, vowels=True)
        n_chan = len(data)
        data = eeg_to_3d(data, epoch, int(data.shape[1] / epoch), n_chan).astype(np.float32)
        labels = labels.astype(np.int64)
        labels[:] = [x - 1 for x in labels]
    elif data_type == 'all_classes':
        w_data, v_data, w_labels, v_labels = load_subject_eeg(subject_id, vowels=True)
        n_chan = len(w_data)
        words = eeg_to_3d(w_data, epoch, int(w_data.shape[1] / epoch), n_chan).astype(np.float32)
        vowels = eeg_to_3d(v_data, epoch, int(v_data.shape[1] / epoch), n_chan).astype(np.float32)
        data = np.concatenate((words, vowels), axis=0)
        labels = np.concatenate((w_labels, v_labels)).astype(np.int64)
        labels[:] = [x - 1 for x in labels]

    return data, labels

def balanced_subsample(features, targets, random_state=12):
    """
    function for balancing datasets by randomly-sampling data
    according to length of smallest class set.
    """
    from sklearn.utils import resample
    unique, counts = np.unique(targets, return_counts=True)
    unique_classes = dict(zip(unique, counts))
    mnm = len(targets)
    for i in unique_classes:
        if unique_classes[i] < mnm:
            mnm = unique_classes[i]

    X_list, y_list = [], []
    for unique in np.unique(targets):
        idx = np.where(targets == unique)
        X = features[idx]
        y = targets[idx]

        X1, y1 = resample(X, y, n_samples=mnm, random_state=random_state)
        X_list.append(X1)
        y_list.append(y1)

    balanced_X = X_list[0]
    balanced_y = y_list[0]

    for i in range(1, len(X_list)):
        balanced_X = np.concatenate((balanced_X, X_list[i]))
        balanced_y = np.concatenate((balanced_y, y_list[i]))

    return balanced_X, balanced_y

def return_indices(event_id, labels):
    """
    Returns indices for each word and vowel in the
    EEG dataset. Enables extraction of individual classes.

    Parameters:
        event_id: dict containing class label and number
        labels: np.array containing labels corresponding to dataset

    Returns:
        list of indices
    """
    indices = []
    for _, k in enumerate(event_id):
        idx = []
        for d, j in enumerate(labels):
            if event_id[k] == j:
                idx.append(d)
        indices.append(idx)
    return indices

def Ciaran():
    pass
def Experiment():
    pass

from scipy.fftpack import dct
import python_speech_features as psf
from python_speech_features.base import get_filterbanks
from python_speech_features import base

def mfcc_f(trial, sample_rate, winlen, winstep, nfilt, nfft, lowfreq, highfreq):
    winfunc=lambda x:np.ones((x,))
    n_chans = trial.shape[1]
    ch_features = []
    for ch in trial:
        frames = psf.sigproc.framesig(ch, winlen*sample_rate, winstep*sample_rate, winfunc)
        fb = get_filterbanks(nfilt,nfft,sample_rate,lowfreq,highfreq)

        pspec = psf.sigproc.powspec(frames,nfft)
        energy = np.sum(pspec,1) # this stores the total energy in each frame
        energy = np.where(energy == 0,np.finfo(float).eps,energy) # if energy is zero, we get problems with log
        feat = np.dot(pspec,fb.T) # compute the filterbank energies
        feat = np.where(feat == 0,np.finfo(float).eps,feat) # if feat is zero, we get problems with log
        feat = np.log(feat)
        feat = dct(feat, type=2, axis=1, norm='ortho')[:,:12] #13 - numceps
        feat = base.lifter(feat,22)

        ch_features.append(np.reshape(feat,((feat.shape[0]*feat.shape[1]))))
    return np.array(ch_features).reshape((np.array(ch_features).shape[0]*np.array(ch_features).shape[1]))
