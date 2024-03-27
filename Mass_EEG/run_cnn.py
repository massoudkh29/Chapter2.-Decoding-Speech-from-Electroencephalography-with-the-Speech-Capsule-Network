import numpy as np
import logging
from Mass_EEG.utils import balanced_subsample, format_data
from sklearn.model_selection import train_test_split

from keras.utils import normalize
log = logging.getLogger(__name__)




def network_model(subject_id, model_type, data_type, parameters):
    model_type = model_type
    best_params = dict()  # dictionary to store hyper-parameter values
    #####Parameter passed to funciton#####
    max_epochs = parameters['max_epochs']
    max_increase_epochs = parameters['max_increase_epochs']
    batch_size = parameters['batch_size']
    epoch = 4096

    #####Collect and format data#####
    if data_type == 'words':
        data, labels = format_data(data_type, subject_id, epoch)
        data = data[:, :, 768:1280]  # within-trial window selected for classification
    elif data_type == 'vowels':
        data, labels = format_data(data_type, subject_id, epoch)
        data = data[:, :, 512:1024]
    elif data_type == 'all_classes':
        data, labels = format_data(data_type, subject_id, epoch)
        data = data[:, :, 768:1280]

    x = lambda a: a * 1e6  # improves numerical stability
    data = x(data)

    data = normalize(data)
    data, labels = balanced_subsample(data, labels)  # downsampling the data to ensure equal classes

    # data1, _, labels1, _ = train_test_split(data, labels, test_size=0.2, random_state=42)  # redundant shuffle of data/labels
    data, X_test, labels, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)  # redundant shuffle of data/labels

    #####model inputs#####
    unique, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique)
    n_chans = int(data.shape[1])
    input_time_length = data.shape[2]

    return data, X_test, labels, y_test





if __name__ == '__main__':

    subject_ids = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
    model_types = ['shallow']
    # model_types = ['shallow', 'deep', 'eegnet']
    data_types = ['words', 'vowels']
    # data_types = ['words', 'vowels', 'all_classes']
    cuda = False
    parameters = dict(max_epochs=40, max_increase_epochs=30, batch_size=64)  # training parameters


    Totals = dict()
    total_words, total_vowels = [], []
    for subject_id in subject_ids:
        Totals[f"{subject_id}"] = dict()
        for model_type in model_types:
            Totals[f"{subject_id}"][f"{model_type}"] = dict()
            for data_type in data_types:
                data, X_test, labels, y_test = network_model(subject_id, model_type, data_type, parameters)