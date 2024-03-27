import logging
import sys
import pandas as pd
import numpy as np
from Mass_EEG.utils import load_subject_eeg, eeg_to_3d, return_indices, Ciaran, Experiment
from Mass_EEG.CapsNet import capsNet, train, test
from sklearn.model_selection import train_test_split
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.datautil.splitters import split_into_two_sets # split by fraction or number
from braindecode.torch_ext.util import set_random_seeds
from keras.utils import to_categorical
log = logging.getLogger(__name__)

def network_model(X_Train, X_Test, y_train, y_test, n_chans, input_time_length, cuda, args, p, q, subject_id):
    max_epochs = 30
    max_increase_epochs = 10
    batch_size = 64
    init_block_size = 1000
    set_random_seeds(seed=20190629, cuda=cuda)
    n_classes = 2
    n_chans = n_chans
    input_time_length = input_time_length
    # model = runCapsNet(n_chans, n_classes, input_time_length=input_time_length,final_conv_length='auto')

    x_train = X_Train
    x_test = X_Test
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))


    # input_shape = x_train.shape[0:]
    # n_class = len(np.unique(np.argmax(y_train, 1)))
    model, eval_model, manipulate_model = capsNet(input_shape=x_train.shape[1:], n_class=2, routings=args.routings)
    model.summary()
    print(len(np.unique(np.argmax(y_train, 1))))

    train(model=model, model1=eval_model, data=((x_train, y_train), (x_test, y_test)), args=args)

    # elif model == 'Ciaran':
    #     model = Ciaran(n_chans, n_classes, input_time_length=input_time_length,final_conv_length='auto').create_network()

    # if cuda:
    #     model.cuda()

    # log.info("%s model: ".format(str(model)))
    #
    # optimizer = AdamW(model.parameters(), lr=0.00625, weight_decay=0)
    # iterator = BalancedBatchSizeIterator(batch_size=batch_size)
    # stop_criterion = Or([MaxEpochs(max_epochs),
    #                      NoDecrease('valid_misclass', max_increase_epochs)])
    # monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]
    # model_constraint = None
    # print('mas', train_set.X.shape[0])

    # model_test = Experiment(model, train_set, valid_set, test_set, iterator=iterator,
    #                         loss_function=F.nll_loss, optimizer=optimizer,
    #                         model_constraint=model_constraint, monitors=monitors,
    #                         stop_criterion=stop_criterion, remember_best_column='valid_misclass',
    #                         run_after_early_stop=True, cuda=cuda)

    #----test-------
    # model_test = Experiment()
    # model_test.run()
    model_test = test(model=eval_model, data=(x_test, y_test), args=args, p=p, q=q, subject_id=subject_id)
    return model_test


if __name__ == '__main__':
    import os
    import argparse
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s', level=logging.DEBUG, stream=sys.stdout)
    # Subject_ids = ['01']
    Subject_ids = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15']
    for sbj in Subject_ids:
        subject_id = sbj
        cuda = False
        w_data, _, w_labels, _ = load_subject_eeg(subject_id, vowels=True)
        n_chan, epoch = len(w_data), 4096
        w_data = eeg_to_3d(w_data, epoch, int(w_data.shape[1] / epoch), n_chan)
        # v_data = eeg_to_3d(v_data, epoch, int(v_data.shape[1] / epoch), n_chan)
        #-------------------downsample?????????????
        w_data = w_data[:, :, 768:1280]
        # v_data = v_data[:, :, 512:1024]
        #------------------------------------------
        #####Compute indices for each word and vowel in the dataset#####
        word_id = dict(Arriba=6, Abajo=7, Adelante=8, Atrás=9, Derecha=10, Izquierda=11)
        # vowel_id = dict(a=1, e=2, i=3, o=4, u=5)
        word_indices = return_indices(word_id, w_labels)
        # vowel_indices = return_indices(vowel_id, v_labels)

        n_chans = int(w_data.shape[1])
        input_time_length = w_data.shape[2]
        #####Train and test on each word/vowel combination#####
        for i, p in enumerate(word_id):
            print("Working on Subject: " + subject_id + ", Word: " + p)
            writer = f"C:/Users/SB00763936/Desktop/EEG/imagined_speech_cnn/S{subject_id}/{p}.xlsx"
            features_w = w_data[word_indices[i]]
            scores_list = []
            scores_all = []
            vowels_results = []
            # vowels_results = pd.DataFrame()
            for j, q in enumerate(word_id):
                if j > i:
                    if j!= i:
                        print("-----------Working on Word: " + q)
                        #####Combine training data#####
                        features_v = w_data[word_indices[j]]
                        features = np.concatenate((features_w, features_v)).astype(np.float32)

                        w_labels = np.zeros(features_w.shape[0])
                        v_labels = np.ones(features_v.shape[0])
                        labels = np.concatenate((w_labels, v_labels)).astype(np.int64)

                        # valid_set_fraction = 0.3

                        X_Train, X_Test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
                        y_train = (y_train).astype(np.int64)

                        print(X_Test.shape)
                        print(y_test.shape)

                        # train_set = SignalAndTarget(X_Train, y_train)
                        # test_set = SignalAndTarget(X_Test, y_test)
                        # train_set, valid_set = split_into_two_sets(train_set, first_set_fraction=1-valid_set_fraction)
                        # run_model = network_model(model, train_set, test_set, valid_set, n_chans, input_time_length, cuda)

                        run_model = network_model(X_Train, X_Test, y_train, y_test, n_chans, input_time_length, cuda, args, p, q, subject_id)
                        # log.info('Last 10 epochs')
                        # log.info("\n" + str(run_model.epochs_df.iloc[-10:]))

                        # vowels_results.append(run_model.epochs_df.iloc[-10:])
                        # vowels_results = vowels_results.append(pd.DataFrame())
                        # print(vowels_results)
                        # run_model.epochs_df.iloc[-10:].to_excel(writer,'sheet%s' %str(j+1))

            print(f"Saving classification results for Subject: {sbj}")
            # vowels_results.to_excel(writer) # saving results for one word vs all 5 vowels
