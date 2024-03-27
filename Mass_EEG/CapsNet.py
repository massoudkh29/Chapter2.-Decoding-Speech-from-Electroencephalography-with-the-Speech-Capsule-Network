from __future__ import print_function
from keras.layers import *
import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
# from utils import combine_images
from PIL import Image
from Mass_EEG.CapsNetF import CapsuleLayer, PrimaryCap, Length, Mask
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv

import os
from keras import callbacks


from keras.preprocessing.image import ImageDataGenerator
# K.set_image_data_format('channels_last')

def capsNet(input_shape, n_class, routings):

    # x = layers.Input(shape=(6,512,1))
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(40, (1, 10), strides=(1, 1), padding="same", activation='relu', name='conv1')(x)
    conv2 = layers.Conv2D(40, (6, 1), strides=(1, 1), padding="valid", activation='relu', name='conv2')(conv1)
    batch1 = layers.BatchNormalization()(conv2)
    primarycaps = PrimaryCap(batch1, dim_capsule=8, n_channels=8, kernel_size=(1, 9), strides=(1, 1), padding='valid')
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, name='digitcaps')(primarycaps)


    # conv1 = layers.Conv2D(40, (1, 10), strides=(1, 1), padding="same", activation='relu', name='conv1')(x)
    # conv2 = layers.Conv2D(40, (6, 1), strides=(1, 1), padding="valid", activation='relu', name='conv2')(conv1)
    # batch1 = layers.BatchNormalization()(conv2)
    # conv3 = layers.Conv2D(40, (1, 50), strides=(1, 1), padding="same", activation='relu', name='conv3')(batch1)
    # primarycaps = PrimaryCap(conv3, dim_capsule=8, n_channels=8, kernel_size=(1, 9), strides=(1, 1), padding='valid')
    # digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, name='digitcaps')(primarycaps)

    # conv1 = layers.Conv2D(100, (1, 80), strides=(1, 1), padding="same", activation='relu', name='conv1')(x)
    # conv2 = layers.Conv2D(100, (6, 1), strides=(1, 1), padding="valid", activation='relu', name='conv2')(conv1)
    # batch1 = layers.BatchNormalization()(conv2)
    # primarycaps = PrimaryCap(batch1, dim_capsule=8, n_channels=40, kernel_size=(1, 90), strides=(1, 1), padding='valid')
    # digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, name='digitcaps')(primarycaps)

    # model = models.Sequential()
    # model.summary()
    # a=0

    out_caps = Length(name='capsnet')(digitcaps)
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])
    masked = Mask()(digitcaps)
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16 * n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))


    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model,model1, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    # log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    # tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
    #                            batch_size=args.batch_size, histogram_freq=int(args.debug))
    # checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
    #                                        save_best_only=True, save_weights_only=True, verbose=1)
    # lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})





    # Training without data augmentation:
    # model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
    #           callbacks=[log, tb, checkpoint, lr_decay])
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]])
    # model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
    #           validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])




    # Training with data augmentation:
    #-------
    # def train_generator(x, y, batch_size, shift_fraction=0.):
    #     train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
    #                                        height_shift_range=shift_fraction,
    #                                        horizontal_flip=True)  # shift up to 2 pixel for MNIST
    #     generator = train_datagen.flow(x, y, batch_size=batch_size)
    #     while 1:
    #         x_batch, y_batch = generator.next()
    #         yield ([x_batch, y_batch], [y_batch, x_batch])
    #
    # # Training with data augmentation. If shift_fraction=0., also no augmentation.
    # model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
    #                     steps_per_epoch=int(y_train.shape[0] / args.batch_size),
    #                     epochs=args.epochs,
    #                     validation_data=[[x_test, y_test], [y_test, x_test]],
    #                     callbacks=[log, tb, checkpoint, lr_decay])
    #-------




    # model.save_weights(args.save_dir + '/trained_model.h5')
    # print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    # from utils import plot_log
    # plot_log(args.save_dir + '/log.csv', show=True)

    # y_pred, x_recon = model1.predict(x_test, batch_size=100)
    # print('-' * 30 + 'Begin: test' + '-' * 30)
    # print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])

    # yy_pred = np.argmax(y_pred, axis=1)
    # oa = accuracy_score(np.argmax(y_test, axis=1), yy_pred)
    # confusion = confusion_matrix(np.argmax(y_test, axis=1), yy_pred)
    # each_acc, aa = AA_andEachClassAccuracy(confusion)
    # kappa = cohen_kappa_score(np.argmax(y_test, axis=1), yy_pred)
    # print('oa:', oa)
    # print('aa:', aa)
    # print('kappa:', kappa)
    # return model

def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def test(model, data, args, p, q, subject_id):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-' * 30 + 'Begin: test' + '-' * 30)
    print('ACCURACY=',p,',',q)
    test_acc = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0]
    print('Test acc:', test_acc)

    file1 = open("C:/Users/SB00763936/Desktop/EEG/imagined_speech_capsule/result6.txt", "a")
    file1.write(f'ACCURACY={subject_id},{p},{q},{test_acc}\n')
    file1.close()

    # yy_pred = np.argmax(y_pred, axis=1)
    # oa = accuracy_score(np.argmax(y_test, axis=1), yy_pred)
    # confusion = confusion_matrix(np.argmax(y_test, axis=1), yy_pred)
    # each_acc, aa = AA_andEachClassAccuracy(confusion)
    # kappa = cohen_kappa_score(np.argmax(y_test, axis=1), yy_pred)
    # print('oa:',oa)
    # print('aa:',aa)
    # print('kappa:',kappa)

    # img = combine_images(np.concatenate([x_test[:50], x_recon[:50]]))
    # image = img * 255
    # Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    # print()
    # print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    # print('-' * 30 + 'End: test' + '-' * 30)
    # plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    # plt.show()


def manipulate_latent(model, data, args):
    print('-' * 30 + 'Begin: manipulate' + '-' * 30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 15, 16])
    # noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:, :, dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    # img = combine_images(x_recons, height=16)
    # image = img * 255
    # Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    # print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    # print('-' * 30 + 'End: manipulate' + '-' * 30)

