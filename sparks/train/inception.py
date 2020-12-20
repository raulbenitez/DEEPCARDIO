import os
import pandas as pd

import keras

import tensorflow as tf

import cv2
import numpy as np
from keras.utils import np_utils

from sklearn.model_selection import train_test_split

from deepcardio_utils import IMAGE_FOLDER, get_frame_wise_classification


def load_data(classesFromFile=False, imageFolder=IMAGE_FOLDER):
    # from images
    # imagePaths = sorted([img for img in os.listdir(imageFolder) if img.endswith(".tif")])
    # imageIdxs = list(range(len(imagePaths)))
    # images = np.array([cv2.imread(os.path.join(imageFolder, imagePaths[i])) for i in imageIdxs])
    # np.save(IMAGE_FOLDER+'/full_images.npy', images)

    images = np.load(os.path.join(imageFolder, 'full_images.npy'))
    # reshape for cnn input
    minsize = 75
    images = np.array([np.concatenate((im, np.full((minsize-im.shape[0], 256, 3), 0))) for im in images])

    imageIdxs = list(range(images.shape[0]))

    classesPath = os.path.join(imageFolder, 'class.csv')
    if not classesFromFile:
        classes = get_frame_wise_classification(imageIdxs)
        np.savetxt(classesPath, classes, delimiter=";", fmt='%d')
    else:
        classes = pd.read_csv(classesPath, header=None).astype(bool)

    # Transform targets to keras compatible format
    num_classes = 2
    Y = np_utils.to_categorical(classes, num_classes)

    # preprocess data
    X = images.astype('float32')
    X = X / 255.0

    # Split train / test data
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=1)
    print(f"Prop of sparks in train dataset: {round(Y_train.sum(axis=0)[1]/Y_train.shape[0]*100, 2)}, "
          f"and in validation dataset: {round(Y_valid.sum(axis=0)[1]/Y_valid.shape[0]*100, 2)}")

    return X_train, Y_train, X_valid, Y_valid

    # Resize training images
    X_train = np.array([cv2.resize(img, (img_rows, img_cols)) for img in X_train[:, :, :, :]])
    X_valid = np.array([cv2.resize(img, (img_rows, img_cols)) for img in X_valid[:, :, :, :]])


def old_metric_spark_recall(y_true, y_pred):
    realSparks = tf.math.argmax(y_true, axis=1) == 1
    predSparks = tf.math.argmax(y_pred, axis=1) == 1
    wellPredSparks = realSparks & predSparks
    return tf.math.reduce_sum(tf.cast(wellPredSparks, np.float32)) / tf.math.reduce_sum(tf.cast(realSparks, np.float32))

def metric_08recall_02accuracy(y_true, y_pred):
    # metrica ponderada 0.2 accuracy + 0.8 recall (classes spark exclusivament), sempre que hi hagi sparks, sino acc
    realSparks = tf.math.argmax(y_true, axis=1) == 1
    predSparks = tf.math.argmax(y_pred, axis=1) == 1
    wellPredSparks = realSparks & predSparks
    wellPredNoSparks = ~realSparks & ~predSparks

    realSparksCount = tf.math.reduce_sum(tf.cast(realSparks, np.float32))
    wellPredSparksCount = tf.math.reduce_sum(tf.cast(wellPredSparks, np.float32))
    wellPredCount = tf.math.reduce_sum(tf.cast(wellPredNoSparks, np.float32)) + wellPredSparksCount

    recall = wellPredSparksCount / realSparksCount
    accuracy = wellPredCount / tf.cast(tf.shape(y_true)[0], np.float32)

    # ret =  recall * 0.8 + accuracy * 0.2if realSparksCount > 0 else accuracy
    return tf.cond(tf.greater(realSparksCount, 0),
                   lambda: recall * 0.8 + accuracy * 0.2,
                   lambda: accuracy)

def _sigmoid(x):
    return 2/(1+tf.math.exp(-5*x))-1

def metric_09recall_01sigmoid_wrong_spark(y_true, y_pred):
    # metrica ponderada 0.2 accuracy + 0.8 recall (classes spark exclusivament), sempre que hi hagi sparks, sino acc
    realSparks = tf.math.argmax(y_true, axis=1) == 1
    predSparks = tf.math.argmax(y_pred, axis=1) == 1
    wellPredSparks = realSparks & predSparks
    wellPredNoSparks = ~realSparks & ~predSparks
    wrongPredSparks = ~realSparks & predSparks

    totalElementsCount = tf.cast(tf.shape(y_true)[0], np.float32)
    realSparksCount = tf.math.reduce_sum(tf.cast(realSparks, np.float32))
    wellPredSparksCount = tf.math.reduce_sum(tf.cast(wellPredSparks, np.float32))
    wellPredCount = tf.math.reduce_sum(tf.cast(wellPredNoSparks, np.float32)) + wellPredSparksCount
    wrongPredSparksCount = tf.math.reduce_sum(tf.cast(wrongPredSparks, np.float32))
    realNonSparksCount = tf.math.reduce_sum(tf.cast(~realSparks, np.float32))

    recall = wellPredSparksCount / realSparksCount
    accuracy = wellPredCount / totalElementsCount
    wrongSparksMetricAux = wrongPredSparksCount / realNonSparksCount
    wrongSparksMetric = 1 - _sigmoid(wrongSparksMetricAux)


    # ret =  recall * 0.8 + accuracy * 0.2if realSparksCount > 0 else accuracy
    return tf.cond(tf.greater(realSparksCount, 0),
                   lambda: recall * 0.6 + wrongSparksMetric * 0.4,
                   lambda: wrongSparksMetric)

def sigmoid_spark_and_non_spark_loss(y_true, y_pred):
    realSparks = tf.math.argmax(y_true, axis=-1) == 1
    predSparks = tf.math.argmax(y_pred, axis=-1) == 1
    wrongPredSparks = ~realSparks & predSparks
    wrongPredNonSparks = realSparks & ~predSparks

    realSparksCount = tf.math.reduce_sum(tf.cast(realSparks, np.float32), axis=-1)
    realNonSparksCount = tf.math.reduce_sum(tf.cast(~realSparks, np.float32), axis=-1)
    wrongPredSparksCount = tf.math.reduce_sum(tf.cast(wrongPredSparks, np.float32), axis=-1)
    wrongPredNonSparksCount = tf.math.reduce_sum(tf.cast(wrongPredNonSparks, np.float32), axis=-1)

    wrongSparksMetricAux = wrongPredSparksCount / realNonSparksCount
    wrongPredNonSparks = wrongPredNonSparksCount / realSparksCount
    aux = _sigmoid(wrongSparksMetricAux)
    comb = 0.5 * aux + 0.5 * wrongPredNonSparks
    return tf.where(tf.math.is_nan(comb), aux, comb)

def sigmoid_loss(y_true, y_pred):
    return _sigmoid(tf.reduce_mean(tf.square(y_true-y_pred), axis=-1))

if __name__=='__main__':
    X_train, Y_train, X_valid, Y_valid = load_data(classesFromFile=True)

    # Inception V3
    modelv3 = keras.applications.InceptionV3(include_top=True, weights=None, classes=2, input_shape=X_train[0].shape)
    modelv3.summary()

    batch_size = 32
    epochs = 25

    opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
    modelv3.compile(loss=sigmoid_spark_and_non_spark_loss, optimizer=opt, metrics=[metric_09recall_01sigmoid_wrong_spark])

    modelv3.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, Y_valid), shuffle=True)

    modelv3.save('train/inceptionv32.h5')
    pass