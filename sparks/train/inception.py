import os
import pandas as pd

import keras

import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D,  \
    Dropout, Dense, Input, concatenate,      \
    GlobalAveragePooling2D, AveragePooling2D,\
    Flatten

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


def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj, name=None):

    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(x)

    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init,
                      bias_initializer=bias_init)(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
                       bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)

    return output

def build_model(h, w, nc):
    input_layer = Input(shape=(h, w, nc))

    global kernel_init
    global bias_init
    kernel_init = keras.initializers.glorot_uniform()
    bias_init = keras.initializers.Constant(value=0.2)

    x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2',
               kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
    x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
    x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

    x = inception_module(x, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128, filters_5x5_reduce=16,
                         filters_5x5=32, filters_pool_proj=32, name='inception_3a')

    x = inception_module(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192, filters_5x5_reduce=32,
                         filters_5x5=96, filters_pool_proj=64, name='inception_3b')

    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

    x = inception_module(x, filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208, filters_5x5_reduce=16,
                         filters_5x5=48, filters_pool_proj=64, name='inception_4a')

    x1 = AveragePooling2D((5, 5), strides=3)(x)
    x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(1024, activation='relu')(x1)
    x1 = Dropout(0.7)(x1)
    x1 = Dense(10, activation='softmax', name='auxilliary_output_1')(x1)

    x = inception_module(x, filters_1x1=160, filters_3x3_reduce=112, filters_3x3=224, filters_5x5_reduce=24,
                         filters_5x5=64, filters_pool_proj=64, name='inception_4b')

    x = inception_module(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=256, filters_5x5_reduce=24,
                         filters_5x5=64, filters_pool_proj=64, name='inception_4c')

    x = inception_module(x, filters_1x1=112, filters_3x3_reduce=144, filters_3x3=288, filters_5x5_reduce=32,
                         filters_5x5=64, filters_pool_proj=64, name='inception_4d')

    x2 = AveragePooling2D((5, 5), strides=3)(x)
    x2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
    x2 = Flatten()(x2)
    x2 = Dense(1024, activation='relu')(x2)
    x2 = Dropout(0.7)(x2)
    x2 = Dense(10, activation='softmax', name='auxilliary_output_2')(x2)

    x = inception_module(x, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5_reduce=32,
                         filters_5x5=128, filters_pool_proj=128, name='inception_4e')

    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

    x = inception_module(x, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5_reduce=32,
                         filters_5x5=128, filters_pool_proj=128, name='inception_5a')

    x = inception_module(x, filters_1x1=384, filters_3x3_reduce=192, filters_3x3=384, filters_5x5_reduce=48,
                         filters_5x5=128, filters_pool_proj=128, name='inception_5b')

    x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)

    x = Dropout(0.4)(x)

    x = Dense(10, activation='softmax', name='output')(x)

    model = Model(input_layer, [x, x1, x2], name='inception_v1')
    return model

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

def wrong_sparks_sigmoid(x):
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
    wrongSparksMetric = 1-wrong_sparks_sigmoid(wrongSparksMetricAux)


    # ret =  recall * 0.8 + accuracy * 0.2if realSparksCount > 0 else accuracy
    return tf.cond(tf.greater(realSparksCount, 0),
                   lambda: recall * 0.9 + wrongSparksMetric * 0.1,
                   lambda: wrongSparksMetric)

if __name__=='__main__':
    X_train, Y_train, X_valid, Y_valid = load_data(classesFromFile=True)

    # Inception V3
    modelv3 = keras.applications.InceptionV3(include_top=True, weights=None, classes=2, input_shape=X_train[0].shape)
    modelv3.summary()

    batch_size = 32
    epochs = 25

    opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
    modelv3.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[metric_spark_recall, 'accuracy'])

    modelv3.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, Y_valid), shuffle=True)

    modelv3.save('train/inceptionv32.h5')

    # # Inception V1 (manual)
    modelv1 = build_model(*X_train[0].shape)
    modelv1.summary()
    pass