import keras
import keras
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from deepcardio_utils import IMAGE_ID, DATASETS_PATH, ImageReader, get_rolled_images


def load_data(classesFromFile=False, imageId=IMAGE_ID, datasetsPath=DATASETS_PATH, gaussianFilter=False, rolledImages=False):
    imageReader = ImageReader(imageId=imageId, datasetsPath=datasetsPath)
    images = imageReader.get_full_padded_images(gaussianFilter=gaussianFilter)

    if rolledImages:
        images = get_rolled_images(images)

    imageIdxs = list(range(images.shape[0]))

    classes = imageReader.get_frame_wise_class_gmm(imageIdxs, classesFromFile=classesFromFile)

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

def _sigmoid(x):
    return 2/(1+tf.math.exp(-5*x))-1

def _recall_and_false_spark(ratio=.6):
    def recall_and_false_spark(y_true, y_pred):
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
                       lambda: recall * ratio + wrongSparksMetric * (1-ratio),
                       lambda: wrongSparksMetric)
    return recall_and_false_spark

def sigmoid_spark_and_non_spark_loss(y_true, y_pred):
    mul = tf.constant([0, 1], shape=[2, 1], dtype=y_true.dtype)

    realSparks = tf.squeeze(tf.matmul(y_true, mul) == 1)
    predSparks = tf.squeeze(tf.matmul(y_pred, mul) == 1)
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
    a = tf.cast(tf.convert_to_tensor([[[0, 1] if np.random.randint(0,2) else [1, 0] for _ in range(5)] for _ in range(10)]), np.float32)
    b = tf.cast(tf.convert_to_tensor([[[0, 1] if np.random.randint(0, 2) else [1, 0] for _ in range(5)] for _ in range(10)]), np.float32)
    sigmoid_loss(a, b)
    sigmoid_spark_and_non_spark_loss(a, b)

    a = tf.convert_to_tensor([[0, 1] if np.random.randint(0, 2) else [1, 0] for _ in range(5)])
    b = tf.convert_to_tensor([[0, 1] if np.random.randint(0, 2) else [1, 0] for _ in range(5)])
    sigmoid_spark_and_non_spark_loss(a, b)

    X_train, Y_train, X_valid, Y_valid = load_data(classesFromFile=True)

    # Inception V3
    modelv3 = keras.applications.InceptionV3(include_top=True, weights=None, classes=2, input_shape=X_train[0].shape)
    modelv3.summary()

    batch_size = 32
    epochs = 25

    opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
    modelv3.compile(loss=sigmoid_loss, optimizer=opt, metrics=[_recall_and_false_spark(0.6)])

    modelv3.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, Y_valid), shuffle=True)

    modelv3.save('train/inceptionv32.h5')
    pass