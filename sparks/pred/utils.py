from abc import ABC, abstractmethod
import os
import sys

import cv2
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils

from deepcardio_utils import ImageReader, get_spark_location
from synthetic_data.synthetic import VERBOSE_SPARKS_FILE
from train.pixelWiseLib import get_model

DEFAULT_MODEL = 'pred/inceptionv3_ep30_b32__classWeights__synData2021-01-23_02-02-14.h5'
PREDS_BASE_PATH = 'pred/predictions_db'
FRAMEWISE_PREDS_FILE = 'pred_class.csv'
PIXELWISE_PREDS_FILE = 'pixelwise_class.npy'


class BasePredictor(ABC):
    def __init__(self, imageId=None, datasetsPath=None, model=None) -> None:
        if not os.path.exists(PREDS_BASE_PATH):
            os.makedirs(PREDS_BASE_PATH)
        if not model:
            args = sys.argv[1:]
            model = args[args.index('--model')+1] if '--model' in args else DEFAULT_MODEL

        self._datasetsPath = datasetsPath
        self._imageReader = ImageReader(imageId=imageId, datasetsPath=datasetsPath)
        self._modelPath = model

        self._X = self._Y = self._model = None

        self.load_X_Y()
        self.load_model()

    @abstractmethod
    def load_X_Y(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def get_preds_filename(self):
        pass

    def get_model_id(self):
        return os.path.splitext(os.path.basename(self._modelPath))[0]

    def get_X_Y(self):
        return self._X, self._Y

    def get_image_reader(self):
        return self._imageReader

    def get_preds_dirname(self):
        dirn = os.path.join(PREDS_BASE_PATH, self._imageReader.get_image_id(), self.get_model_id())
        if not os.path.exists(dirn):
            os.makedirs(dirn)
        return dirn

    def get_preds_file_path(self):
        return os.path.join(self.get_preds_dirname(), self.get_preds_filename())

    @abstractmethod
    def predict(self, forcePrediction=False):
        pass

    @abstractmethod
    def get_prediction_frame(self, idx, image, Y_pred):
        pass

    def generate_prediction_frames(self):
        Y_pred = self.predict()
        images = self._imageReader.get_full_images()

        if not os.path.exists(os.path.join(self.get_preds_dirname(), 'figures')):
            os.makedirs(os.path.join(self.get_preds_dirname(), 'figures'))
        video_path = os.path.join(self.get_preds_dirname(), 'pred_labels.avi')
        height, width, _ = self._imageReader.get_shape()
        video = cv2.VideoWriter(video_path, 0, 30, (width, height*2+1))

        for idx, image in enumerate(images):
            if idx % 100 == 0:
                print(f"labeling frame {idx}/{len(images)}")

            image = self.get_prediction_frame(idx, image, Y_pred)

            fig = plt.figure(figsize=(20, 3))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.savefig(os.path.join(self.get_preds_dirname(), 'figures', str(idx).zfill(5)), bbox_inches='tight',
                        pad_inches=0)
            plt.close(fig)
            video.write(image)

        cv2.destroyAllWindows()
        video.release()


class FrameWisePredictor(BasePredictor):
    def load_X_Y(self):
        # X & Y
        self._X = self._imageReader.get_full_padded_images()
        self._Y = self._imageReader.get_frame_wise_class_gmm(classesFromFile=True)
        # preprocess data
        self._X = self._X.astype('float32') / 255.

    def load_model(self):
        self._model = keras.applications.InceptionV3(include_top=True, weights=None,
                                                     classes=2, input_shape=self._X[0].shape)
        self._model.load_weights(self._modelPath)

    def get_preds_filename(self):
        return FRAMEWISE_PREDS_FILE

    def predict(self, forcePrediction=False):
        # from file
        if not forcePrediction and os.path.exists(self.get_preds_file_path()):
            return pd.read_csv(self.get_preds_file_path(), header=None, squeeze=True)

        # prediction
        Y_pred = self._model.predict(self._X)
        Y_pred =  Y_pred.round().argmax(axis=-1)
        pd.Series(Y_pred).to_csv(self.get_preds_file_path(), header=False, index=False)
        return Y_pred

    def get_prediction_highlights(self, Y_pred):
        falsePositives = ~self._Y.astype(bool) & Y_pred.astype(bool)
        falseNegatives = self._Y.astype(bool) & ~Y_pred.astype(bool)
        truePositives = self._Y.astype(bool) & Y_pred.astype(bool)
        trueNegatives = ~self._Y.astype(bool) & ~Y_pred.astype(bool)

        return falsePositives, falseNegatives, truePositives, trueNegatives

    def get_prediction_statistics(self, Y_pred=None):
        if Y_pred is None:
            Y_pred = self.predict()

        # preditction statistics
        hightlights = self.get_prediction_highlights(Y_pred)
        falsePositives, falseNegatives, truePositives, trueNegatives = hightlights

        precisionT = round(truePositives.sum() / Y_pred.sum(), 4)
        precisionF = round(trueNegatives.sum() / (len(Y_pred) - Y_pred.sum()), 4)
        recallT = round(truePositives.sum() / self._Y.sum(), 4)
        recallF = round(trueNegatives.sum() / (len(self._Y) - self._Y.sum()), 4)
        predictionStatistics = f"Precision (spark / no-spark): {(precisionT, precisionF)}, " +\
                               f"recall (spark / no-spark): {(recallT, recallF)}"
        with open(os.path.join(self.get_preds_dirname(), 'statistics.txt'), 'w') as f:
            f.write(predictionStatistics)
        print(predictionStatistics)

        self.get_prediciton_statistics_from_verbose_sparks(Y_pred, hightlights)

    def get_prediciton_statistics_from_verbose_sparks(self, Y_pred, hightlights):
        verboseSparkPath = os.path.join(self._imageReader.get_image_folder(), VERBOSE_SPARKS_FILE)
        if not os.path.exists(verboseSparkPath):
            return

        falsePositives, falseNegatives, truePositives, trueNegatives = hightlights
        df = pd.read_csv(verboseSparkPath, sep=';')
        df.loc[:, 'falseNegatives'] = falseNegatives[self._Y == 1]

        df.plot.scatter('size_sigma', 'noise_sigma', c=df.loc[:, 'falseNegatives'], cmap='viridis')
        plt.show()
        df.plot.scatter('size_sigma', 'spark_max', c=df.loc[:, 'falseNegatives'], cmap='viridis')
        plt.show()
        pass

    def get_prediction_frame(self, idx, image, Y_pred):
        im = image.copy()
        if Y_pred[idx]:
            cv2.putText(im, 'spark', (15, 15), cv2.FONT_HERSHEY_SIMPLEX, .25, (255, 255, 255))

        return self._imageReader.get_image_with_circled_sparks(idx, im)


class PixelWisePredictor(BasePredictor):
    def load_X_Y(self):
        self._X = self._imageReader.get_full_images().astype(np.float32) / 255.

        try:
            self._Y = imageReader.get_pixel_wise_classification()
            self._Y = np_utils.to_categorical(self._Y, 2)
        except Exception as e:
            self._Y = None

    def load_model(self):
        imagesShape = self._imageReader.get_full_images()[0].shape
        self._model = get_model(imagesShape[:-1], 2)
        # Free up RAM in case the model definition cells were run multiple times
        keras.backend.clear_session()

        # Configure the model for training.
        # We use the "sparse" version of categorical_crossentropy
        # because our target data is integers.
        self._model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
        self._model.load_weights(self._modelPath)

    def get_preds_filename(self):
        return PIXELWISE_PREDS_FILE

    def get_image_reader(self):
        return self._imageReader

    def predict(self, forcePrediction=False):
        # from file
        if not forcePrediction and os.path.exists(self.get_preds_file_path()):
            return np.load(self.get_preds_file_path())

        Y_pred = (self._model.predict(self._X)[:, :, :, 1] > 0.75).astype(int)
        np.save(self.get_preds_file_path(), Y_pred)
        return Y_pred

    def get_spark_centroid_from_pred(self, pred):
        return tuple(c.mean().astype(int) for c in np.where(pred))

    def get_prediction_frame(self, idx, image, Y_pred):
        im = image.copy()
        predIm = np.zeros(im.shape)

        if Y_pred[idx].any():
            im = self._imageReader.get_image_with_circled_sparks(idx, im, self.get_spark_centroid_from_pred(Y_pred[idx]))
            predIm[:, :, 2] = (Y_pred[idx] * 255)

        return np.concatenate((im, np.full((1,image.shape[1], 3), 255), predIm)).astype('uint8')


if __name__=='__main__':
    predictor = PixelWisePredictor('170215_RyR-GFP30_RO_01_Serie2_SPARKS-calcium', model='pred/pixelWiseUNet.h5')
    imageReader = predictor.get_image_reader()
    images = imageReader.get_full_images()
    X, _ = predictor.get_X_Y()
    Y_pred = predictor.predict()

    rSparkCond = Y_pred.reshape(Y_pred.shape[0], -1).any(axis=-1)

    for rSparkIdx in np.random.choice(np.arange(Y_pred.shape[0])[rSparkCond], 10):  # [15, 52, 95]: #

        auxIm = X[rSparkIdx].copy()
        auxIm[:, :, 1] = Y_pred[rSparkIdx] / 10.

        plt.figure(figsize=(20, 5))
        plt.imshow(cv2.cvtColor(auxIm, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()