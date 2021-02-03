import os
import sys
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from deepcardio_utils import ImageReader
from synthetic_data.synthetic import VERBOSE_SPARKS_FILE

DEFAULT_MODEL = 'pred/inceptionv3_ep30_b32__classWeights__synData2021-01-23_02-02-14.h5'
PREDS_FILE = 'pred_class.csv'


class SparkPredictor:
    def __init__(self, imageId=None, datasetsPath=None, model=None) -> None:
        super().__init__()

        if not model:
            args = sys.argv[1:]
            model = args[args.index('--model')+1] if '--model' in args else DEFAULT_MODEL

        self._datasetsPath = datasetsPath
        self._imageReader = ImageReader(imageId=imageId, datasetsPath=datasetsPath)
        self._modelPath = model

        # X & Y
        self._X = self._imageReader.get_full_padded_images()
        self._Y = self._imageReader.get_frame_wise_class_gmm(classesFromFile=True)
        # preprocess data
        self._X = self._X.astype('float32') / 255.

        self._inceptionv3 = keras.applications.InceptionV3(include_top=True, weights=None,
                                                           classes=2, input_shape=self._X[0].shape)
        self._inceptionv3.load_weights(self._modelPath)

    def get_X_Y(self):
        return self._X, self._Y

    def get_image_reader(self):
        return self._imageReader

    def get_preds_file_path(self):
        return os.path.join(self._imageReader.get_image_folder(), PREDS_FILE)

    def predict(self, forcePrediction=False):
        # from file
        if not forcePrediction and os.path.exists(self.get_preds_file_path()):
            return pd.read_csv(self.get_preds_file_path(), header=None, squeeze=True)

        # prediction
        Y_pred = self._inceptionv3.predict(self._X)
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
        print(f"Precision (spark / no-spark): {(precisionT, precisionF)}, " +\
              f"recall (spark / no-spark): {(recallT, recallF)}")

        self.get_prediciton_statistics_from_verbose_sparks(Y_pred, hightlights)

    def get_prediciton_statistics_from_verbose_sparks(self, Y_pred, hightlights):
        verboseSparkPath = os.path.join(self._imageReader.get_image_folder(), VERBOSE_SPARKS_FILE)
        if not os.path.exists(verboseSparkPath):
            return

        falsePositives, falseNegatives, truePositives, trueNegatives = hightlights
        df = pd.read_csv(verboseSparkPath, sep=';')
        df.loc[:, 'falseNegatives'] = falseNegatives[self._Y == 1].reset_index(drop=True)

        df.plot.scatter('size_sigma', 'noise_sigma', c=df.loc[:, 'falseNegatives'], cmap='viridis')
        plt.show()
        df.plot.scatter('size_sigma', 'spark_max', c=df.loc[:, 'falseNegatives'], cmap='viridis')
        plt.show()

        pass