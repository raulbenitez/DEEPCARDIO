import os
import sys

import cv2
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from deepcardio_utils import ImageReader
from pred.utils import SparkPredictor

if __name__=='__main__':
    args = sys.argv[1:]
    assert '--model' in args, 'USAGE: --model path_to_model.h5 [--imageid imageid]'

    sparkPredictor = SparkPredictor()
    imageReader = sparkPredictor.get_image_reader()
    X, Y = sparkPredictor.get_X_Y()
    Y_pred = sparkPredictor.predict(forcePrediction=True)

    sparkPredictor.get_prediction_statistics(Y_pred)
    falsePositives, falseNegatives, truePositives, trueNegatives = sparkPredictor.get_prediction_highlights(Y_pred)

    idxs = np.array(range(len(Y)))
    idx = np.random.choice(idxs[falseNegatives], 1)[0]
    imageReader.plot_img_circled_spark(idx)

    pass