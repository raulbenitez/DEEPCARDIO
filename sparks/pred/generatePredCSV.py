import os
import sys

import cv2
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from deepcardio_utils import ImageReader
from pred.utils import FrameWisePredictor

if __name__=='__main__':
    args = sys.argv[1:]
    assert '--model' in args, 'USAGE: --model path_to_model.h5 [--imageid imageid]'

    sparkPredictor = FrameWisePredictor(rollingSize=3)
    imageReader = sparkPredictor.get_image_reader()
    X, Y = sparkPredictor.get_X_Y()
    Y_pred = sparkPredictor.predict()

    sparkPredictor.get_prediction_statistics(Y_pred)
    falsePositives, falseNegatives, truePositives, trueNegatives = sparkPredictor.get_prediction_highlights(Y_pred)
    print(f"Total of {Y_pred.sum()} elements of class spark out of {Y_pred.shape[0]}")

    idxs = np.array(range(len(Y)))
    idx = np.random.choice(idxs[falseNegatives], 1)[0]
    imageReader.plot_img_circled_spark(idx)

    pass