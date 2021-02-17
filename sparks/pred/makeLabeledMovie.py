import sys
import cv2
import numpy as np

from pred.utils import SparkPredictor

if __name__=='__main__':
    args = sys.argv[1:]
    assert '--model' in args, 'USAGE: --model path_to_model.h5 [--imageid imageid]'

    sparkPredictor = SparkPredictor()
    sparkPredictor.generate_prediction_frames()