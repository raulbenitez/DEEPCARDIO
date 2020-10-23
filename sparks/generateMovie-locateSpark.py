import cv2
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from deepcardio_utils import *

if __name__=='__main__':
    normalize = '--normalize' in sys.argv[1:]
    video_name = f"{IMAGE_ID}{'_norm' if normalize else ''}.avi"

    images = sorted([img for img in os.listdir(IMAGE_FOLDER) if img.endswith(".tif")])
    frame = cv2.imread(os.path.join(IMAGE_FOLDER, images[0]))
    height, width, layers = frame.shape

    mat = loadmat(os.path.join(DATASETS_PATH, MAT_PATH))['xytspark']
    sparksDF = pd.DataFrame(mat, columns=['x','y','tIni','tFin'])
    F0 = None

    video = cv2.VideoWriter(video_name, 0, 30, (width,height))

    for i, image in enumerate(images):
        im = cv2.imread(os.path.join(IMAGE_FOLDER, image))
        if normalize:
            if F0 is None:
                F0, _, _ = get_f0_and_cellmask()
            im = get_normalized_frame(im, F0)
        sparkLocations = get_spark_location(sparksDF, i)
        for i, sparkLocation in sparkLocations.iterrows():
            color = int(im.max()*2)
            cv2.circle(im, (sparkLocation['x'], sparkLocation['y']), 20, color, thickness=1, lineType=8, shift=0)
        video.write(im)

    cv2.destroyAllWindows()
    video.release()
