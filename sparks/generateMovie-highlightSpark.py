import cv2
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.cluster import KMeans
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

    video = cv2.VideoWriter(video_name, 0, 30, (width,height*2+1))

    for i, image in enumerate(images):
        im = cv2.imread(os.path.join(IMAGE_FOLDER, image))
        if normalize:
            if F0 is None:
                F0, _, _ = get_f0_and_cellmask()
            im = get_normalized_frame(im, F0)

        im_ = im.copy()
        sparkLocations = get_spark_location(sparksDF, i)
        for i, sparkLocation in sparkLocations.iterrows():
            color = int(im.max()*2)
            cv2.circle(im, (sparkLocation['x'], sparkLocation['y']), 20, color, thickness=1, lineType=8, shift=0)

            mask = get_mask(im.shape[0], im.shape[1], sparkLocation['x'], sparkLocation['y'], 20)

            k = KMeans(n_clusters=3).fit(im_[mask].reshape(-1,3))
            maxRValue = im_[mask,2].max()
            maxValue = np.array([0, 0, maxRValue]).reshape(-1,3)
            maxValueIdx = im_[mask, 2].argmax()
            isSparkCond = np.full(im_[mask].shape, False)
            isSparkCond[k.labels_ == k.labels_[maxValueIdx], 2] = True
            im_[mask] = np.where(isSparkCond, maxRValue, im_[mask])
        maxRValue = im.max()
        conc_im = np.concatenate((im, np.full((1,im.shape[1], 3), maxRValue), im_))
        video.write(conc_im)

    cv2.destroyAllWindows()
    video.release()
