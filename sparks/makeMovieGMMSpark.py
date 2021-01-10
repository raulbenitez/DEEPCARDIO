import cv2
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.cluster import KMeans
from skimage.filters import gaussian
from deepcardio_utils import *

def get_spark_visibility_frame_mask2(sparkIni, sparkFin, sparkX, sparkY):
    nCompList = []
    for j, idx in enumerate(range(sparkIni, sparkFin)):
        im = cv2.imread(get_image_path(idx))
        imOutput = im.copy()
        imFiltered = gaussian(im.copy(), sigma=1, multichannel=True, preserve_range=True).astype('uint8')

        mask = get_mask(im.shape[0], im.shape[1], sparkX, sparkY, 20)
        nComp, bic = get_optimal_ncomponents_and_bic_gmm(imFiltered[mask].reshape(-1,3))
        nCompList.append(nComp)
    nCompList = np.array(nCompList)
    return np.logical_or((nCompList == 2), (nCompList == 3))

if __name__=='__main__':
    args = sys.argv[1:]
    normalize = '--normalize' in args

    imageReader = ImageReader()
    video_name = f"{imageReader.get_image_id()}{'_norm' if normalize else ''}.avi"
    height, width, layers = imageReader.get_shape()

    sparksDF = imageReader.get_sparks_df()
    F0 = None

    video = cv2.VideoWriter(video_name, 0, 30, (width,height*3+2))

    for i, image in enumerate(imageReader.get_images_names()):
        im = cv2.imread(os.path.join(imageReader.get_image_folder(), image))
        if normalize:
            if F0 is None:
                F0, _, _ = get_f0_and_cellmask()
            im = get_normalized_frame(im, F0)

        imOutput = im.copy()
        imFiltered = gaussian(im.copy(), sigma=1, multichannel=True, preserve_range=True).astype('uint8')
        sparkLocations = get_spark_location(sparksDF, i)
        for _, sparkLocation in sparkLocations.iterrows():
            # sparkVisibilityMask = get_spark_visibility_frame_mask2(sparkLocations.loc[0, 'tIni'], sparkLocations.loc[0, 'tFin'], sparkLocations.loc[0, 'x'], sparkLocations.loc[0, 'y'])

            mask = get_mask(im.shape[0], im.shape[1], sparkLocation['x'], sparkLocation['y'], 20)
            nComp, bic = get_optimal_ncomponents_and_bic_gmm(imFiltered[mask].reshape(-1,3))

            # visibilityEnabled
            if (nComp == 2) or (nComp == 3):
                gmm = mixture.GaussianMixture(n_components=nComp).fit(imFiltered[mask].reshape(-1,3))
                lab = gmm.predict(imFiltered[mask].reshape(-1,3))
                probs = gmm.predict_proba(imFiltered[mask].reshape(-1,3))
                mixtIdx = gmm.means_[:, 2].argmax()

                maxRValue = im[mask,2].max()
                maxValue = np.array([0, 0, maxRValue]).reshape(-1,3)
                maxValueIdx = im[mask, 2].argmax()
                isSparkCond = np.full(imOutput[mask].shape, False)
                isSparkCond[lab == mixtIdx, 1] = True # probs[:,mixtIdx]>.8 (amb nCom fixat (2/3)) vs lab == mixtIdx (automatic nComp)
                imOutput[mask] = np.where(isSparkCond, maxRValue, imOutput[mask])

            color = int(im.max()*2)
            cv2.circle(imOutput, (sparkLocation['x'], sparkLocation['y']), 20, color, thickness=1, lineType=8, shift=0)

        maxRValue = im.max()
        conc_im = np.concatenate((im, np.full((1,im.shape[1], 3), maxRValue), imOutput, np.full((1,im.shape[1], 3), maxRValue), imFiltered))
        video.write(conc_im)

    cv2.destroyAllWindows()
    video.release()
