import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat
from skimage.filters import threshold_otsu, gaussian
from sklearn import mixture

DATASETS_PATH = '../_datasets/deepcardio'
IMAGE_ID = '170215_RyR-GFP30_RO_01_Serie2_SPARKS-calcium'
IMAGE_FOLDER = os.path.join(DATASETS_PATH, IMAGE_ID)
IMAGE_FILE_TEMPLATE = '170215_RyR-GFP30_RO_01_Serie2_z1{}_ch01.tif'
MAT_PATH = '170215_RyR-GFP30_01_Serie2_Sparks.mat'

BACKGROUND_MAX_VALUE = 0

def get_image_path(idx):
    return os.path.join(IMAGE_FOLDER, IMAGE_FILE_TEMPLATE.format(str(idx).zfill(4)))

def get_image_array(idx):
    imagePath = get_image_path(idx)
    return np.array(Image.open(imagePath))

def get_spark_location_(sparksDF, idx):
    candidates = sparksDF.loc[(sparksDF.loc[:,'tIni'] <= idx) & (sparksDF.loc[:, 'tFin'] >= idx), :]
    # assert len(candidates) <= 1
    return candidates.loc[candidates.index.min(), ['x', 'y']]

def get_spark_location(sparksDF, idx):
    candidates = sparksDF.loc[(sparksDF.loc[:,'tIni'] <= idx) & (sparksDF.loc[:, 'tFin'] > idx), :]
    return candidates.loc[:, ['x', 'y']]

def get_mask(h, w, centerx, centery, radius):
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - centerx)**2 + (y-centery)**2)
    return dist_from_center <= radius

def get_f0_and_cellmask():
    imageList = sorted(os.listdir(IMAGE_FOLDER))

    cellMask = np.zeros(get_image_array(0).shape)
    for i in range(len(imageList)):
        im1 = get_image_array(i)
        auxMask = im1 > BACKGROUND_MAX_VALUE
        cellMask += auxMask

    cellMask = cellMask > threshold_otsu(cellMask)

    res = pd.Series(dtype='int')
    for i in range(len(imageList)):
        im1 = get_image_array(i)
        s1 = pd.Series(im1[cellMask].flatten())
        g1 = s1.groupby(s1).size()

        res = pd.concat([res, g1], axis=1).fillna(0, downcast='infer').sum(axis=1)

    return (res*res.index).sum()/res.sum(), res.index.max(), cellMask

def get_normalized_frame(im, F0, max=None):
    amax = max if max else im.max()
    mmax = (amax-F0)/F0+1
    if len(im.shape) > 2:
        im_ = im.copy()
        im_[:, :, 2] = ((im[:, :, 2]-F0)/F0 + 1)/mmax*255
        return im_
    return (im-F0)/F0

def get_optimal_ncomponents_and_bic_gmm(data, nmin=1, nmax=10):
    #calcula el BIC per trobar el número de gaussianes òptim
    bic = []
    for kG in np.arange(nmin,nmax+1):
        gmm = mixture.GaussianMixture(n_components=kG).fit(data)
        bic.append(gmm.bic(data)) #cada cop va afegint el bic amb kG+1, així ho tens tot en un vector i pots calcualr el mínim
    return np.argmin(bic)+nmin, min(bic)

def is_spark_visible(imFiltered, mask):
    nComp, bic = get_optimal_ncomponents_and_bic_gmm(imFiltered[mask].reshape(-1, imFiltered.shape[-1]))
    return nComp == 2 or nComp == 3

def get_pixel_wise_classification(frameList, sparksDF=None):
    if not sparksDF:
        mat = loadmat(os.path.join(DATASETS_PATH, MAT_PATH))['xytspark']
        sparksDF = pd.DataFrame(mat, columns=['x', 'y', 'tIni', 'tFin'])

    sparkLocationsList = []

    for i, idx in enumerate(frameList):
        im = cv2.imread(get_image_path(idx))
        imFiltered = gaussian(im.copy(), sigma=1, multichannel=True, preserve_range=True).astype('uint8')

        sparkLocationsList.append(np.full(im.shape[:-1], False))

        sparkLocationsDF = get_spark_location(sparksDF, idx)
        for _, sparkLocation in sparkLocationsDF.iterrows():
            mask = get_mask(im.shape[0], im.shape[1], sparkLocation['x'], sparkLocation['y'], 20)

            if is_spark_visible(imFiltered, mask):
                aux = imFiltered[mask].reshape(-1, imFiltered.shape[-1])
                nComp, bic = get_optimal_ncomponents_and_bic_gmm(aux)
                gmm = mixture.GaussianMixture(n_components=nComp).fit(aux)
                lab = gmm.predict(aux)
                mixtIdx = gmm.means_[:, 2].argmax()
                isSparkCond = np.copy(mask)
                isSparkCond[isSparkCond] = lab == mixtIdx

                sparkLocationsList[i] += isSparkCond

    return np.array(sparkLocationsList)

def get_frame_wise_classification(frameList, sparksDF=None):
    if not sparksDF:
        mat = loadmat(os.path.join(DATASETS_PATH, MAT_PATH))['xytspark']
        sparksDF = pd.DataFrame(mat, columns=['x', 'y', 'tIni', 'tFin'])

    classes = np.full((len(frameList), 1), False)

    for i, idx in enumerate(frameList):
        im = cv2.imread(get_image_path(idx))
        imFiltered = gaussian(im.copy(), sigma=1, multichannel=True, preserve_range=True).astype('uint8')

        sparkLocationsDF = get_spark_location(sparksDF, idx)
        for _, sparkLocation in sparkLocationsDF.iterrows():
            mask = get_mask(im.shape[0], im.shape[1], sparkLocation['x'], sparkLocation['y'], 20)
            classes[i] = classes[i] or is_spark_visible(imFiltered, mask)

    return classes

def get_gmm_from_all_sparks():
    imageFilesList = sorted([img for img in os.listdir(IMAGE_FOLDER) if img.endswith(".tif")])
    mat = loadmat(os.path.join(DATASETS_PATH, MAT_PATH))['xytspark']
    sparksDF = pd.DataFrame(mat, columns=['x', 'y', 'tIni', 'tFin'])
    fullSparkPixelsList = []
    for i, image in enumerate(imageFilesList):
        im = cv2.imread(os.path.join(IMAGE_FOLDER, image))
        imFiltered = gaussian(im.copy(), sigma=1, multichannel=True, preserve_range=True).astype('uint8')

        sparkLocations = get_spark_location(sparksDF, i)
        for _, sparkLocation in sparkLocations.iterrows():
            mask = get_mask(im.shape[0], im.shape[1], sparkLocation['x'], sparkLocation['y'], 20)
            nComp, bic = get_optimal_ncomponents_and_bic_gmm(imFiltered[mask].reshape(-1,3))

            if (nComp == 2) or (nComp == 3):
                fullSparkPixelsList.append(imFiltered[mask].reshape(-1, 3))
    fullSparkPixels = np.concatenate(tuple(fullSparkPixelsList))
    del fullSparkPixelsList
    nComp, bic = get_optimal_ncomponents_and_bic_gmm(fullSparkPixels)
    gmm = mixture.GaussianMixture(n_components=nComp).fit(fullSparkPixels)
    np.savetxt('fullSparkPixels.csv', fullSparkPixels, delimiter=";")
    return gmm

if __name__=='__main__':
    # gmm = get_gmm_from_all_sparks()

    aux = get_frame_wise_classification([2000, 2005])
    aux = get_pixel_wise_classification([2000, 2005])

    pass