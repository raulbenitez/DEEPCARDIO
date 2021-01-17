import os
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat
from scipy.stats import norm
from skimage.filters import threshold_otsu, gaussian
from sklearn import mixture

DATASETS_PATH = '../_datasets/deepcardio'
IMAGE_ID = '170215_RyR-GFP30_RO_01_Serie2_SPARKS-calcium'
IMAGE_FOLDER = os.path.join(DATASETS_PATH, IMAGE_ID)
IMAGE_FILE_TEMPLATE = '170215_RyR-GFP30_RO_01_Serie2_z1{}_ch01.tif'
MAT_PATH = '170215_RyR-GFP30_01_Serie2_Sparks.mat'

BACKGROUND_MAX_VALUE = 0

class ImageReader:
    def __init__(self, imageId=None, datasetsPath=None) -> None:
        super().__init__()

        if not imageId:
            args = sys.argv[1:]
            imageId = args[args.index('--imageid')+1] if '--imageid' in args else IMAGE_ID
        self._datasetsPath = datasetsPath if datasetsPath else DATASETS_PATH
        self._imageId = imageId
        self._imageFolderPath = os.path.join(self._datasetsPath, self._imageId)
        self._imagesNames = sorted([img for img in os.listdir(self._imageFolderPath) if img.endswith(".tif")])
        self._matFile = None

    def get_datasets_path(self):
        return self._datasetsPath

    def get_image_folder(self):
        return self._imageFolderPath

    def get_image_id(self):
        return self._imageId

    def get_image_path(self, idx):
        return os.path.join(self._imageFolderPath, self._imagesNames[idx])

    def get_shape(self):
        frame = cv2.imread(os.path.join(self._imageFolderPath, self._imagesNames[0]))
        return frame.shape

    def get_mat_file(self):
        if self._matFile:
            return self._matFile
        matFiles = [img for img in os.listdir(self._imageFolderPath) if img.endswith(".mat")]
        csvFiles = [img for img in os.listdir(self._imageFolderPath) if img.endswith(".csv") and not img.startswith('class')]
        assert (len(matFiles)+len(csvFiles))==1, 'It is only possible to have sparks either in matlab form or csv form.'

        self._matFile = matFiles[0] if len(matFiles) else csvFiles[0]
        return self._matFile

    def get_sparks_df(self):
        matFile = self.get_mat_file()
        matFilePath = os.path.join(self._imageFolderPath, matFile)
        if matFile.endswith('.mat'):
            mat = loadmat(matFilePath)['xytspark']
            sparksDF = pd.DataFrame(mat, columns=['x', 'y', 'tIni', 'tFin'])
        else: # csv
            sparksDF = pd.read_csv(matFilePath, sep=',')
            sparksDF.columns = ['x', 'y', 'tIni', 'tFin']

        return sparksDF

    def get_images_names(self):
        return self._imagesNames

    def create_npy_file(self):
        # from images
        imagePaths = sorted([img for img in os.listdir(self._imageFolderPath) if img.endswith(".tif")])
        imageIdxs = list(range(len(imagePaths)))
        images = np.array([cv2.imread(os.path.join(self._imageFolderPath, imagePaths[i])) for i in imageIdxs])
        np.save(self._imageFolderPath+'/full_images.npy', images)

    def get_full_images(self, gaussianFilter=False):
        if not len([img for img in os.listdir(self._imageFolderPath) if img.endswith(".npy")]):
            self.create_npy_file()
        images = np.load(os.path.join(self._imageFolderPath, 'full_images.npy'))

        if gaussianFilter:
            images = [gaussian(im, sigma=1, multichannel=True, preserve_range=True).astype('uint8') for im in images]

        return images

    def get_full_padded_images(self, minsize=75, gaussianFilter=False):
        images = self.get_full_images(gaussianFilter=gaussianFilter)
        height, width, layers = self.get_shape()
        # reshape for cnn input
        return np.array(
            [np.concatenate((im, np.full((minsize - im.shape[0], width, layers), 0))).astype('uint8') for im in images])

    def get_frame_wise_class_gmm(self, frameList=None, classesFromFile=False, sparksDF=None):
        if not frameList:
            frameList=list(range(len(self._imagesNames)))
        classesPath = os.path.join(self.get_image_folder(), 'class.csv')
        if classesFromFile:
            return pd.read_csv(classesPath, header=None, sep=';').loc[frameList].squeeze().to_numpy()

        if not sparksDF:
            sparksDF = self.get_sparks_df()

        classes = np.full((len(frameList), 1), False)

        for i, idx in enumerate(frameList):
            im = cv2.imread(self.get_image_path(idx))
            imFiltered = gaussian(im.copy(), sigma=1, multichannel=True, preserve_range=True).astype('uint8')

            sparkLocationsDF = get_spark_location(sparksDF, idx)
            for _, sparkLocation in sparkLocationsDF.iterrows():
                mask = get_mask(im.shape[0], im.shape[1], sparkLocation['x'], sparkLocation['y'], 20)
                classes[i] = classes[i] or is_spark_visible_gmm(imFiltered, mask)

        np.savetxt(classesPath, classes, delimiter=";", fmt='%d')
        return classes

    def get_frame_wise_class_windowed(self, frameList=None, classesFromFile=False, sparksDF=None):
        if not frameList:
            frameList=list(range(len(self._imagesNames)))
        classesPath = os.path.join(self.get_image_folder(), 'class_win.csv')
        if classesFromFile:
            return pd.read_csv(classesPath, header=None, sep=';').loc[frameList].squeeze().to_numpy()

        if not sparksDF:
            sparksDF = self.get_sparks_df()

        sparksDF.loc[:, 'maxIdx'] = pd.Series(np.nan, index=sparksDF.index)

        for sidx, spark in sparksDF.iterrows():
            meansList = []
            for idx in range(int(spark['tIni']), int(spark['tFin'])):
                im = cv2.imread(self.get_image_path(idx))
                imFiltered = gaussian(im.copy(), sigma=1, multichannel=True, preserve_range=True).astype('uint8')
                mask = get_mask(im.shape[0], im.shape[1], spark['x'], spark['y'], 20)
                meansList.append(imFiltered[mask].reshape(-1,1).mean())
            sparksDF.loc[sidx, 'maxIdx'] = spark['tIni']+np.argmax(meansList)
        sparksDF = sparksDF.astype({'maxIdx': int})

        classes = np.full((len(frameList),), False)

        for i, idx in enumerate(frameList):
            im = cv2.imread(self.get_image_path(idx))
            imFiltered = gaussian(im.copy(), sigma=1, multichannel=True, preserve_range=True).astype('uint8')

            sparkLocationsDF = get_spark_location(sparksDF, idx)
            for _, sparkLocation in sparkLocationsDF.iterrows():
                mask = get_mask(im.shape[0], im.shape[1], sparkLocation['x'], sparkLocation['y'], 20)
                SPARK_WINDOW = 3
                classes[i] = classes[i] or (idx == sparkLocation['maxIdx']) or \
                             ((idx-3 <= sparkLocation['maxIdx'] <= idx+3) and is_spark_visible_gmm(imFiltered, mask))

        classes = classes.astype(int)
        np.savetxt(classesPath, classes, delimiter=";", fmt='%d')
        return classes

    def get_cellmask(self, images=None):
        if images is None:
            images = self.get_full_images()
        cellMask = images[:, :, :, 2].sum(axis=0).astype(int)
        return cellMask > threshold_otsu(cellMask)

    def background_noise_images_generator(self, multichannel=False):
        images = self.get_full_images()
        shp = images[0].shape

        # masking all sparks for generating background noise
        superMask = np.full((len(images), shp[0], shp[1]), False)
        sparksDF = self.get_sparks_df()
        for i, sparkS in sparksDF.iterrows():
            mask = get_mask(*shp[:2], *sparkS[['x', 'y']])
            superMask[sparkS['tIni']:sparkS['tFin']] = mask
        superMask = superMask.reshape((-1,))

        # cell mask
        cellMask = self.get_cellmask(images)

        # background noise image
        flatImages = images[:, :, :, 2].flatten()[~superMask]

        while True:
            noisyImage = np.random.choice(flatImages, shp[:2])
            noisyImage[~cellMask] = 0
            if multichannel:
                aux = np.full(shp, 0)
                aux[:, :, 2] = noisyImage
                noisyImage = aux
            yield noisyImage.astype(np.uint8)

    def spark_images_generator(self, multichannel=False):
        images = self.get_full_images()
        shp = images[0].shape
        noisyGen = self.background_noise_images_generator()

        # cell mask
        cellMask = self.get_cellmask(images)

        def gen_spark(sparkCentroid, sparkSigma=0.2, noiseSigma=0.5):
            # circle mask (from centroid)
            circMask = get_mask(*shp[:2], sparkCentroid[1], sparkCentroid[0])

            # distance matrix
            y, x = np.ogrid[:shp[0], :shp[1]]
            centerDist = np.sqrt((x - sparkCentroid[1]) ** 2 + (y - sparkCentroid[0]) ** 2)
            distProp = centerDist / centerDist[circMask].max()
            # based on normal pdf
            distProp = norm.pdf(distProp, loc=0, scale=sparkSigma)
            # with some noise
            noisyDistProp = np.random.normal(1, noiseSigma, shp[:2]) * distProp

            # create spark image
            sparkMaxValue = 127
            pureSparkImage = noisyDistProp * sparkMaxValue
            sparkImage = next(noisyGen)
            sparkImage[circMask] = np.where(pureSparkImage > sparkImage, pureSparkImage, sparkImage)[circMask]
            sparkImage[sparkImage > 255] = 255
            sparkImage[sparkImage < 0] = 0
            sparkImage[~cellMask] = 0
            if multichannel:
                aux = np.full(shp, 0)
                aux[:, :, 2] = sparkImage
                sparkImage = aux
            return sparkImage.astype(np.uint8)
        return gen_spark


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
    return candidates

def get_mask(h, w, centerx, centery, radius=20):
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

def is_spark_visible_gmm(imFiltered, mask):
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

            if is_spark_visible_gmm(imFiltered, mask):
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
            classes[i] = classes[i] or is_spark_visible_gmm(imFiltered, mask)

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
    nComp, bic = get_optimal_ncomponents_and_bic_gmm(fullSparkPixels)
    gmm = mixture.GaussianMixture(n_components=nComp).fit(fullSparkPixels)
    np.savetxt('fullSparkPixels.csv', fullSparkPixels, delimiter=";")
    return gmm

if __name__=='__main__':
    # gmm = get_gmm_from_all_sparks()

    aux = get_frame_wise_classification([2000, 2005])
    aux = get_pixel_wise_classification([2000, 2005])

    pass