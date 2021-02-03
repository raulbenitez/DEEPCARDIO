import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import datetime
import os
import pytz
from scipy.stats import norm
from skimage.filters import gaussian
from math import sqrt, pi, e

from deepcardio_utils import ImageReader, get_mask

N_FRAMES = 2000
SPARKS_N_FRAMES = (2, 10)
SPARKS_SIZE_SIGMA = (0.2, 0.5)
SPARKS_NOISE_SIGMA = (0.3, 2)
SPARK_PROP = 0.146 / 10.

VERBOSE_SPARKS_FILE = 'verboseSparks.csv'


if __name__=='__main__':
    imageReader = ImageReader()
    images = imageReader.get_full_images()
    shp = images[0].shape

    # cell mask
    cellMask = imageReader.get_cellmask(images)

    # synthetic no-spark gen
    noisyGen = imageReader.background_noise_images_generator(multichannel=True)
    # synthetic spark
    sparkGen = imageReader.spark_images_generator(multichannel=True)

    timeID = datetime.datetime.now(pytz.timezone('Europe/Madrid')).strftime('%Y-%m-%d_%H-%M-%S')
    GEN_IMAGE_ID = f"{timeID}_synthetic"
    savePath = os.path.join(imageReader.get_datasets_path(), GEN_IMAGE_ID)
    print(savePath)
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    sparksTill = -1
    sparkCentroid = None
    classes = pd.Series(0, index=list(range(N_FRAMES)))
    verboseSparksList = []
    sparksCentroidsList = []
    sparkIdx = -1

    for idx in range(N_FRAMES):
        print(f"{idx}/{N_FRAMES}")
        if idx <= sparksTill or np.random.rand() < SPARK_PROP: # spark
            if idx > sparksTill:
                sparkIdx += 1
                sparksNFrames = np.random.randint(*SPARKS_N_FRAMES)
                sparksTill = idx + (sparksNFrames-1)
                sparkCentroid = np.random.randint(0, shp[0]), np.random.randint(0, shp[1])
                while not cellMask[sparkCentroid[0], sparkCentroid[1]]:
                    sparkCentroid = np.random.randint(0, shp[0]), np.random.randint(0, shp[1])
                sparksCentroidsList.append(sparkCentroid)
            sizeSigma = np.random.uniform(*SPARKS_SIZE_SIGMA)
            noiseSigma = np.random.uniform(*SPARKS_NOISE_SIGMA)
            im, sparkMaxValue = sparkGen(sparkCentroid, sparkSigma=sizeSigma, noiseSigma=noiseSigma)
            classes[idx] = 1
            verboseSparksList.append((sparkIdx, sizeSigma, noiseSigma, sparkMaxValue))
        else: # no-spark
            im = next(noisyGen)

        fileName = f"im{str(idx).zfill(5)}.tif"
        cv2.imwrite(os.path.join(savePath, fileName), im)

    # store classes
    classes.to_csv(os.path.join(savePath, 'class.csv'), header=False, index=False, sep=';')
    verboseSparksDF = pd.DataFrame(verboseSparksList, columns=['sparkIdx', 'size_sigma', 'noise_sigma', 'spark_max'])
    verboseSparksDF.to_csv(os.path.join(savePath, VERBOSE_SPARKS_FILE), index=False, sep=';')

    # store compact npy file
    genImageReader = ImageReader(imageId=GEN_IMAGE_ID)
    genImageReader.create_npy_file()

    fullSparkDF = classes.index[classes==1].to_series().reset_index(drop=True)
    iniFinDF = fullSparkDF.groupby(verboseSparksDF.loc[:, 'sparkIdx']).agg(['min', 'max'])
    iniFinDF.columns = ['tIni', 'tFin']
    sparksDF = pd.concat([pd.DataFrame(sparksCentroidsList, columns=['y', 'x']), iniFinDF], axis=1).loc[:, ['x', 'y', 'tIni', 'tFin']]
    sparksDF.to_csv(os.path.join(savePath, f"{timeID}_xyt.csv"), index=False, sep=",")

    # plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()

    # spark 0.2 - 0.5
    # noise 0.3 - 2

    pass