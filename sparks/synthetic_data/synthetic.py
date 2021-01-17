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


if __name__=='__main__':
    imageReader = ImageReader()
    images = imageReader.get_full_images()
    shp = images[0].shape

    # synthetic no-spark gen
    noisyGen = imageReader.background_noise_images_generator(multichannel=True)
    # synthetic spark
    sparkGen = imageReader.spark_images_generator(multichannel=True)

    N_FRAMES = 2000
    SPARKS_N_FRAMES = (2, 10)
    SPARKS_SIZE_SIGMA = (0.2, 0.5)
    SPARKS_NOISE_SIGMA = (0.3, 2)
    SPARK_PROP = 0.146 / 5.

    timeID = datetime.datetime.now(pytz.timezone('Europe/Madrid')).strftime('%Y-%m-%d_%H-%M-%S')
    GEN_IMAGE_ID = f"{timeID}_gen_images"
    savePath = os.path.join(imageReader.get_datasets_path(), GEN_IMAGE_ID)
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    sparksTill = -1
    sparkCentroid = None
    classes = pd.Series(0, index=list(range(N_FRAMES)))

    for idx in range(N_FRAMES):
        print(f"{idx}/{N_FRAMES}")
        if idx <= sparksTill or np.random.rand() < SPARK_PROP: # spark
            if idx > sparksTill:
                sparksNFrames = np.random.randint(*SPARKS_N_FRAMES)
                sparksTill = idx + (sparksNFrames-1)
                sparkCentroid = np.random.randint(0, shp[0]), np.random.randint(0, shp[1])
            sizeSigma = np.random.uniform(*SPARKS_SIZE_SIGMA)
            noiseSigma = np.random.uniform(*SPARKS_NOISE_SIGMA)
            im = sparkGen(sparkCentroid, sparkSigma=sizeSigma, noiseSigma=noiseSigma)
            classes[idx] = 1
        else: # no-spark
            im = next(noisyGen)

        fileName = f"im{idx}.tif"
        cv2.imwrite(os.path.join(savePath, fileName), im)

    classes.to_csv(os.path.join(savePath, 'class.csv'))

    # plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()

    # spark 0.2 - 0.5
    # noise 0.3 - 2

    pass