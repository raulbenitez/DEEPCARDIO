import cv2
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from deepcardio_utils import ImageReader

MX = 120
MY = 800
BASE_PATH = '../_datasets/deepcardio/Blas_data'

if __name__=='__main__':
    imageReader = ImageReader('170215_RyR-GFP30_RO_01_Serie2_SPARKS-calcium')
    imagesShape = imageReader.get_full_images()[0].shape

    folder = sys.argv[1]
    outputFolderPath = os.path.join('../_datasets/deepcardio', f"Blas_data_{folder}")
    if not os.path.exists(outputFolderPath):
        os.makedirs(outputFolderPath)

    for f in sorted(os.listdir(os.path.join(BASE_PATH, folder))):
        s = pd.read_csv(os.path.join(BASE_PATH, folder, f), header=None, sep='       ')
        im = s.to_numpy().reshape(MX, MY, order='F')
        res_im = np.zeros(imagesShape)
        res_im[:, :, 2] = cv2.resize(im, dsize=(imagesShape[1], imagesShape[0]))
        res_im = (res_im*255.).astype(np.uint8)

        myDpi = 96
        plt.figure(figsize=(imagesShape[1]/myDpi, imagesShape[0]/myDpi), dpi=myDpi)
        plt.imshow(cv2.cvtColor(res_im, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.savefig(os.path.join(outputFolderPath, f.replace('.data', '.tif')), bbox_inches='tight', pad_inches=0, dpi=myDpi/0.75)
        # plt.show()
        plt.close()

