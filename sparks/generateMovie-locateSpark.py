import cv2
import os
import sys
import pandas as pd

from scipy.io import loadmat

DATASETS_PATH = '../_datasets/deepcardio'
IMAGE_ID = '170215_RyR-GFP30_RO_01_Serie2_SPARKS-calcium'
IMAGE_FOLDER = os.path.join(DATASETS_PATH, IMAGE_ID)
IMAGE_FILE_TEMPLATE = '170215_RyR-GFP30_RO_01_Serie2_z1{}_ch01.tif'
MAT_PATH = '170215_RyR-GFP30_01_Serie2_Sparks.mat'

def get_image_path(idx):
    return os.path.join(IMAGE_FOLDER, IMAGE_FILE_TEMPLATE.format(str(idx).zfill(4)))

def get_image_array(idx):
    imagePath = get_image_path(idx)
    return np.array(Image.open(imagePath))

def get_spark_location(sparksDF, idx):
    candidates = sparksDF.loc[(sparksDF.loc[:,'tIni'] <= idx) & (sparksDF.loc[:, 'tFin'] > idx), :]
    return candidates.loc[:, ['x', 'y']]

if __name__=='__main__':
    video_name = f'{IMAGE_ID}.avi'

    images = sorted([img for img in os.listdir(IMAGE_FOLDER) if img.endswith(".tif")])
    frame = cv2.imread(os.path.join(IMAGE_FOLDER, images[0]))
    height, width, layers = frame.shape

    mat = loadmat(os.path.join(DATASETS_PATH, MAT_PATH))['xytspark']
    sparksDF = pd.DataFrame(mat, columns=['x','y','tIni','tFin'])

    video = cv2.VideoWriter(video_name, 0, 30, (width,height))

    for i, image in enumerate(images):
        im = cv2.imread(os.path.join(IMAGE_FOLDER, image))
        sparkLocations = get_spark_location(sparksDF, i)
        for i, sparkLocation in sparkLocations.iterrows():
            color = int(im.max()*2)
            cv2.circle(im, (sparkLocation['x'], sparkLocation['y']), 20, color, thickness=1, lineType=8, shift=0)
        video.write(im)

    cv2.destroyAllWindows()
    video.release()
