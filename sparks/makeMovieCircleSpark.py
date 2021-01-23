import cv2
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from deepcardio_utils import *

if __name__=='__main__':
    imageId = sys.argv[1]
    imageReader = ImageReader(imageId)
    imagePath = imageReader.get_image_folder()
    normalize = '--normalize' in sys.argv[1:]
    video_name = f"{imageReader.get_datasets_path()}/{imageId}{'_norm' if normalize else ''}_circ.avi"

    images = imageReader.get_images_names()
    frame = cv2.imread(os.path.join(imagePath, images[0]))
    height, width, layers = frame.shape

    sparksDF = imageReader.get_sparks_df()
    F0 = None

    video = cv2.VideoWriter(video_name, 0, 30, (width,height))

    for i, image in enumerate(images):
        im = cv2.imread(os.path.join(imagePath, image))
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
