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
    videoSize = 4

    images = imageReader.get_images_names()
    frame = cv2.imread(os.path.join(imagePath, images[0]))
    height, width, layers = frame.shape

    sparksDF = imageReader.get_sparks_df()
    F0 = None

    videoWidth = width * videoSize
    videoHeight = height * videoSize
    video = cv2.VideoWriter(video_name, 0, 30, (videoWidth, videoHeight))

    for i, image in enumerate(images):
        im = (cv2.imread(os.path.join(imagePath, image))*1.5).astype('uint8')
        if normalize:
            if F0 is None:
                F0, _, _ = get_f0_and_cellmask()
            im = get_normalized_frame(im, F0)
        im = imageReader.get_image_with_circled_sparks(i, im)
        video.write(cv2.resize(im, (videoWidth, videoHeight)))

    cv2.destroyAllWindows()
    video.release()
