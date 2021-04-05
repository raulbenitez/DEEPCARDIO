import numpy as np
import matplotlib.pyplot as plt
import cv2
from deepcardio_utils import ImageReader, plot_cell, get_rolled_images
from pred.utils import FrameWisePredictor, PixelWisePredictor, get_clustered_pred_sparks


def plot_cells(*cells, titles=None):
    fig, ax = plt.subplots(len(cells), 1, figsize=(20, 4*len(cells)))

    for i, img in enumerate(cells):
        ax[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[i].axis('off')
        if titles is not None:
            ax[i].set_title(titles[i])

    plt.show(bbox_inches='tight', pad_inches=0)


def get_rolled_images_(images, rollsize=3):
    return np.array([np.ma.average(images[max(i-rollsize+1, 0):i+1], axis=0) for i in range(images.shape[0])])


if __name__=='__main__':
    framePredictor = FrameWisePredictor(model='pred/i3_comp_2021-01-23_02-02-14_gen_images_ep10_b64_ler0.01_ite0.h5')
    pixelPredictor = PixelWisePredictor(model='pred/pixelWiseUNet.h5')

    imageReader = framePredictor.get_image_reader()
    Y_pred = framePredictor.predict()

    get_clustered_pred_sparks(framePredictor, pixelPredictor)

    # classes = imageReader.get_frame_wise_class_gmm()
    # pixelClass = imageReader.get_pixel_wise_classification()
    #
    # idx = 17
    # im = images[idx].copy()
    # im[:, :, 1] = pixelClass[idx]*50
    # plot_cell(im)

    # generate synthetic Leif data
    confsDF, detSparksDF = imageReader.get_spark_simple_data()

    backGen = imageReader.background_noise_images_generator(multichannel=True)
    im = next(backGen)
    plot_cell(im, title='sense spark')
    sparGen = imageReader.spark_images_generator(multichannel=True, saltAndPepper=True)
    for pepperThreshold in np.arange(0.05, 0.51, 0.05):
        im = sparGen(tuple(detSparksDF.loc[0, ['Ypix', 'Xpix']].astype(int)), pepperThreshold=pepperThreshold)[0]
        plot_cell(im, title=f"spark amb salt and pepper amb pepperThreshold={pepperThreshold}")

    pass