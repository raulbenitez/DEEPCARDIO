import numpy as np
import matplotlib.pyplot as plt
import cv2
from deepcardio_utils import ImageReader, plot_cell


def plot_cells(*cells, titles=None):
    fig, ax = plt.subplots(len(cells), 1, figsize=(20, 4*len(cells)))

    for i, img in enumerate(cells):
        ax[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[i].axis('off')
        if titles:
            ax[i].set_title(titles[i])

    plt.show(bbox_inches='tight', pad_inches=0)


if __name__=='__main__':
    imageReader = ImageReader()
    images = imageReader.get_full_images()

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