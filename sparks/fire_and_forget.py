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
    framePredictor.predict()
    get_clustered_pred_sparks(framePredictor, pixelPredictor)

    imageReader = framePredictor.get_image_reader()
    Y_pred = framePredictor.predict()

    images = imageReader.get_full_images()

    # plot occurrence heatmap
    sparksDFAsList, sparkPredMasksL = get_clustered_pred_sparks(framePredictor, pixelPredictor)
    sparkPredMasks = np.array(sparkPredMasksL)
    rawOccurrenceHeatmap = sparkPredMasks.astype('float').sum(axis=0)
    occurrenceHeatmap = (rawOccurrenceHeatmap / rawOccurrenceHeatmap.max() * 255).astype('uint8')
    plt.figure(figsize=(20, 5))
    plt.imshow(occurrenceHeatmap, cmap='jet')
    plt.axis('off')
    plt.show()

    # plot intensity heatmap
    fullSparksOccurrenceMask = rawOccurrenceHeatmap.astype(bool)
    fullPixelsSum = imageReader.get_full_images()[:, :, :, 2].sum(axis=0)
    rawIntensityHeatmap = np.where(fullSparksOccurrenceMask, fullPixelsSum, 0).astype('float')
    intensityHeatmap = (rawIntensityHeatmap / rawIntensityHeatmap.max() * 255).astype('uint8')
    plt.figure(figsize=(20, 5))
    plt.imshow(intensityHeatmap, cmap='jet')
    plt.axis('off')
    plt.show()

    m = sparkPredMasksL[0]
    ints = []
    for ii in range(45, 60):
        ints.append(images[ii][:, :, 2][m].mean())
    plt.figure(figsize=(20, 5))
    plt.plot(ints)
    plt.show()

    # plot spark images
    images = imageReader.get_full_images()
    classes = imageReader.get_frame_wise_class_gmm()
    idx = np.random.choice(np.arange(classes.shape[0])[classes == 1], 1)[0]
    plot_cell(images[idx])

    ir1 = ImageReader(imageId='2021-01-23_02-02-14_gen_images')
    ir2 = ImageReader(imageId='2021-03-24_00-35-00_TLeif__synthetic')

    images1 = ir1.get_full_images()
    plot_cell(images1[40])
    plot_cell(cv2.resize(images1[40], (images1.shape[2], 75)))

    images2 = ir2.get_full_images()
    plot_cell(images2[1])
    plot_cell(cv2.resize(images2[1], (images1.shape[2], images1.shape[1])))

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