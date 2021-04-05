from deepcardio_utils import ImageReader, plot_cell

if __name__=='__main__':
    imageReader = ImageReader(rollingSize=3)
    images = imageReader.get_full_images()
    classes = imageReader.get_frame_wise_class_gmm()
    pixelClass = imageReader.get_pixel_wise_classification()

    idx = 3
    im = images[idx].copy()
    im[:, :, 1] = pixelClass[idx]*50
    plot_cell(im)