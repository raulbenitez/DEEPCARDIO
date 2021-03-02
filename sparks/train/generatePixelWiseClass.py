from deepcardio_utils import ImageReader

if __name__=='__main__':
    imageReader = ImageReader()
    classes = imageReader.get_pixel_wise_classification()