from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from deepcardio_utils import DATASETS_PATH, ImageReader


def load_data(imageId, datasetsPath=DATASETS_PATH, gaussianFilter=False):
    imageReader = ImageReader(imageId=imageId, datasetsPath=datasetsPath)
    images = imageReader.get_full_images(gaussianFilter=gaussianFilter)
    classes = imageReader.get_pixel_wise_classification()

    # Transform targets to keras compatible format
    num_classes = 2
    Y = np_utils.to_categorical(classes, num_classes)

    # preprocess data
    X = images.astype('float32')
    X = X / 255.0

    # Split train / test data
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=1)
    # print(f"Prop of sparks in train dataset: {round(Y_train.sum(axis=0)[1]/Y_train.shape[0]*100, 2)}, "
    #       f"and in validation dataset: {round(Y_valid.sum(axis=0)[1]/Y_valid.shape[0]*100, 2)}")

    return X_train, Y_train, X_valid, Y_valid


if __name__=='__main__':
    X_train, Y_train, X_valid, Y_valid = load_data('2021-01-23_02-52-32_gen_images')

    pass