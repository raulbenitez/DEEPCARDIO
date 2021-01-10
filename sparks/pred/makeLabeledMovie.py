import keras
import sys
import cv2
import numpy as np

from deepcardio_utils import ImageReader

if __name__=='__main__':
    args = sys.argv[1:]
    imageReader = ImageReader()
    assert '--model' in args, 'USAGE: --model path_to_model.h5 [--imageid imageid]'
    pathToModel = args[args.index('--model')+1]

    X = imageReader.get_full_padded_images()
    # preprocess data
    X = X.astype('float32')
    X = X / 255.0

    inceptionv3 = keras.applications.InceptionV3(include_top=True, weights=None, classes=2, input_shape=X[0].shape)
    inceptionv3.load_weights(pathToModel)
    Y_pred = inceptionv3.predict(X)
    Y_pred = Y_pred.round()

    video_name = imageReader.get_image_id() + '_pred_labels.avi'
    height, width, _ = imageReader.get_shape()
    video = cv2.VideoWriter(video_name, 0, 30, (width, height))

    for im, c in zip(X, Y_pred):
        image = (im[:40] * 255.0).astype(np.uint8)
        if c.argmax():
            cv2.putText(image, 'spark', (15, 15), cv2.FONT_HERSHEY_SIMPLEX, .25, (255,255,255))
        video.write(image)

    cv2.destroyAllWindows()
    video.release()