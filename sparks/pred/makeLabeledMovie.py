import keras
import sys
import cv2
import numpy as np

from deepcardio_utils import ImageReader, get_spark_location

if __name__=='__main__':
    args = sys.argv[1:]
    imageReader = ImageReader()
    assert '--model' in args, 'USAGE: --model path_to_model.h5 [--imageid imageid]'
    pathToModel = args[args.index('--model')+1]

    X = imageReader.get_full_padded_images()
    # preprocess data
    X = X.astype('float32')
    X = X / 255.0

    Y = imageReader.get_frame_wise_classification(classesFromFile=True)
    sparksDF = imageReader.get_sparks_df()

    inceptionv3 = keras.applications.InceptionV3(include_top=True, weights=None, classes=2, input_shape=X[0].shape)
    inceptionv3.load_weights(pathToModel)
    Y_pred = inceptionv3.predict(X)
    Y_pred = Y_pred.round()

    video_name = imageReader.get_image_id() + '_pred_labels.avi'
    height, width, _ = imageReader.get_shape()
    video = cv2.VideoWriter(video_name, 0, 30, (width, height))

    for i, im in enumerate(X):
        c = Y_pred[i]
        image = (im[:40] * 255.0).astype(np.uint8)
        if c.argmax():
            cv2.putText(image, 'spark', (15, 15), cv2.FONT_HERSHEY_SIMPLEX, .25, (255,255,255))
        if Y[i]:
            sparkLocations = get_spark_location(sparksDF, i)
            for i, sparkLocation in sparkLocations.iterrows():
                color = int(image.max() * 2)
                cv2.circle(image, (sparkLocation['x'], sparkLocation['y']), 20, color, thickness=1, lineType=8, shift=0)
        video.write(image)

    cv2.destroyAllWindows()
    video.release()