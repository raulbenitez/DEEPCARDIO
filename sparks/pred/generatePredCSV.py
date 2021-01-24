import os
import sys

import cv2
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    Y = imageReader.get_frame_wise_class_gmm(classesFromFile=True)
    # sparksDF = imageReader.get_sparks_df()

    inceptionv3 = keras.applications.InceptionV3(include_top=True, weights=None, classes=2, input_shape=X[0].shape)
    inceptionv3.load_weights(pathToModel)
    Y_pred = inceptionv3.predict(X)
    Y_pred = Y_pred.round().argmax(axis=-1)

    pd.Series(Y_pred).to_csv(os.path.join(imageReader.get_image_folder(), 'pred_class.csv'), header=False, index=False)

    # preditction statistics
    falsePositives = ~Y.astype(bool) & Y_pred.astype(bool)
    falseNegatives = Y.astype(bool) & ~Y_pred.astype(bool)
    truePositives = Y.astype(bool) & Y_pred.astype(bool)
    trueNegatives = ~Y.astype(bool) & ~Y_pred.astype(bool)

    precisionT = round(truePositives.sum() / Y_pred.sum(), 4)
    precisionF = round(trueNegatives.sum() / (len(Y_pred)-Y_pred.sum()), 4)
    recallT = round(truePositives.sum() / Y.sum(), 4)
    recallF = round(trueNegatives.sum() / (len(Y)-Y.sum()), 4)
    print(f"Precision (spark / no-spark): {(precisionT, precisionF)}, recall (spark / no-spark): {(recallT, recallF)}")

    # idxs = np.array(range(len(Y)))
    # idx = np.random.choice(idxs[falsePositives], 1)[0]
    # images = imageReader.get_full_images()
    # plt.figure(figsize=(20,3))
    # plt.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()
