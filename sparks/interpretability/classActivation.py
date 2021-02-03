import os
import sys

import cv2
# from tensorflow import keras
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
import matplotlib.cm as cm

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

    inceptionv3 = keras.applications.InceptionV3(include_top=True, weights=None, classes=2, input_shape=X[0].shape)
    inceptionv3.load_weights(pathToModel)

    # Y_pred = inceptionv3.predict(X)
    # Y_pred = Y_pred.round().argmax(axis=-1)
    # Y_pred = (Y_pred[:, 1] > 0.95).astype(int)
    Y_pred = pd.read_csv(os.path.join(imageReader.get_image_folder(), 'pred_class.csv'), header=None, squeeze=True)

    falsePositives = ~Y.astype(bool) & Y_pred.astype(bool)
    falseNegatives = Y.astype(bool) & ~Y_pred.astype(bool)
    truePositives = Y.astype(bool) & Y_pred.astype(bool)
    trueNegatives = ~Y.astype(bool) & ~Y_pred.astype(bool)

    idxs = np.array(range(len(Y)))
    images = imageReader.get_full_images()

    idx = np.random.choice(idxs[truePositives], 1)[0]
    plt.figure(figsize=(20,3))
    plt.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    img_array = np.expand_dims(X[idx], axis=0)

    # interpretability DL thesis

    import numpy as np
    from keras.models import Model, load_model
    from keras.applications.inception_v3 import preprocess_input

    weights = inceptionv3.get_layer('predictions').get_weights()[0]
    extractor = Model(inputs=inceptionv3.inputs, outputs=inceptionv3.get_layer('mixed10').output)
    class_to_explain = 1
    # img_array = preprocess_input(img_array)
    fm = extractor.predict(img_array)
    cm = np.dot(fm[0], weights[:, class_to_explain])
    cm = cv2.resize(cm, ((X[idx].shape[1], X[idx].shape[0])))
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    ax[0].imshow(cv2.cvtColor(X[idx], cv2.COLOR_BGR2RGB))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('NT image')
    ax[1].imshow(cv2.cvtColor(X[idx], cv2.COLOR_BGR2RGB), alpha=0.5)
    ax[1].imshow(cm, cmap='jet', alpha=0.5)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('Class Activation Map')
    plt.show()

    # https://keras.io/examples/vision/grad_cam/

    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    classifier_layer_names = [
        "avg_pool",
        "predictions",
    ]
    last_conv_layer = inceptionv3.get_layer('mixed10')
    last_conv_layer_model = keras.Model(inceptionv3.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = inceptionv3.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, tf.cast(top_pred_index, tf.int32)]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    plt.matshow(heatmap)
    plt.show()

    img = images[idx]
    # We rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    plt.figure(figsize=(10,3))
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()

    # # Save the superimposed image
    # save_path = "classActivation.jpg"
    # superimposed_img.save(save_path)


    # https://valentinaalto.medium.com/class-activation-maps-in-deep-learning-14101e2ec7e1

    spark = inceptionv3.output[:, np.argmax(inceptionv3.predict(X[idx:idx+1])[0])]
    last_conv_layer = inceptionv3.get_layer('mixed10')

    grads = K.gradients(spark, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([inceptionv3.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([X[idx:idx+1]])
    for i in range(pooled_grads.shape[0]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)
    plt.show()

    img = X[idx]
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    merged = heatmap * 0.4 + img
    plt.imshow(merged)
    plt.show()
