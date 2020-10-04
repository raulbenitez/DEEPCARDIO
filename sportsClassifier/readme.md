This example has been created following the tutorial in https://www.pyimagesearch.com/2019/07/15/video-classification-with-keras-and-deep-learning/.
It is used the predefined model ResNet50 with pre-loaded weights from imagenet and replacing its head with a custom model for our dataset, this last part will be the one fie-tuned.

- Image dataset: https://github.com/jurjsorinliviu/Sports-Type-Classifier
- Example with reduced set of classes: weight_lifting, football and ice_hockey.
- Preprocess images: swap color chanels and resize to 224×224px.
- Data agumentation *
- Mean substraction *
- ResNet50 with pre-loaded imagenet weights + custom head model for classification.
- CNN frame by frame
- Prediction flickering reduced with rolling prediction averaging.
