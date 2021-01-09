_out - Output images and videos.
deepcardio_utils.py - Functions and utils (used in the other files).  
explore.ipynb - Jupyter where different strategies are explored.  

fullSparkPixels.czsv - It is the concatenation of all pixels in all frames of 170215_RyR-GFP30_RO_01_Serie2_SPARKS-calcium movie. The purpose was to build a unique gmm with all pixels. Not very useful at the end.  

### train/  
inception.py - Build and train inceptionv3 neural networks.
inception_test.ipynb - Display different metrics for each trained model (confusion matrix, recalls...).
imagenet_inception.ipynb - Transfer learning with imagenet.

### Generate movies from frames in ../_datasets/deepcardio/170215_RyR-GFP30_RO_01_Serie2_SPARKS-calcium  
makeMovie.py - Regular movie.  
makeMovieCircleSpark.py - Drawing a circumference over the sparks.  
makeMovieGMMSpark.py - Circle spark area, and when it is visible it is highlighted in green using gmms over the filtered image. The video i the concatenation of the original image, the image with circle+highlight and the filtered image.
