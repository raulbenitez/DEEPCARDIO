All code in this repository is built assuming a folder ```../_datasets/deepcardio``` relative to the root of the project. This folder should contain the datasets on which to train / test.


**deepcardio_utils.py** - Functions and utils (used in the other files).  

### explore_data/
There is a jupyter which explorates the details of the data. It contains the pre-built for the frame-wise and pixel-wise models.  
There are also some useful scripts for generating movies out of frames.

### train/
Folder dedicated to the training process itself, with multiple useful classes and functions. As well as the training jupyters (which were executed on Google Colab).

### synthetic_data/
Script for the generation of synthetic data.

### pred/
Folder dedicated to the prediction and performance testing.

### interpretability/
Script for the interpretability technique used.

### dash_app/
Code for the dash app.
