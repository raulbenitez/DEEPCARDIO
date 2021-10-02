<h2 align=center>DEEPCARDIO</h2>
This repository is built as a consequence of the development of the MSc thesis:
<h4>DEEPCARDIO: Detection of calcium events in cardiac cells using deep learning in fluorescence microscopy images</h4>

<!-- abstract -->
<p style="text-align: center">The aim of this thesis is to apply deep learning techniques to calcium imaging sequences of cardiac myocytes for the detection of calcium events. 
For the scope of this thesis, the work is centered in the detection and localization of sparks, which are small spontaneous calcium events. 
The experimental data is obtained by means of fluorescence microscopy.  
There are two different models developed, first the frame-wise model which is a CNN with InceptionV3 architecture, which 
classifies the video sequence frame by frame detecting the presence of sparks. And secondly, the pixel-wise segmentation 
model which is a CNN with an adaptation of the UNet architecture, which provides a segmentation indicating the area of the sparks.  
The original experimental data did not provide an accurate enough labeling of the data for training the models. 
In this thesis it is proposed a relabeling method. However, even though the resulting relabeling is better, is not good 
enough for training the models and extracting the adequate patterns. For this reason, a method for generating realistic 
synthetic data is provided, so that the labeling of the data can be completely controlled.  
The results obtained training the model with synthetic data and testing over real data are very satisfactory. 
The conclusions are that deep learning is a good approach for this problem, however CNN are very sensitive to the typology of noise.  
Both pixel-wise and frame-wise models obtained, have been encapsulated in a dash application. The objective of this
application is to be very intuitive so that it can be used by experts in the field of cardiac physiology.</p>

This repository gathers all code developed throughout the work of this thesis and its parts:
- first data exploration
- relabeling of data and synthetic data generator
- training the models
- dash app

### Project structure

The project is divided in the following directories in order to provide a logical division in the different parts:
- `dash_app/` Contains the code for the dash application which encapsulates both frame-wise and pixel-wise predictors.
- `explore_data/` Contains a jupyter which explores the details of the data. It contains the pre-built for the frame-wise 
and pixel-wise models. There are also some useful scripts for generating movies out of frames.
- `interpretability/` Script for the interpretability technique used.
- `pred/` Directory dedicated to the prediction and performance testing. Its main file is `utils.py` which contains the 
code for the class ``BasePredictor`` and the inheriting classes `FrameWisePredictor` and `PixelWisePredictor`.
When the predictions are performed, a `predictions_db/` is created inside this folder for performance optimization. 
This folder contains both frame-wise model object `frameWiseInceptionV3.h5` and pixel-wise model object `pixelWiseUNet.h5`.
- `synthetic_data/` Script for the generation of synthetic data.
- `train/` Folder dedicated to the training process itself, with multiple useful classes and functions. As well as the 
training jupyters (which were executed on Google Colab).

Finally, the file `deepcardio_utils.py` contains multiple functions and utils used in the other files. And both 
`requirements.txt` and `docker_requirements.txt` define the full requirements of this project for being executed and the 
extra requirements needed for the `tensorflow/tensorflow` docker image to run the dash app, respectively.

<h3 id="h3-image-datasets">Image datasets</h3>
All code in this repository is built assuming a folder ```../_datasets/deepcardio``` relative to the root of the project sparks. 
This folder should contain the datasets on which to train / test. Note that for the dash app the idea is to mount this
folder as a volume in the path `/opt/_datasets/deepcardio`, so that it will be in the appropiate relative path to the scripts.
These datasets should be structured in separate directories, following the structure:
- The name of the directory defines the name of the dataset itself.
- The folder of the dataset should directly contain the images of the experiment in the ch01 and format `.tif`.
- Should the sparks be inventoried, they should be provided either in a `.mat` of `.csv` file with structure: `'x', 'y', 'tIni', 'tFin'`.
- The images should be named in a way that if they are ordered by name they should appear in chronological order.

An example with synthetic images and the appropiate structure can be found at [`_datasets/deepcardio/example_dataset`](../_datasets/deepcardio/example_dataset).
This folder contains the images of the sequence as well as an example of sparks inventary `.csv` file.

### Deepcardio dash app
The classification systems developed have been encapsulated in a dash app so that they can be easily used. 
The deepcardio dash app integrates both the frame-wise and the pixel-wise models, the objective is to provide an insightful 
analysis for a given sequence of fluorescence microscopy images of a cardiac myocyte.

The docker image is accessible at the Docker Hub repo [aleixsacrest/deepcardio-dashapp](https://hub.docker.com/repository/docker/aleixsacrest/deepcardio-dashapp)
and it can directly be used with the `docker run` command. There are a couple of considerations to bear in mind when using 
this docker image:
- First that the port `8050` needs to be exposed from the insight since it is the one used by dash.
- The computations performed by the dash app can become quite heavy. For this reason, it is possible that the app crashes
when not enough memory is allocated. This might depend on the size of the dataset used, but it is recommended to provide
around 4GB.
- The datasets of images used within the app are thought to be accessed through a binded volume. This volume has to be 
mounted to the `../_datasets/deepcardio` relative to the current folder (`DEEPCARDIO/sparks`) which in the docker image 
will correspond to `/opt/_datasets/deepcardio`. More details about the composition of these datasets is provided in a [previous section](#h3-image-datasets).  

Taking all this into account the docker image can be initiated with the following command, where `$path_to_datasets` is 
the path in the local system to the datasets folder.  
`docker run -dp 8050:8050 --memory 4g -v $path_to_datasets:/opt/_datasets --name deepcardio aleixsacrest/deepcardio-dashapp`  

Alternatively, the docker image can be easily built using the [Dockerfile](Dockerfile). To build the image it is only needed 
to clone the repository and execute the following command in the current folder `DEEPCARDIO/sparks`:  
`docker build -t deepcardio-image .`  

