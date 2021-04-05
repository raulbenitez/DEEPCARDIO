# -*- coding: utf-8 -*-

# We start with the import of standard ML librairies
import base64
import io
import os

import pandas as pd
import numpy as np
import math
from PIL import Image

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

# We add all Plotly and Dash necessary librairies
import plotly.graph_objects as go

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash.dependencies import Input, Output, State
import plotly.express as px

from deepcardio_utils import ImageReader, get_plottable_image
from pred.utils import FrameWisePredictor, PixelWisePredictor

GLOB_IMG_READER = ImageReader('170215_RyR-GFP30_RO_01_Serie2_SPARKS-calcium')
GLOB_DICT = {}


###############################################################################

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# The page structure will be:
#    Features Importance Chart
#    <H4> Feature #1 name
#    Slider to update Feature #1 value
#    <H4> Feature #2 name
#    Slider to update Feature #2 value
#    <H4> Feature #3 name
#    Slider to update Feature #3 value
#    <H2> Updated Prediction
#    Callback fuction with Sliders values as inputs and Prediction as Output


# images path

imagesBasePathInput = dbc.Input(id='images-base-path', placeholder='Images base path', value='../_datasets/deepcardio')

baseImagePath = '../_datasets/deepcardio'
imagesIdsList = sorted([d for d in os.listdir(baseImagePath) if os.path.isdir(os.path.join(baseImagePath, d))])
imagesSelector = dbc.InputGroup(
    [
        dbc.InputGroupAddon("Images id", addon_type="prepend"),
        dbc.Select(id="images-id-selector", options=[{"label": t, "value": t} for t in imagesIdsList],
                   value='170215_RyR-GFP30_RO_01_Serie2_SPARKS-calcium')
    ])

imagesFilePath = html.Div([dbc.Row(html.H3('Images path')),
                           dbc.Row([dbc.Col(imagesBasePathInput), dbc.Col(imagesSelector)])])

#################
# model selection

modelsBasePathInput = dbc.Input(id='models-base-path', placeholder='Models base path', value='pred/')
trainedModels = [f for f in os.listdir('pred/') if f.endswith('.h5')]
frameWiseModelSelection = dbc.InputGroup(
    [
        dbc.InputGroupAddon("Frame-Wise model", addon_type="prepend"),
        dbc.Select(id="frame-wise-model", options=[{"label": t, "value": t} for t in trainedModels], value=None)
    ])

pixelWiseModelSelection = dbc.InputGroup(
    [
        dbc.InputGroupAddon("Pixel-Wise model", addon_type="prepend"),
        dbc.Select(id="pixel-wise-model", options=[{"label": t, "value": t} for t in trainedModels], value=None)
    ])

modelSelection = html.Div([dbc.Row(html.H3('Model selection')),
                           dbc.Row([dbc.Col(modelsBasePathInput),
                                    dbc.Col(frameWiseModelSelection),
                                    dbc.Col(pixelWiseModelSelection),
                                    dbc.Col(dbc.Button(id='button-load-models', children='Load models'), md=1)]),
                           html.Div(id='model-selection-output')])

##################
# frame navigation buttons

navigationButtons = html.Div(dbc.Row([
    dbc.Col(dbc.Input(id='input-frame-idx', type='number', value=0, min=0), md=1),
    dbc.Col([dbc.Row(html.Div(children='Frame nav')),
             dbc.Row([
                 dbc.Button(id='button-left-frame', children='<', className='mr-1'),
                 dbc.Button(id='button-right-frame', children='>', className='mr-1')
             ])]),
    dbc.Col([dbc.Row(html.Div(children='Spark nav')),
             dbc.Row([
                 dbc.Button(id='spark-left-frame', n_clicks=0, color='info', children='<', className='mr-1'),
                 dbc.Button(id='spark-right-frame', n_clicks=0, color='info', children='>', className='mr-1')
             ])])
]))

##################

# We apply basic HTML formatting to the layout
app.layout = dbc.Container(style={'margin': 'auto', 'font-family': 'Verdana'},
                           fluid=True,

                           children=[

                               # DEEPCARDIO
                               html.H1(children="DEEPCARDIO"),
                               html.Hr(),

                               imagesFilePath,
                               html.Hr(),
                               modelSelection,

                               html.H4(children='Frame selector'),

                               navigationButtons,
                               dcc.Graph(id='show-img'),
                               html.Div(id='show-coses'),



                               dcc.Upload(
                                   id='upload-image',
                                   children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                                   style={
                                       'width': '100%',
                                       'height': '60px',
                                       'lineHeight': '60px',
                                       'borderWidth': '1px',
                                       'borderStyle': 'dashed',
                                       'borderRadius': '5px',
                                       'textAlign': 'center',
                                       'margin': '10px'
                                   },
                                   # Allow multiple files to be uploaded
                                   multiple=True
                               ),
                               html.Div(id='show-img2')
                           ])


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    im = np.asarray(Image.open(io.BytesIO(decoded)))[:, :, :3]
    fig = px.imshow(im)
    # return dcc.Graph(fig)

    return html.Div([
        html.H5(filename),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
        html.Div('Raw Content'),
        html.Pre(contents[:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])


@app.callback(Output('show-img2', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


@app.callback(Output('images-id-selector', 'options'),
              Input('images-base-path', 'value'))
def image_base_path_selected (bPath):
    aux = sorted([d for d in os.listdir(bPath) if os.path.isdir(os.path.join(bPath, d))])
    return [{"label": t, "value": t} for t in aux]


@app.callback(Output('pixel-wise-model', 'options'), Output('frame-wise-model', 'options'),
              Input('models-base-path', 'value'))
def image_base_path_selected (bPath):
    aux = sorted([f for f in os.listdir(bPath) if f.endswith('.h5')])
    ret = [{"label": t, "value": t} for t in aux]
    return ret, ret


@app.callback(Output('input-frame-idx', 'max'),
              Input('images-id-selector', 'value'), State('images-base-path', 'value'))
def image_id_selected(imageId, bPath):
    GLOB_DICT['imageReader'] = ImageReader(imageId=imageId, datasetsPath=bPath)
    return len(GLOB_DICT['imageReader'].get_images_names())


@app.callback(Output('show-img', 'figure'),
              Input('input-frame-idx', 'value'),
              State('spark-left-frame', 'n_clicks'), State('spark-right-frame', 'n_clicks'))
def frame_idx_selected(inputValue, nClicksLeft, nClicksRight):
    if not 'nClicksLeft' in GLOB_DICT:
        GLOB_DICT['nClicksLeft'] = nClicksLeft
        GLOB_DICT['nClicksRight'] = nClicksRight
    imageReader = GLOB_DICT.get('imageReader')
    images = imageReader.get_full_images()
    img = get_plottable_image(images[inputValue])
    fig = px.imshow(img, title=f"idx {inputValue}")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


@app.callback(Output('input-frame-idx', 'value'),
              Input('spark-left-frame', 'n_clicks'), Input('spark-right-frame', 'n_clicks'),
              State('input-frame-idx', 'value'))
def previous_spark(nClicksLeft, nClicksRight, inputFrame):
    if not 'nClicksLeft' in GLOB_DICT:
        return inputFrame
    if nClicksLeft > GLOB_DICT['nClicksLeft']:
        click = 'left'
        GLOB_DICT['nClicksLeft'] = nClicksLeft
    else:
        click = 'right'
        GLOB_DICT['nClicksRight'] = nClicksRight
    return inputFrame + (-1 if click=='left' else 1)


@app.callback(Output('model-selection-output', 'children'),
              Input('button-load-models', 'n_clicks'),
              State('frame-wise-model', 'value'), State('pixel-wise-model', 'value'), State('models-base-path', 'value'))
def previous_spark(nClicksLoadModels, frameWiseModel, pixelWiseModel, modelsBasePath):
    if frameWiseModel is None or pixelWiseModel is None:
        return 'Select both frame-wise and pixel-wise models'

    imageReader = GLOB_DICT['imageReader']
    framePredictor = FrameWisePredictor(imageId=imageReader.get_image_id(),
                                        model=os.path.join(modelsBasePath, frameWiseModel))
    pixelPredictor = PixelWisePredictor(imageId=imageReader.get_image_id(),
                                        model=os.path.join(modelsBasePath, pixelWiseModel))

    return f"Successfully loades models: frame-wise {frameWiseModel} and pixel-wise {pixelWiseModel}"

if __name__ == "__main__":
   app.run_server()
