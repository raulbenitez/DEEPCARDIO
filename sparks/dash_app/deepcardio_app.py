# -*- coding: utf-8 -*-

# We start with the import of standard ML librairies
import base64
import io
import os
import time

import dash_table
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
from pred.utils import FrameWisePredictor, PixelWisePredictor, get_clustered_pred_sparks, get_intensity_heatmap

GLOB_IMG_READER = ImageReader('170215_RyR-GFP30_RO_01_Serie2_SPARKS-calcium')
GLOB_DICT = {}


###############################################################################

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP, "assets/custom-style.css"])


# Modal
with open("dash_app/deepcardioapp.md", "r") as f:
    howto_md = f.read()

modal_overlay = dbc.Modal(
    [
        dbc.ModalBody(html.Div([dcc.Markdown(howto_md)], id="howto-md")),
        dbc.ModalFooter(dbc.Button("Close", id="howto-close", className="howto-bn")),
    ],
    id="modal",
    size="lg",
)


button_howto = dbc.Button(
    "Learn more",
    id="howto-open",
    outline=True,
    color="info",
    # Turn off lowercase transformation for class .button in stylesheet
    style={"textTransform": "none"},
)

button_github = dbc.Button(
    "View Code on github",
    outline=True,
    color="primary",
    # href="https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-image-segmentation",
    href="https://github.com/raulbenitez/DEEPCARDIO/tree/master/sparks",
    id="gh-link",
    style={"text-transform": "none"},
)


########
# Header
header = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(html.Img(id="logo", src=app.get_asset_url("dash-logo-new.png"), height="30px"), md="auto"),
                    dbc.Col(
                        [html.Div([html.H3("DEEPCARDIO"), html.P("Deep learning for calcium imaging on heart cells")], id="app-title")],
                        md=True,
                        align="center",
                    ),
                ], align="center"
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.NavbarToggler(id="navbar-toggler"),
                            dbc.Collapse(
                                dbc.Nav(
                                    [
                                        dbc.NavItem(button_howto),
                                        dbc.NavItem(button_github),
                                    ],
                                    navbar=True,
                                ),
                                id="navbar-collapse",
                                navbar=True,
                            ),
                            modal_overlay,
                        ],
                        md=2,
                    ),
                ],
                align="center",
            ),
        ], fluid=True,
    ),
    dark=True,
    color="dark",
    sticky="top",
)


# images path

imagesBasePathInput = dbc.InputGroup([
    dbc.InputGroupAddon("Base path", addon_type="prepend"),
    dbc.Input(id='images-base-path', placeholder='Images base path', value='../_datasets/deepcardio')
    ])

baseImagePath = '../_datasets/deepcardio'
imagesIdsList = sorted([d for d in os.listdir(baseImagePath) if os.path.isdir(os.path.join(baseImagePath, d))])
imagesSelector = dbc.InputGroup(
    [
        dbc.InputGroupAddon("Images id", addon_type="prepend"),
        dbc.Select(id="images-id-selector", options=[{"label": t, "value": t} for t in imagesIdsList],
                   value='170215_RyR-GFP30_RO_01_Serie2_SPARKS-calcium')
    ])

imagesFilePathCard = dbc.Col([
    dbc.Card(id='image-path-card', children=[
        dbc.CardHeader("Images path"),
        dbc.CardBody([dbc.Row(imagesBasePathInput), dbc.Row(imagesSelector)])
    ])
], md=6)

#################
# model selection

modelsBasePathInput = dbc.InputGroup(
    [
        dbc.InputGroupAddon('Base path', addon_type='prepend'),
        dbc.Input(id='models-base-path', placeholder='Models base path', value='pred/')
    ])

trainedModels = [f for f in os.listdir('pred/') if f.endswith('.h5')]
frameWiseModelSelection = dbc.InputGroup(
    [
        dbc.InputGroupAddon("Frame-Wise model", addon_type="prepend"),
        dbc.Select(id="frame-wise-model", options=[{"label": t, "value": t} for t in trainedModels], value=None)
    ])

pixelWiseModelSelection = dbc.InputGroup(
    [
        dbc.InputGroupAddon("Pixel-Wise  model", addon_type="prepend"),
        dbc.Select(id="pixel-wise-model", options=[{"label": t, "value": t} for t in trainedModels], value=None)
    ])

modelSelection = html.Div([dbc.Row(html.H3('Model selection')),
                           dbc.Row([dbc.Col(modelsBasePathInput),
                                    dbc.Col(frameWiseModelSelection),
                                    dbc.Col(pixelWiseModelSelection),
                                    dbc.Col(dbc.Button(id='button-load-models', children='Load models'), md=1.5)]),
                           html.Div(id='model-selection-output')])

modelSelectionCard = dbc.Col([
    dbc.Card(id='model-selection-card', children=[
        dbc.CardHeader("Model selection"),
        dbc.CardBody([dbc.Row(modelsBasePathInput),
                      dbc.Row(frameWiseModelSelection),
                      dbc.Row(pixelWiseModelSelection)])
    ])
], md=6)

#################
# load model button and spark selector

loadModelsButtAndSparkSel = dbc.Row([
    dbc.Col(dbc.Button(id='button-load-models', children='Load models'), md=2),
    dbc.Col(dbc.InputGroup([
        dbc.InputGroupAddon("Spark selector", addon_type="prepend"),
        dbc.Select(id="spark-selector-dropdown", options=[{"label": t, "value": t} for t in []], value=None)
    ]), md=6)
])

##################
# frame navigation buttons

navigationButtons = html.Div(dbc.Row([
    dbc.Col(dbc.Input(id='input-frame-idx', type='number', value=0, min=0), md=1),
    dbc.Col([
             dbc.Button(id='spark-left-frame', n_clicks=0, color='info', children='<', className='mr-1'),
             dbc.Button(id='spark-right-frame', n_clicks=0, color='info', children='>', className='mr-1')
             ])
]))

##################

# We apply basic HTML formatting to the layout
app.layout = html.Div([
    header,
    dbc.Container(style={'margin': 'auto', 'font-family': 'Verdana'}, fluid=True,
           children=[
               dbc.Row([imagesFilePathCard, modelSelectionCard]),

               loadModelsButtAndSparkSel,
               html.Div(id='heatmap-div'),

               html.H4(children='Frame selector'),
               navigationButtons,
               dcc.Graph(id='show-img'),
               html.Div(id='show-img2'),
               dcc.Loading(
                   id="loading-1",
                   type="default",
                   children=html.Div(id="loading-output-1"),
                   fullscreen=True
               )
           ])
    ])



# Callback for modal popup
@app.callback(
    Output("modal", "is_open"),
    [Input("howto-open", "n_clicks"), Input("howto-close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


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
    fig = px.imshow(img)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


@app.callback(Output('input-frame-idx', 'value'),
              Input('spark-left-frame', 'n_clicks'), Input('spark-right-frame', 'n_clicks'),
              Input('trace-plot', 'hoverData'),
              State('input-frame-idx', 'value'))
def previous_spark(nClicksLeft, nClicksRight, hov, inputFrame):
    if not 'nClicksLeft' in GLOB_DICT:
        return inputFrame
    ret = inputFrame
    if nClicksLeft > GLOB_DICT['nClicksLeft']:
        click = 'left'
        GLOB_DICT['nClicksLeft'] = nClicksLeft
        ret -= 1
    elif nClicksRight > GLOB_DICT['nClicksRight']:
        click = 'right'
        GLOB_DICT['nClicksRight'] = nClicksRight
        ret += 1
    elif hov is not None:
        firstP = hov['points'][0]
        ret = firstP.get('x', inputFrame)
    return ret


@app.callback(Output('spark-selector-dropdown', 'options'),
              Output('loading-1', 'children'), Output('spark-selector-dropdown', 'value'),
              Input('button-load-models', 'n_clicks'),
              State('frame-wise-model', 'value'), State('pixel-wise-model', 'value'), State('models-base-path', 'value'))
def load_model(nClicksLoadModels, frameWiseModel, pixelWiseModel, modelsBasePath):
    if frameWiseModel is None or pixelWiseModel is None:
        return [], None, None

    imageReader = GLOB_DICT['imageReader']
    GLOB_DICT['framePredictor'] = framePredictor = FrameWisePredictor(imageId=imageReader.get_image_id(),
                                        model=os.path.join(modelsBasePath, frameWiseModel))
    GLOB_DICT['pixelPredictor'] = pixelPredictor = PixelWisePredictor(imageId=imageReader.get_image_id(),
                                        model=os.path.join(modelsBasePath, pixelWiseModel))

    sparksDFAsList, sparkPredMasksL = get_clustered_pred_sparks(framePredictor, pixelPredictor)
    GLOB_DICT['sparksDFAsList'], GLOB_DICT['sparkPredMasksL'] = sparksDFAsList, sparkPredMasksL

    retList = ['all'] + [f"Spark{i}" for i in range(len(sparksDFAsList))]

    return [{'label': e, 'value': e} for e in retList], "", "all"


@app.callback(
    Output('heatmap-div', 'children'),
    Input('spark-selector-dropdown', 'value')
)
def plot_heatmap(sparkSelected):
    if sparkSelected is None or not 'sparkPredMasksL' in GLOB_DICT:
        return

    imageReader = GLOB_DICT['imageReader']
    sparksPredMasksL = GLOB_DICT['sparkPredMasksL']
    sparksDFAsList = GLOB_DICT['sparksDFAsList']

    if sparkSelected == 'all':
        intensityHeatmap = get_intensity_heatmap(sparksPredMasksL, imageReader)
        fig = px.imshow(intensityHeatmap)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return dcc.Graph(figure=fig)

    sparkIdx = int(sparkSelected.replace('Spark', ''))
    intensityHeatmap = get_intensity_heatmap([GLOB_DICT['sparkPredMasksL'][sparkIdx]], GLOB_DICT['imageReader'])
    fig = px.imshow(intensityHeatmap)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    g = dcc.Graph(figure=fig, style={'height': '350px'})

    images = imageReader.get_full_images()
    sIni, sFin = sparksDFAsList[sparkIdx][2:]
    sIni = max(sIni-10, 0)
    sFin = min(sFin+10, len(images)-1)
    _masks = np.array([sparksPredMasksL[sparkIdx]]*(sFin-sIni+1))

    trace = (images[sIni:sFin+1, :, :, 2]*_masks).mean(axis=(1, 2))
    fig2 = px.line(x=range(sIni, sFin+1), y=trace, labels={'frame': range(sIni, sFin+1), 'lum': trace})
    fig2.update_traces(mode="markers+lines")
    fig2.update_layout(hovermode="x unified")
    g2 = dcc.Graph(figure=fig2, id='trace-plot', style={'height': '300px'})

    return [g, g2]


if __name__ == "__main__":
    host = os.getenv('DASH_HOST')
    if host is None:
        host = '127.0.0.1'
    app.run_server(threaded=False, host=host)
