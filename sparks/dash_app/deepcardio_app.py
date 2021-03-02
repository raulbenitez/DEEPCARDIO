# -*- coding: utf-8 -*-

# We start with the import of standard ML librairies
import base64
import io

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

# We add all Plotly and Dash necessary librairies
import plotly.graph_objects as go

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output, State
import plotly.express as px

from deepcardio_utils import ImageReader

GLOB_IMG_READER = ImageReader('170215_RyR-GFP30_RO_01_Serie2_SPARKS-calcium')


###############################################################################

app = dash.Dash()

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

# We apply basic HTML formatting to the layout
app.layout = html.Div(style={'textAlign': 'center', 'width': '800px', 'font-family': 'Verdana'},

                   children=[

                       # DEEPCARDIO
                       html.H1(children="DEEPCARDIO"),

                       html.H4(children='image selector'),

                       dcc.Slider(
                           id='slider',
                           min=0,
                           max=1000,
                           step=1,
                           value=0,
                           # marks={i: '{}'.format(i) for i in range(100)},
                       ),
                        html.Div(id='show-img2'),

                        dcc.Upload(
                            id='upload-image',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                            ]),
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
                        html.Div(id='show-img'),

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


@app.callback(Output('show-img', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


# # The callback function will provide one "Ouput" in the form of a string (=children)
# @app.callback(Output(component_id="show-img2", component_property="children"),
#               # The values correspnding to the three sliders are obtained by calling their id and value property
#               [Input("slider", "value")])
# # The input variable are set in the same order as the callback Inputs
# def update_prediction(idx):
#     im = GLOB_IMG_READER.get_full_images()[idx]
#     input_X = np.array([X1,
#                         df["Viscosity"].mean(),
#                         df["Particles_size"].mean(),
#                         X2,
#                         df["Inlet_flow"].mean(),
#                         df["Rotating_Speed"].mean(),
#                         X3,
#                         df["Color_density"].mean()]).reshape(1, -1)
#
#     # Prediction is calculated based on the input_X array
#     prediction = model.predict(input_X)[0]
#
#     # And retuned to the Output of the callback function
#     return "Prediction: {}".format(round(prediction, 1))

if __name__ == "__main__":
   app.run_server()
