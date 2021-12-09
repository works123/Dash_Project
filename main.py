# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output

import model 

app = dash.Dash(__name__)

app.layout = html.Div(children=[
        html.H1('Predict the value of Y'),
    html.Label('Provide X: ', style={'color':'red'}),
        dcc.Input(id='x_value', value='5', type='number'),
        html.Br(),
        html.Br(),
        html.Label('Predict Y: ', style={'color':'blue'}),
        html.Br(),
      
        html.Label(id='my-output',children=5),
  
])

@app.callback(
   Output(component_id='my-output', component_property='children'),
    Input(component_id='x_value', component_property='value')
)
def update_output_div(input_value):
    
    #new data 
    new_data = float(input_value)
    
    #preprocessing
    new_data_transformed = model.preprocess_data(new_data)
  
    #prediction 
    predicted_Y = model.predict_result(new_data_transformed)

    # print out prediction
    return ' {}'.format(predicted_Y)


if __name__ == '__main__':
    app.run_server(debug=True)