#!/usr/bin/env python
# coding: utf-8

import os
import io
import time
import numpy as np
import urllib.request
import pandas as pd
import dash
from dash import dcc, dash_table, html, callback, Output, Input
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from dash_bootstrap_templates import ThemeSwitchAIO, ThemeChangerAIO, template_from_url
from datetime import date
import datetime
from dash.dependencies import Input, Output
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# assign path for reading csv files.
path, dirs, files = next(os.walk("./csv/"))
file_count = len(files)
# create empty outing session list
sessions_list = []

# append outing sessions to list
for i in range(file_count):
    temp_df = pd.read_csv("./csv/" + files[i], skiprows=28, usecols=[1, 3, 4, 5, 8, 9, 10, 22, 23]).drop([0])

    # Change type of most columns to float
    temp_df = temp_df.astype({"Distance (GPS)": float, 'Speed (GPS)': float, 'Stroke Rate': float, 'Total Strokes': int,
                              'Distance/Stroke (GPS)': float, 'GPS Lat.': float, 'GPS Lon.': float})

    # Convert elapsed time to seconds using string split, asfloat and multiplying by seconds
    temp_df['Elapsed Time'] = (
            temp_df['Elapsed Time'].str.split(':', n=2, expand=True).iloc[:, -3:-2].astype(float) * 3600).join(
        temp_df['Elapsed Time'].str.split(':', n=2, expand=True).iloc[:, -2:-1].astype(float) * 60).join(
        temp_df['Elapsed Time'].str.split(':', n=2, expand=True).iloc[:, -1:].astype(float)).sum(axis=1)

    # Convert split to seconds (similar as above)
    temp_df['Split (GPS)'] = (
            temp_df['Split (GPS)'].str.split(':', n=2, expand=True).iloc[:, -2:-1].astype(float) * 60).join(
        temp_df['Split (GPS)'].str.split(':', n=2, expand=True).iloc[:, -1:].astype(float)).sum(axis=1)
    sessions_list.append(temp_df)

# Parameter Definition

# Green Dragon Bridge latitude and longitude. This is used later to define after which location pieces
# are actually identified. Also defining stroke slice which is used to define which sections of a dataframe
# are analysed.
gr_dr_lat = 52.217426
gr_dr_lon = 0.145928
stroke_slice = (0, -1)

# Function Definition
# Reading a session's date and time. Credit to Rob Sales.
def read_session_datetime(fname):
    import datetime

    date_string = fname.split(" ")[-2]
    date_y = int(date_string[0:4])
    date_m = int(date_string[4:6])
    date_d = int(date_string[6:8])

    time_string = fname.split(" ")[-1]
    time_h = int(time_string[0:2])
    time_m = int(time_string[2:4])

    if "pm" in fname:
        time_h = time_h + 12
    else:
        pass

    session = datetime.datetime(date_y, date_m, date_d, time_h, time_m)

    session_datetime = session.strftime("%a %d %b %Y - %H:%M %p".format())

    return session_datetime


def get_statistics(fname, stroke_slice):
    data = fname

    if stroke_slice:
        indices = slice(stroke_slice[0], stroke_slice[1] - 1)
    else:
        indices = slice(0, (data["Total Strokes"].size - 1))

    speed = data["Speed (GPS)"][indices]
    min_speed = speed.min()
    max_speed = speed.max()
    avg_speed = np.average(speed)

    split = data["Split (GPS)"][indices]
    min_split = split.min()
    max_split = split.max()
    avg_split = np.average(split)

    stroke_rate = data["Stroke Rate"][indices]
    min_stroke_rate = stroke_rate.min()
    max_stroke_rate = stroke_rate.max()
    avg_stroke_rate = np.average(stroke_rate)

    distance_per_stroke = data["Distance/Stroke (GPS)"][indices]
    min_distance_per_stroke = distance_per_stroke.min()
    max_distance_per_stroke = distance_per_stroke.max()
    avg_distance_per_stroke = np.average(distance_per_stroke)

    total_strokes = data["Total Strokes"][indices]
    total_strokes = int(total_strokes.iloc[-1] - total_strokes.iloc[0])

    stroke_count = "{} - {}".format("Total Number of Strokes", total_strokes)

    distance = data["Distance (GPS)"][indices]
    total_distance = distance.iloc[-1] - distance.iloc[0]

    distance = "{} - {:.2f}".format("Total Distance Rowed (m)", total_distance)

    elapsed_time = data["Elapsed Time"][indices]
    elapsed_time = elapsed_time.iloc[-1] - elapsed_time.iloc[0]
    elapsed_time = time.strftime("%M:%S", time.gmtime(elapsed_time))

    time_el = "{} - {}".format("Total Elapsed Time (mm:ss)", elapsed_time)

    sum_data = [[min_speed, max_speed, avg_speed],
                [min_split, max_split, avg_split],
                [min_stroke_rate, max_stroke_rate, avg_stroke_rate],
                [min_distance_per_stroke, max_distance_per_stroke, avg_distance_per_stroke]]

    sum_table = pd.DataFrame(data=sum_data, index=['Speed (m/s)', 'Split (s/500m)', 'Rate (spm)', 'DPS (m)'],
                             columns=['Min', 'Max', 'Avg'])
    sum_table['Avg'] = sum_table['Avg'].round(2)

    return sum_table, stroke_count, distance, time_el


def plot_split(data):
    df = data
    range_color = [80, 130]
    split_list = list(range(range_color[0], range_color[1] + 1, 5))
    splits = [time.strftime("%M:%S", time.gmtime(item)) for item in split_list]
    hover_name = df['Stroke Count'].apply(lambda x: 'Stroke {:7.0f}'.format(x)).copy()
    df['Split'] = df['Split (GPS)'].apply(lambda x: time.strftime("%M:%S", time.gmtime(x)))
    fig = px.scatter_mapbox(df, lat="GPS Lat.", lon="GPS Lon.", color="Split (GPS)",
                            color_continuous_scale='plasma_r', range_color=range_color,
                            hover_name=hover_name, hover_data={'Split': True,
                                                               'Stroke Rate': True,
                                                               'Piece Time (s)': True,
                                                               'Piece Distance (m)': True,
                                                               'Split (GPS)': False,
                                                               'GPS Lon.': False,
                                                               'GPS Lat.': False},
                            size_max=10, zoom=13)
    fig.update_layout(height=500, mapbox_style="open-street-map")
    fig.update_layout(coloraxis_colorbar=dict(
        title='Boat Split (mm:ss)',
        titleside='right',
        ticks='outside',
        tickmode='array',
        tickvals=split_list,
        ticktext=splits,
        ticksuffix="s"))

    return fig


# Creating a list of outing dates using the read_session_datetime function. This is needed for Dash dropdown menus
dates = []
for name in files:
    dates.append(read_session_datetime(name))

x_axis = ['Stroke Count', 'Piece Time (s)', 'Piece Distance (m)']

dash.register_page(__name__, path='/', name='Home', title='Home', image='wcbc_crest.jpg',
                   description='Come here for all your sweet split and rate analysis')

# app.title = "Outing Analysis"
load_figure_template('SOLAR')

layout = html.Div(
    dbc.Row(dbc.Col([
        html.P(children="First, choose the outing you want to analyse from the dropdown menu below",
               className="header-description"
               ),
        dcc.Dropdown(options=dates, value=dates[-1], id='A', placeholder='Select Outing Date'),
        html.H2(children="Outing summary"),
        html.Div([dash_table.DataTable(data=[], id='session_summary')],
                 style={'width': '20%', }, className="dbc"),
        html.P(id='str_out'),
        html.P(id='dis_out'),
        html.P(id='tim_out'),
        dcc.Store(id='store_piece_list', data=[], storage_type='memory'),
        html.Hr(),
        html.H3(children="Piece Identification"),
        html.P(
            children="Now, choose the stroke rate above which a stroke is considered a piece and the stroke count "
                     "below which a piece will not be included:",
            className="header-description"),
        html.Div(['Stroke rate limit:',
                  dcc.Input(id="piece_rate",
                            type='number', value=30,
                            placeholder="Select rate for piece identification", ),
                  'Stroke count limit:',
                  dcc.Input(id="stroke_count",
                            type='number', value=10,
                            placeholder="Select stroke count for piece exclusion", )
                  ], style={'display': 'inline-block'}),

        html.P(children="Now, choose the piece in this outing that you want to analyse from the dropdown menu below:",
               className="header-description"),
        dcc.Dropdown(options=[], value='0', id='Piece', placeholder='Select Piece', clearable=False),
        html.H3(children="Piece Summary"),
        html.Div([dash_table.DataTable(data=[], id='piece_summary')],
                 style={'width': '20%', }, className="dbc"),
        html.P(id='str_pie'),
        html.P(id='dis_pie'),
        html.P(id='tim_pie'),
        html.Hr(),
        html.H3(children="Piece Map"),
        html.P(
            children="The selected piece is mapped below and will update if you select another one. It's an "
                     "interactive map so hover over each point (stroke) to see the data (e.g. split and rate) "
                     "associated with that stroke:",
            className="header-description"),
        html.Div(
            children=[
                html.Div(
                    children=dcc.Graph(
                        id="session_chart", ))]),
        html.P(children="Plot against:",
               className="header-description"),
        dcc.Dropdown(options=x_axis, value=x_axis[0], id='x_axis', placeholder='Select variable to plot against',
                     clearable=False),
        html.Div(['Split and rate range for plot:']),
        dcc.RangeSlider(60, 150, 5, count=1, value=[80, 120], id="split_range"),
        dcc.RangeSlider(15, 50, 1, count=1, value=[30, 45], id="rate_range"),
        html.Div(
            children=[
                html.Div(
                    dcc.Graph(
                        id="piece_chart", ),
                )]),
        html.P("Add benchmark lines for split and rate"),
        html.Div(['Split benchmark:',
                  dcc.Input(id="split_bench_2", type='time', value=None),
                  'Rate benchmark:',
                  dcc.Input(id="rate_bench_2", type='number', value=None, step=0.5, placeholder="e.g. 32 spm"),
                  ]),
        html.Hr(),
        html.H3(children="Full Piece Data"),
        html.P(children="See the full piece data below",
               className="header-description"),
        html.Div([dash_table.DataTable(data=[], id='piece_data', export_format='csv')],
                 style={'width': '40%', }, className="dbc")
    ])),
)


#  ======= Select Outing and output session summary data ============
@callback(Output('session_summary', 'data'),
          Output('str_out', 'children'),
          Output('dis_out', 'children'),
          Output('tim_out', 'children'),
          Input('A', 'value')
          )
def update_output(value):
    stats = get_statistics(sessions_list[dates.index(value)], stroke_slice)
    stats[0].loc['Split (s/500m)'] = stats[0].loc['Split (s/500m)'].apply(
        lambda x: time.strftime("%M:%S", time.gmtime(x)))
    return stats[0].reset_index(names='').to_dict('records'), stats[1], stats[2], stats[3]


#  ======= Select Outing, Piece Rate lower limit and Stroke Count lower limit to produce piece list ============
@callback(Output('Piece', 'options'),
          Output('Piece', 'value'),
          Output('store_piece_list', 'data'),
          Input('A', 'value'),
          Input('piece_rate', 'value'),
          Input('stroke_count', 'value')
          )
def piece_dropdown(value, rate, stroke_count):
    df = sessions_list[dates.index(value)]
    df_past_gr_dr = df.loc[(df['GPS Lat.'] >= gr_dr_lat) & (df['GPS Lon.'] >= gr_dr_lon)]
    df1 = df_past_gr_dr.loc[df['Stroke Rate'] >= rate]
    list_of_df = np.split(df1, np.flatnonzero(np.diff(df1['Total Strokes']) != 1) + 1)
    list_of_pieces = [i for i in list_of_df if len(i) >= stroke_count]
    prompt = []
    for count, piece in enumerate(list_of_pieces):
        stroke_count = len(piece)
        dist = round(piece['Distance (GPS)'].iloc[-1] - piece['Distance (GPS)'].iloc[0], -1)
        piece_time = round(piece['Elapsed Time'].iloc[-1] - piece['Elapsed Time'].iloc[0], 2)
        piece_time = time.strftime("%M:%S", time.gmtime(piece_time))
        piece_rate = round(piece['Stroke Rate'].mean(), 1)
        piece_split = time.strftime("%M:%S", time.gmtime(piece['Split (GPS)'].mean()))
        prompt.append(
            "Piece {}: {}m piece at average rate of {}, average split of {}, lasting {} and {} strokes".format(
                count + 1, dist, piece_rate, piece_split, piece_time, stroke_count))
    return prompt, prompt[-1], [df.to_dict() for df in list_of_pieces]


#  ======= Produce graphs, tables and plots for piece ============
@callback(Output('piece_summary', 'data'),
          Output('str_pie', 'children'),
          Output('dis_pie', 'children'),
          Output('tim_pie', 'children'),
          Output('piece_data', 'data'),
          Output('session_chart', 'figure'),
          Output('piece_chart', 'figure'),
          Input('Piece', 'value'),
          Input('x_axis', 'value'),
          Input('split_range', 'value'),
          Input('rate_range', 'value'),
          Input('store_piece_list', 'data'),
          Input('split_bench_2', 'value'),
          Input('rate_bench_2', 'value')
          )
def piece_summary(piece_value, x_axis, split_range, rate_range, piece_list, spl_bench, rt_bench):
    list_of_pieces = [pd.DataFrame.from_dict(i) for i in piece_list]
    stats = get_statistics(list_of_pieces[int(re.search(r'\d+', piece_value).group()) - 1], stroke_slice)
    stats[0].loc['Split (s/500m)'] = stats[0].loc['Split (s/500m)'].apply(
        lambda x: time.strftime("%M:%S", time.gmtime(x)))
    piece_data = list_of_pieces[int(re.search(r'\d+', piece_value).group()) - 1]
    piece_data['Stroke Count'] = np.arange(piece_data.shape[0] + 1)[1:]
    piece_data['Piece Time (s)'] = [round(piece_data['Elapsed Time'].loc[i] - piece_data['Elapsed Time'].iloc[0], 2) for
                                    i in piece_data['Elapsed Time'].index]
    piece_data['Piece Time (s)'] = piece_data['Piece Time (s)'].apply(lambda x: time.strftime("%M:%S", time.gmtime(x)))
    piece_data['Piece Distance (m)'] = [
        round(piece_data['Distance (GPS)'].loc[i] - piece_data['Distance (GPS)'].iloc[0], 2) for i in
        piece_data['Distance (GPS)'].index]
    piece_data = piece_data.rename(columns={'Elapsed Time': 'Outing Time', 'Distance (GPS)': 'Outing Distance'})
    plot = plot_split(list_of_pieces[int(re.search(r'\d+', piece_value).group()) - 1])

    data = piece_data
    x = data[x_axis]
    data['Split'] = data['Split (GPS)'].apply(lambda x: time.strftime("%M:%S", time.gmtime(x)))
    colors = px.colors.qualitative.Plotly
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, x_title=x_axis, specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=x, y=data['Split (GPS)'], hovertemplate='%{text}',
                             text=['{}'.format(data['Split'].iloc[x]) for x, y in enumerate(data.index)], name='Split',
                             mode='lines', line=dict(color=colors[0])))
    fig.add_trace(go.Scatter(x=x, y=data['Stroke Rate'], name='Rate', mode='lines', line=dict(color=colors[1])),
                  secondary_y=True)

    range_color = split_range
    split_list = list(range(range_color[0], range_color[1] + 1, 5))
    splits = [time.strftime("%M:%S", time.gmtime(item)) for item in split_list]
    fig.update_yaxes(title_text="Split (s/500m)", range=range_color, row=1, col=1, secondary_y=False,
                     tickmode='array', tickvals=split_list, ticktext=splits, ticksuffix="s")
    fig.update_yaxes(title_text="Stroke rate (s/m)", range=rate_range, row=1, col=1, secondary_y=True)
    fig.layout.yaxis2.showgrid = False
    if spl_bench:
        spl_bench_str = int(spl_bench[1])*60+int(spl_bench[3])*10+int(spl_bench[4])
        fig.add_trace(go.Scatter(x=[x.min(), x.max()], y=[spl_bench_str, spl_bench_str],
                                 name='Benchmark: {}s'.format(spl_bench),
                                 mode='lines', line_dash="dash", hovertemplate='', line=dict(color=colors[0])))

    if rt_bench:
        fig.add_trace(go.Scatter(x=[x.min(), x.max()], y=[rt_bench, rt_bench],
                                 name='Benchmark: {}s/m'.format(rt_bench),
                                 mode='lines', line_dash="dash", hovertemplate='', line=dict(color=colors[1])),
                      secondary_y=True)

    fig.update_layout(height=500, hovermode="x unified", legend_traceorder="normal")

    return stats[0].reset_index(names='').to_dict('records'), stats[1], stats[2], stats[3], piece_data.drop(
        ['GPS Lon.', 'GPS Lat.'], axis=1).to_dict('records'), plot, fig