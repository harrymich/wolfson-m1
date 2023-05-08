#!/usr/bin/env python
# coding: utf-8

import dash
from dash import Dash, dcc, html, Input, Output, callback, dash_table
import plotly.express as px
import os
import numpy as np
import pandas as pd
import time
import re
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Code from: https://github.com/plotly/dash-labs/tree/main/docs/demos/multi_page_example1
dash.register_page(__name__, path='/piece_comparison', name='Piece Comparison', title='Piece Comparison',
                   image='wcbc_crest.jpg', description='Compare pieces\' splits and rates')

# Green Dragon Bridge latitude and longitude
lat = 52.221795
lon = 0.163976

# Upstream reach spinning post coordinates
# lat = 52.221814
# lon = 0.164065

# Earith coordinates
# lat = 52.356794
# lon = 0.049909

def read_session_date_time(fname):
    import datetime

    date_string = fname.split(" ")[-2]
    date_y = int(date_string[0:4])
    date_m = int(date_string[4:6])
    date_d = int(date_string[6:8])

    time_string = fname.split(" ")[-1]
    time_h = int(time_string[0:2])
    time_m = int(time_string[2:4])

    if "pm" in fname:
        if time_h == 12:
            time_h = time_h
        else:
            time_h = time_h + 12

    session = datetime.datetime(date_y, date_m, date_d, time_h, time_m)

    session_datetime = session.strftime("%a %d %b %Y - %H:%M %p".format())

    return session_datetime


# assign path
path, dirs, files = next(os.walk("./csv/"))
file_count = len(files)
# create empty session list
sessions_list = []

# append sessions to list
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

# Creating a list of outing dates using the read_session_date_time function. This is needed for Dash dropdown menus
dates = []
for name in files:
    dates.append(read_session_date_time(name))
# The below line was meant to show outing dates in the dropdown in order but since the csv files are not read in order,
# it messes up the reading of the csv. The wrong date is shown for a given csv.
# dates.sort(key=lambda v: datetime.datetime.strptime(v[5:10], '%d %b'))

layout = html.Div([
    html.H1(children='Piece Comparison'),
    html.P(
        children='This page will allow for picking pieces within and across outings and plotting them on the same '
                 'graph to compare. Both splits and rate will be options of metrics to plot'),
    html.Div(children='''
        Select the outing date:
    '''),
    dcc.Dropdown(options=dates, value=dates[-1:], id='select_outing', placeholder='Select Outing Date', multi=True),
    html.P(children=''),
    dcc.Store(id='store_pieces', data=[], storage_type='memory'),
    html.P(
        children="Now, choose the stroke rate above which a stroke is considered a piece and the stroke count "
                 "below which a piece will not be included:",
        className="header-description"),
    html.Div(['Stroke rate limit:',
              dcc.Input(id="piece_rate_2",
                        type='number', value=25,
                        placeholder="Select rate for piece identification", ),
              'Stroke count limit:',
              dcc.Input(id="stroke_count_2",
                        type='number', value=30,
                        placeholder="Select stroke count for piece exclusion", )
              ], style={'display': 'inline-block'}),
    html.P(children="Now, choose the pieces that you want to compare:"),
    dcc.Checklist(id='piece_selection', options=[]),
    html.P(id='err', style={'color': 'red'}),
    html.Hr(),
    html.Div(['Split range for plot:', dcc.RangeSlider(60, 150, 5, count=1, value=[100, 130], id="split_range")]),
    html.Div(['Rate range for plot:', dcc.RangeSlider(15, 50, 1, count=1, value=[26, 34], id="rate_range")]),
    dcc.Graph(id="piece_figure"),
    html.P("Add benchmark lines for split and rate"),
    html.Div(['Split benchmark:',
              dcc.Input(id="split_bench", type='time', value=None),
              'Rate benchmark:',
              dcc.Input(id="rate_bench", type='number', value=None, step=0.5, placeholder="e.g. 32 spm"),
              ]),
    html.Hr(),
    html.H1(children='Start Comparison'),
    html.P(children="Set how many draw, wind and burn strokes you do during a racing start:",
           className="header-description"),
    html.Div(['Draws:',
              dcc.Input(id="draws",
                        type='number', value=3,
                        placeholder="No of draws", ),
              'Winds:',
              dcc.Input(id="winds",
                        type='number', value=5,
                        placeholder="No of winds", ),
              'Burns:',
              dcc.Input(id="burns",
                        type='number', value=10,
                        placeholder="No of burns", )
              ], style={'display': 'inline-block'}),
    html.Div([dash_table.DataTable(data=[], id='start_comp', export_format='csv')],
             style={'width': '40%', }, className="dbc")
])


# ====== Return a checklist of pieces within the outing to select ======
@callback(Output('piece_selection', 'options'), Output('piece_selection', 'value'), Output('store_pieces', 'data'),
          Input("select_outing", "value"), Input('piece_rate_2', 'value'), Input('stroke_count_2', 'value'))
def piece_prompts(outings, pcrate, strcount):
    prompt = []
    piece_list = []
    rate = pcrate
    stroke_count = strcount

    outings.sort(key=lambda v: datetime.datetime.strptime(v[5:10], '%d %b'))
    for session, datestring in zip([sessions_list[i] for i in [dates.index(value) for value in outings]], outings):

        session_datetime = datestring[4:10] + ' ' + datestring[18:26] + ','

        # df_past_gr_dr = session.loc[(session['GPS Lat.'] >= lat) & (session['GPS Lon.'] >= lon)]
        df_past_gr_dr = session
        df1 = df_past_gr_dr.loc[df_past_gr_dr['Stroke Rate'] >= rate]
        list_of_df = np.split(df1, np.flatnonzero(np.diff(df1['Total Strokes']) != 1) + 1)
        list_of_pieces = [piece for piece in list_of_df if len(piece) >= stroke_count]
        piece_list.extend(list_of_pieces)
        for count, piece in enumerate(list_of_pieces):
            # stroke_count =
            dist = round(piece['Distance (GPS)'].iloc[-1] - piece['Distance (GPS)'].iloc[0], -1)
            piece_time = round(piece['Elapsed Time'].iloc[-1] - piece['Elapsed Time'].iloc[0], 2)
            piece_time = str(datetime.timedelta(seconds = piece_time))[2:9]
            piece_rate = round(piece['Stroke Rate'].mean(), 1)
            piece_split = time.strftime("%M:%S", time.gmtime(piece['Split (GPS)'].mean()))
            prompt.append(
                "{} Piece {} : {}m piece at average rate of {}, average split of {}, lasting {} and {} strokes".format(
                    session_datetime, count + 1, dist, piece_rate, piece_split, piece_time, len(piece)))

    return prompt, prompt[-2:], [df.to_dict() for df in piece_list]


#  ======= Select Outing, Piece Rate lower limit and Stroke Count lower limit to produce piece list ============
@callback(Output('piece_figure', 'figure'),
          Output('start_comp', 'data'),
          Output('err', 'children'),
          Input('piece_selection', 'value'),
          Input('split_range', 'value'),
          Input('rate_range', 'value'),
          Input('draws', 'value'),
          Input("winds", "value"),
          Input('burns', 'value'),
          Input("split_bench", "value"),
          Input('rate_bench', 'value'),
          Input('store_pieces', 'data'),
          Input('piece_selection', 'options')
          )
def piece_list(pieces, split_range, rate_range, draws, winds, burns, split_bench, rate_bench, store_pieces,
               prompt):
    list_of_pieces = [pd.DataFrame.from_dict(i) for i in store_pieces]
    pieces.sort(key=lambda v: (datetime.datetime.strptime(v[:6], '%d %b'), int(v[23:25])))
    pieces_to_plot = [list_of_pieces[i] for i in [prompt.index(i) for i in pieces]]

    colors = px.colors.qualitative.Antique
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, x_title='Stroke Count')

    for x, (i, title) in enumerate(zip(pieces_to_plot, pieces)):
        piece_data = i
        piece_data['Split'] = piece_data['Split (GPS)'].apply(lambda x: time.strftime("%M:%S", time.gmtime(x)))
        piece_data['Stroke Count'] = np.arange(piece_data.shape[0] + 1)[1:]
        piece_data['Piece Time (s)'] = [round(piece_data['Elapsed Time'].loc[i] - piece_data['Elapsed Time'].iloc[0], 2)
                                        for i in piece_data['Elapsed Time'].index]
        piece_data['Piece Time (s)'] = piece_data['Piece Time (s)'].apply(
            lambda x: time.strftime("%M:%S", time.gmtime(x)))
        piece_data['Piece Distance (m)'] = [
            round(piece_data['Distance (GPS)'].loc[i] - piece_data['Distance (GPS)'].iloc[0], 2) for i in
            piece_data['Distance (GPS)'].index]
        piece_data = piece_data.rename(columns={'Elapsed Time': 'Outing Time', 'Distance (GPS)': 'Outing Distance'})
        data = piece_data
        fig.add_trace(go.Scatter(x=data['Stroke Count'], y=data['Split (GPS)'], hovertemplate='%{text}',
                                 text=['{}'.format(data['Split'].iloc[x]) for x, y in enumerate(data.index)],
                                 name=title[:25], mode='lines', line=dict(color=colors[x]), legendrank=x), row=1, col=1)
        fig.add_trace(go.Scatter(x=data['Stroke Count'], y=data['Stroke Rate'], hovertemplate='%{text}',
                                 text=['{}'.format(data['Stroke Rate'].iloc[x]) for x, y in enumerate(data.index)],
                                 name=title[:25], mode='lines', line=dict(color=colors[x]), showlegend=False), row=2,
                      col=1)

    fig.update_layout(height=750,
                      hovermode="x unified",
                      legend_traceorder="normal")
    range_color = split_range
    split_list = list(range(range_color[0], range_color[1] + 1, 5))
    splits = [time.strftime("%M:%S", time.gmtime(item)) for item in split_list]
    fig.update_yaxes(title_text="Split (s/500m)", range=range_color, row=1, col=1,
                     tickmode='array', tickvals=split_list, ticktext=splits, ticksuffix="s")
    fig.update_yaxes(title_text="Stroke Rate (s/m)", range=rate_range, row=2, col=1,
                     tickmode='array')
    full_fig = fig.full_figure_for_development(warn=False)
    if split_bench:
        spl_bench_str = int(split_bench[1]) * 60 + int(split_bench[3]) * 10 + int(split_bench[4])
        fig.add_trace(go.Scatter(x=[0, full_fig.layout.xaxis.range[1]], y=[spl_bench_str, spl_bench_str],
                                 name='Benchmark: {}s'.format(split_bench),
                                 mode='lines', line_dash="dash", hovertemplate='', line=dict(color='red')), row=1,
                      col=1)

    if rate_bench:
        fig.add_trace(go.Scatter(x=[0, full_fig.layout.xaxis.range[1]], y=[rate_bench, rate_bench],
                                 name='Benchmark: {}s/m'.format(rate_bench),
                                 mode='lines', line_dash="dash", hovertemplate='', line=dict(color='red')), row=2,
                      col=1)

    columns_split = {}
    columns_rate = {}
    draws = draws
    winds = winds
    burns = burns

    for x, i in enumerate(pieces_to_plot):
        piece_data = i
        start_length = draws + winds + burns
        if len(piece_data['Split']) < start_length:
            return dash.no_update, dash.no_update, 'One of the pieces you\'ve selected is less than the total start ' \
                                                   'length of {} strokes. Please unselect it or change your start ' \
                                                   'definition below!'.format(start_length)
        draws_split = piece_data['Split'].iloc[draws - 1]
        winds_split = piece_data['Split'].iloc[draws + winds - 1]
        burns_split = piece_data['Split'].iloc[draws + winds + burns - 1]
        draws_rate = piece_data['Stroke Rate'].iloc[draws - 1]
        winds_rate = piece_data['Stroke Rate'].iloc[draws + winds - 1]
        burns_rate = piece_data['Stroke Rate'].iloc[draws + winds + burns - 1]
        column_split = {pieces[x][:15]: [draws_split, winds_split, burns_split]}
        column_rate = {pieces[x][:15]: [draws_rate, winds_rate, burns_rate]}
        columns_split.update(column_split)
        columns_rate.update(column_rate)
    sp = pd.DataFrame(data=columns_split, index=['Draws', 'Winds', 'Burns'])
    ra = pd.DataFrame(data=columns_rate, index=['Draws', 'Winds', 'Burns'])
    df = pd.concat([sp, ra], keys=['Split after:', 'Rate after:'])
    df = df.reset_index()
    df.loc[df['level_0'].duplicated(), 'level_0'] = ''
    df.rename(columns={"level_0": "", 'level_1': ' '}, inplace=True)

    fig.add_vrect(x0=0, x1=draws, annotation_text="Draws", annotation_position='bottom left', row='all', line_width=0,
                  fillcolor="green", opacity=0.2)
    fig.add_vrect(x0=draws, x1=draws + winds, annotation_text="Winds", annotation_position='bottom left', row='all',
                  line_width=0, fillcolor="yellow", opacity=0.2)
    fig.add_vrect(x0=draws + winds, x1=draws + winds + burns, annotation_text="Burns",
                  annotation_position='bottom left', row='all', line_width=0, fillcolor="red", opacity=0.2)
    fig.update_traces(xaxis='x2')

    return fig, df.to_dict('records'), ''
