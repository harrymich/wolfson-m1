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

    # Remove rows with '---' as value because this messes up the float type change later.
    temp_df = temp_df[temp_df['Speed (GPS)'] != '---']

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
        if time_h == 12:
            time_h = time_h
        else:
            time_h = time_h + 12
    else:
        pass

    session = datetime.datetime(date_y, date_m, date_d, time_h, time_m)

    session_datetime = session.strftime("%a %d %b %Y - %H:%M %p".format())

    return session_datetime


session_dist_list = []
dates = []
split_list = []

for session, name in zip(sessions_list, files):
    session_dist = session['Distance (GPS)'].iloc[-1]
    session_dist_list.append(session_dist)
    dates.append(read_session_datetime(name))
    split_list.append(session['Split (GPS)'].mean())

df = pd.DataFrame({'Date': dates, 'Distance': session_dist_list, 'Av Split Raw': split_list})
df['Average Split'] = df['Av Split Raw'].apply(lambda x: time.strftime("%M:%S", time.gmtime(x)))
df.index = pd.to_datetime([date for date in dates], format="%a %d %b %Y - %H:%M %p")
df = df.sort_index()

colors = px.colors.qualitative.Antique
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=df.index, y=df['Distance'], name='Mileage', marker_color=colors[1]))
fig.add_trace(go.Scatter(x=df.index, y=df['Av Split Raw'], name='Avg. Split', mode='lines',
                         line=dict(color=colors[0]),
                         hovertemplate='%{text}',
                         text=['{}'.format(df['Average Split'].iloc[x]) for x, y in enumerate(df.index)]),
              secondary_y=True)
fig.update_xaxes(
    dtick="D1",
    tickformat='%b %d',
    ticklabelmode="period")
fig.update_xaxes(rangeslider_visible=True)
fig.layout.yaxis2.showgrid = False
fig.update_layout(height=500, hovermode="x unified", legend_traceorder="normal")

full_fig = fig.full_figure_for_development(warn=False)
split_list = list(range(120, int(full_fig.layout.yaxis2.range[1]), 20))
splits = [time.strftime("%M:%S", time.gmtime(item)) for item in split_list]

fig.update_yaxes(title_text="Outing Mileage (m)",
                 tickmode='array', ticksuffix="m", secondary_y=False)
fig.update_yaxes(title_text="Split (s/500m)",
                 tickmode='array', secondary_y=True, tickvals=split_list, ticktext=splits, ticksuffix="s")

dash.register_page(__name__, path='/', name='Home', title='Home', image='wcbc_crest.jpg',
                   description='Outing Summary')

# app.title = "Outing Analysis"
load_figure_template('SOLAR')

layout = html.Div(
    dbc.Row(dbc.Col([
        html.H1(children="Outing Summary"),
        html.P(children="This is a summary of the sessions so far:",
               className="header-description"),
        html.Div(children=[dcc.Graph(figure=fig, id="outing_summary")]),
        html.P(children="You've rowed a total of {} km so far. The average split was: {}/500m.".format(
            round(df['Distance'].sum() / 1000, 1),
            time.strftime("%M:%S", time.gmtime(df['Av Split Raw'].mean())),
        )),
    ])),
)
