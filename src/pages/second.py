#!/usr/bin/env python
# coding: utf-8

import dash
from dash import html, dcc
import os

def ReadSessionDateTime(fname):
    
    import datetime
    
    date_string = fname.split(" ")[-2]
    date_y = int(date_string[0:4])
    date_m = int(date_string[4:6])
    date_d = int(date_string[6:8])
    
    time_string = fname.split(" ")[-1]
    time_h = int(time_string[0:2])
    time_m = int(time_string[2:4])
    
    if "pm" in fname: time_h = time_h + 12
    
    session = datetime.datetime(date_y,date_m,date_d,time_h,time_m)
        
    session_datetime = session.strftime("%a %d %b %Y - %H:%M %p".format())
    
    return session_datetime

# assign path
path, dirs, files = next(os.walk("./csv/"))

# Creating a list of outing dates using the ReadSessionDateTime function. This is needed for Dash dropdown menus
dates = []
for name in files:
    dates.append(ReadSessionDateTime(name))

dash.register_page(__name__, path='/pieces', name='Piece Comparison', title='Pieces',image='wcbc_crest.jpg', description='Compare pieces\' splits and rates')

layout = html.Div(children=[
    html.H1(children='Piece Comparison'),
    html.P(children='This page will allow for picking pieces within and across outings and plotting them on the same graph to compare. Both splits and rate will be options of metrics to plot'),
    html.Div(children='''
        Select the outing date:
    '''),
    dcc.Dropdown(options = dates, value=dates[-1], id='A', placeholder='Select Outing Date'),

])