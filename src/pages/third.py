#!/usr/bin/env python
# coding: utf-8

import dash
from dash import html, dcc

dash.register_page(__name__)

layout = html.Div(children=[
    html.H1(children='This is our third page'),

    html.Div(children='''
        This is our third page content. Placeholder for now.
    '''),

])