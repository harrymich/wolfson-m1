#!/usr/bin/env python
# coding: utf-8

import dash
from dash import html
import dash_bootstrap_components as dbc

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.SOLAR, dbc_css])
server = app.server

sidebar = dbc.Nav(
    [
        dbc.NavLink(
            [
                html.Div(page["name"], className="ms-2"),
            ],
            href=page["path"],
            active="exact",
        )
        for page in dash.page_registry.values()
    ],
    vertical=True,
    pills=True,
    className="bg-dark",
)

app.layout = dbc.Container([
    dbc.Row([dbc.Col(
        [
            html.Img(src='assets\wcbc_crest.jpg', style={'height': '100%', 'width': '100%'})
        ], width=1
    ),
        dbc.Col(html.Div("Outing Analysis",
                         style={'fontSize': 50, 'textAlign': 'center'})),
        dbc.Col(
            [
                html.Img(src='assets\Wolfson_College_Rowing_Blade.png', style={'height': '100%', 'width': '100%'})
            ], width=1
        ),
    ]),
    html.Hr(),

    dbc.Row(
        [
            dbc.Col(
                [
                    sidebar
                ], xs=4, sm=4, md=2, lg=2, xl=2, xxl=2),

            dbc.Col(
                [
                    dash.page_container
                ], xs=8, sm=8, md=10, lg=10, xl=10, xxl=10)
        ]
    )
], fluid=True)

if __name__ == "__main__":
    app.run(debug=False)
