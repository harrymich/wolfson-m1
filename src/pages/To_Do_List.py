import dash
from dash import dcc, html

dash.register_page(__name__, name='To-do List')

todo = ['Compare change in rate and split after stride. Think about presenting in a table or in a graph instead. Maybe '
        'highlight 5 strokes after stride on graph and then have text output below.',
        'Find a way to plot piece comparison against distance and time and not just stroke rate',
        'Add graphs for the different pieces in the piece comparison page',
        'why the fuck is the legend for a line showing 3 strokes after that line is finished...',
        'Find a way to add weather data on overall plot']

layout = html.Div(
    [
        dcc.Markdown('Things to add later!'),
        html.Div(
            className="trend",
            children=[
                html.Ul(id='my-list', children=[html.Li(i) for i in todo])
            ],
        )
    ]
)
