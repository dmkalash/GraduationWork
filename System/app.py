# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc, dash_table
import plotly.express as px
# from server import Server
from dash.dependencies import Input, Output
from backend import BackEnd

def generate_table(dataframe):
    return dash_table.DataTable(dataframe.to_dict('records'),
                                [{"name": col, "id": col} for col in dataframe.columns],
                                style_cell={'textAlign': 'center'},
                                style_cell_conditional=[
                                    {
                                        'if': {'column_id': 'Region'},
                                        'textAlign': 'center'
                                    }],
                                id='tbl')


app = Dash('app')
service = BackEnd()

ts_df, doc_df = service.step()
fig = px.line(ts_df, x='timestamp', y='value', color='forecast', markers=True)
fig.update_layout({
    #'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(247, 249, 251, 0)'
})

app.layout = html.Div(children=[
            html.H1(children='Система мониторинга',
                    style={
                        'textAlign': 'center'
                    }),

            html.Div(children='''
                        Мониторинг данных индекса Dow-Jones в реальном времени. \n
                        Для получения прогноза анализируются прошлые значения индекса и поступающие в реальном времени
                        заголовки новостей Reuters.
                    ''',
                    style={
                        'textAlign': 'center',
                        'fontSize': 20
                    }),
            dcc.Graph(
                id='live-update-graph',
                figure=fig
            ),
            html.Div([
                html.H2(children='Опорные документы'),
                html.Div([
                    generate_table(doc_df)
                ], id='text-dataframe')],
                style={
                        'textAlign': 'center',
                        'fontSize': 20
                    }),
            dcc.Interval(
                        id='interval-component',
                        interval= 2 * 1000, # in milliseconds
                        n_intervals=0
            ),
            ], style={'padding': '3% 15% 15% 15%', 'backgroundColor':'#F7F9FB'})  #C2CAD0 style={}


@app.callback(Output('live-update-graph', 'figure'),
              Output('text-dataframe', 'children'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    ts_df, doc_df = service.step()
    fig = px.line(ts_df, x='timestamp', y='value', color='forecast', markers=True)
    fig.update_layout({
        # 'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(247, 249, 251, 0)'
    })
    table = generate_table(doc_df)
    return fig, table


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False, port=11000)
