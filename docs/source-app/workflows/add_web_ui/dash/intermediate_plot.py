from typing import Optional

import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

from lightning.app import LightningWork, LightningFlow, LightningApp
from lightning.app.storage import Payload


class LitDash(LightningWork):
    def __init__(self):
        super().__init__(parallel=True)
        self.df = None
        self.selected_year = None

    def run(self):
        df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv")
        self.df = Payload(df)

        dash_app = Dash(__name__)

        dash_app.layout = html.Div(
            [
                dcc.Graph(id="graph-with-slider"),
                dcc.Slider(
                    df["year"].min(),
                    df["year"].max(),
                    step=None,
                    value=df["year"].min(),
                    marks={str(year): str(year) for year in df["year"].unique()},
                    id="year-slider",
                ),
            ]
        )

        @dash_app.callback(Output("graph-with-slider", "figure"), Input("year-slider", "value"))
        def update_figure(selected_year):
            self.selected_year = selected_year
            filtered_df = df[df.year == selected_year]

            fig = px.scatter(
                filtered_df,
                x="gdpPercap",
                y="lifeExp",
                size="pop",
                color="continent",
                hover_name="country",
                log_x=True,
                size_max=55,
            )

            fig.update_layout(transition_duration=500)

            return fig

        dash_app.run_server(host=self.host, port=self.port)


class Processor(LightningWork):
    def run(self, df: Payload, selected_year: Optional[str]):
        if selected_year:
            df = df.value
            filtered_df = df[df.year == selected_year]
            print(f"[PROCESSOR|selected_year={selected_year}]")
            print(filtered_df)


class LitApp(LightningFlow):
    def __init__(self):
        super().__init__()
        self.lit_dash = LitDash()
        self.processor = Processor(parallel=True)

    def run(self):
        self.lit_dash.run()

        # Launch some processing based on the Dash Dashboard.
        self.processor.run(self.lit_dash.df, self.lit_dash.selected_year)

    def configure_layout(self):
        tab1 = {"name": "home", "content": self.lit_dash}
        return tab1


app = LightningApp(LitApp())
