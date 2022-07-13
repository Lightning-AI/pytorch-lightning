import panel as pn
import plotly.express as px

ACCENT = "#792EE5"

pn.extension("plotly", sizing_mode="stretch_width", template="fast")
pn.state.template.param.update(
    title="⚡ Hello Panel + Lightning ⚡", accent_base_color=ACCENT, header_background=ACCENT
)

pn.config.raw_css.append(
    """
  .bk-root:first-of-type {
    height: calc( 100vh - 150px ) !important;
  }
  """
)


def get_panel_theme():
    """Returns 'default' or 'dark'"""
    return pn.state.session_args.get("theme", [b"default"])[0].decode()


def get_plotly_template():
    if get_panel_theme() == "dark":
        return "plotly_dark"
    return "plotly_white"


def get_plot(length=5):
    xseries = [index for index in range(length + 1)]
    yseries = [x**2 for x in xseries]
    fig = px.line(
        x=xseries,
        y=yseries,
        template=get_plotly_template(),
        color_discrete_sequence=[ACCENT],
        range_x=(0, 10),
        markers=True,
    )
    fig.layout.autosize = True
    return fig


length = pn.widgets.IntSlider(value=5, start=1, end=10, name="Length")
dynamic_plot = pn.panel(
    pn.bind(get_plot, length=length), sizing_mode="stretch_both", config={"responsive": True}
)
pn.Column(length, dynamic_plot).servable()
