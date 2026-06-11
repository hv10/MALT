from nicegui import ui
import plotly.express as px
from ..utils import format_cls_label, with_loading_overlay
from ..state import update_selected_rows

@with_loading_overlay
def make_emb_plot(state):
    COLORS = px.colors.qualitative.Plotly  # 10 colors
    SYMBOLS = ["circle", "square", "diamond", "triangle-up", "triangle-down"]
    raw_labels = sorted(
        state.DATA["cls"].unique(),
        key=lambda x: (x == (), str(x)),
    )
    ordered_labels = [format_cls_label(c) for c in raw_labels]
    color_map, symbol_map = {}, {}
    for i, lbl in enumerate(ordered_labels[:-1]):
        color_map[lbl] = COLORS[i % len(COLORS)]
        symbol_map[lbl] = SYMBOLS[(i // len(COLORS)) % len(SYMBOLS)]
    if ordered_labels[-1] == "<unlabeled>":
        if len(ordered_labels) > 1:
            color_map[ordered_labels[-1]] = "gray"
        symbol_map[ordered_labels[-1]] = (
            "circle-open" if len(ordered_labels) > 1 else "circle"
        )

    df = state.DATA[["fpth", "pos_x", "pos_y", "cls", "annot"]].copy()
    df["idx"] = df.index
    df["cls"] = df["cls"].map(format_cls_label)
    if state.HIDE_LABELED:
        df = df[df["annot"] != "h"]
    x_min, x_max = df["pos_x"].min(), df["pos_x"].max()

    fig = px.scatter(
        df,
        x="pos_x",
        y="pos_y",
        color="cls",
        symbol="cls",
        render_mode="webgl",
        labels={"pos_x": "X", "pos_y": "Y", "cls": "Labels"},
        category_orders={"cls": ordered_labels},
        color_discrete_map=color_map,
        symbol_map=symbol_map,
        hover_data={"fpth": True, "cls": True, "annot": True, "idx": True},
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "X: %{x:.3f}<br>"
            "Label: %{customdata[1]}<br>"
            "Annotator: %{customdata[2]}<br>"
            "<extra></extra>"
        )
    )
    MODE_DISP = ""
    if len(ordered_labels) > 1 and state.COEFF is not None:
        fig.add_vline(x=0, opacity=0.5)
        annots = [("← surely not", "top left"), ("surely →", "top right")]
        margin = 1 if state.CLS_TYPE == "MIN_MARGIN" else 0.462
        fig.add_vline(
            x=-margin,
            opacity=0.5,
            line_dash="dash",
            annotation_text=annots[0][0],
            annotation_position=annots[0][1],  # think about if I want this.
            annotation_font_size=15,
        )
        fig.add_vline(
            x=margin,
            opacity=0.5,
            line_dash="dash",
            annotation_text=annots[1][0],
            annotation_position=annots[1][1],  # think about if I want this.
            annotation_font_size=15,
        )
        MODE_DISP = (
            " (Min-Margin)" if state.CLS_TYPE == "MIN_MARGIN" else " (Projection)"
        )
    fig.update_xaxes(
        range=[min(x_min - 0.1, -1.1), max(x_max + 0.1, 1.1)],
        title_text="Projection" if len(ordered_labels) > 1 else "",
    )
    fig.update_yaxes(showticklabels=False, ticks="", title_text="")
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.10, xanchor="right", x=1),
        xaxis_title=f"X{MODE_DISP}",
        yaxis_title="Y",
        uirevision="emb_plot",
    )
    return fig

def update_plot(state):
    if state.PLOT is not None:
        state.PLOT.update_figure(make_emb_plot(state))

def emb_plot(state):
    fig = make_emb_plot(state)
    with ui.row().classes("w-full relative"):
        plt = ui.plotly(fig).classes("w-full h-[65svh]").props("id='emb_plot'")
        hover_img = ui.image().classes("absolute bottom-5 right-5 w-12 z-[10]")
        hover_img.set_visibility(False)
    handler = """(event) => {
        emitEvent('emb_pts_sel', event.points.map(point => point.customdata[3]));
    }"""
    plt.on("plotly_click", js_handler=handler)
    plt.on(
        "plotly_selected",
        js_handler=handler,
    )
    ui.on("emb_pts_sel", lambda e: update_selected_rows(state, e.args))
    plt.on("plotly_deselect", lambda: update_selected_rows(state, []))
    plt.on(
        "plotly_hover",
        lambda e: (
            hover_img.set_source("/images/" + e.args["points"][0]["customdata"][0]),
            hover_img.set_visibility(True),
        ),
    )
    plt.on("plotly_unhover", lambda: hover_img.set_visibility(False))
    state.PLOT = plt
