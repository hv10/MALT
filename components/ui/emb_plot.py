from functools import partial
from components.pa_clf import get_sample_prediction
from components.ui.prediction_viz import prediction_viz
import plotly.express as px
import pandas as pd
from nicegui import ui, html

from ..state import update_selected_rows
from ..utils import format_cls_label, with_loading_overlay


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
    fig.update_traces(hovertemplate=None, hoverinfo="none")
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


def make_hover_sample_prev(el, point):
    el.clear()
    with el:
        el.classes("bg-amber-500")
        ui.image("/samples/" + point["customdata"][0]).classes("w-full h-auto")
    el.props("style='opacity:1;'")  # position near cursor


async def make_hover_info(el, state, event):
    sender = event.sender
    point = event.args["points"][0]
    pos = event.args["event"]
    size = await ui.run_javascript(f"""
        const el = getElement({sender.id}).$el;
        return {{width: el.clientWidth, height: el.clientHeight}};
    """)
    el.clear()
    xpos = pos["pointerX"] / size["width"]
    direction = "left"
    not_direction = "right"
    if xpos > 0.5:
        direction = "right"
        not_direction = "left"
        xpos = 1.0 - xpos
    ypos = pos["pointerY"] / size["height"]
    with el:
        with ui.card().tight().classes("p-2 w-48"):
            ui.label(f"{point['customdata'][0]}").classes("font-bold")
            ui.label(f"X: {point['x']:.3f}")
            with ui.row().classes("w-full justify-between items-center"):
                ui.label(f"{point['customdata'][1]}").classes("text-sm")
                ann = point["customdata"][2]
                ui.chip(
                    str(ann).capitalize(),
                    color="green"
                    if ann == "h"
                    else "blue"
                    if ann == "m"
                    else "default",
                )
            prediction_viz(get_sample_prediction(state, point["customdata"][3]), state)
    el.style(
        f"{not_direction}: auto; {direction}: calc({xpos * 100}% + 5px); top: calc({ypos * 100}% - 20px); opacity:1;"
    )  # position near cursor


def hide_hover(els):
    for el in els:
        el.style("opacity:0;")


def update_plot(state):
    if state.PLOT is not None:
        state.PLOT.update_figure(make_emb_plot(state))


def emb_plot(state):
    fig = make_emb_plot(state)
    with ui.row().classes("w-full relative"):
        plt = ui.plotly(fig).classes("w-full h-[65svh]").props("id='emb_plot'")
        hover_prev_div = ui.element("div").classes(
            "absolute bottom-5 right-5 w-12 h-12 z-[10] "
            "transition-[opacity] duration-100 ease-in-out opacity-0"
        )
        hover_div = ui.element("div").classes(
            "z-[10] p-2 absolute opacity-0 pointer-events-none "
            "transition-[opacity] duration-100 ease-in-out opacity-0"
        )
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
        lambda e: (make_hover_sample_prev(hover_prev_div, e.args["points"][0]),),
    )
    plt.on(
        "plotly_hover",
        partial(make_hover_info, hover_div, state),
    )
    plt.on(
        "plotly_unhover",
        partial(hide_hover, [hover_div, hover_prev_div]),
    )
    state.PLOT = plt
