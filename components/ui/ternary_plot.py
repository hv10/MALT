import numpy as np
from nicegui import ui

from ..data_mgmt import update_data_positions
from ..projection import update_projection
from .emb_plot import update_plot
from .info_chip import make_info_chip


def perpendicular_distance(x, y, x1, y1, x2, y2):
    """Calculate the perpendicular distance from point (x, y) to the line through (x1, y1) and (x2, y2)."""
    return abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / np.sqrt(
        (y2 - y1) ** 2 + (x2 - x1) ** 2
    )


def cart_to_tril(x, y):
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, np.sqrt(3) / 2)
    A = (0, np.sqrt(3) / 2)
    B = (1, np.sqrt(3) / 2)
    C = (0.5, 0)

    u = perpendicular_distance(x, y, C[0], C[1], B[0], B[1])
    v = perpendicular_distance(x, y, A[0], A[1], C[0], C[1])
    w = perpendicular_distance(x, y, B[0], B[1], A[0], A[1])

    return (u, v, w)


def tril_to_cart(a, b, c):
    A = np.array([0, -np.sqrt(3) / 2])
    B = np.array([1, -np.sqrt(3) / 2])
    C = np.array([0.5, 0])
    tril = b * (C - A) + a * (C - B) + C
    return tril


def svg_pointer_event(e, state, width=300, height=300):
    if e.args["buttons"] == 1:
        update_axis(state, e.args["layerX"] / width, e.args["layerY"] / height)


@ui.refreshable
def ternary_plot(state):
    pos_xy = tril_to_cart(*state.META["axis"].values())
    width, height = 250, 250
    content = f"""
        <svg viewBox="0 0 1 1" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <path id="triangle" d="M0.5,0L0,0.8660254L1,0.8660254Z" fill="grey" fill-opacity="0.5" pointer-events="fill" />
            <line x1="0.5" y1="0" x2="0.5" y2="0.866025403784" stroke="darkgrey" stroke-width="0.01" />
            <line x1="0" y1="0.866025403784" x2="0.75" y2="0.433012701892" stroke="darkgrey" stroke-width="0.01" />
            <line x1="1" y1="0.866025403784" x2="0.25" y2="0.433012701892" stroke="darkgrey" stroke-width="0.01" />
            <circle cx="0.5" cy="0" r="0.02" fill="grey" />
            <circle cx="0" cy="0.866025403784" r="0.02" fill="grey" />
            <circle cx="1.0" cy="0.866025403784" r="0.02" fill="grey" />
            <circle id="pos" cx="{pos_xy[0]}" cy="{pos_xy[1]}" r="0.02" fill="orange" />
        </svg>"""
    with ui.column(align_items="center").classes("w-full"):
        with ui.row().classes("w-full"):
            plot = (
                ui.interactive_image(size=(width, height), content=content)
                .on(
                    "pointermove",
                    lambda e: svg_pointer_event(e, state, width, height),
                )
                .on(
                    "pointerdown",
                    lambda e: svg_pointer_event(e, state, width, height),
                )
                .classes(f"w-[{width}px] h-[{height}px] m-auto")
            )
        with ui.row(align_items="baseline").classes("w-full"):
            make_info_chip("""
                Linear combination of Principal Components defining the Y-axis of the Embedding Plot. </br>
                Think *similarity aside from the selected class(es)*. </br>
                Adjust to change the projection.
            """)
            ui.label("Y= ").classes("text-lg")
            ui.label().bind_text_from(
                state.META["axis"], "c", backward=lambda v: f"{v:.2f}·P₁ +"
            )
            ui.label().bind_text_from(
                state.META["axis"], "a", backward=lambda v: f"{v:.2f}·P₂ +"
            )
            ui.label().bind_text_from(
                state.META["axis"], "b", backward=lambda v: f"{v:.2f}·P₃"
            )
            ui.button("reset", on_click=lambda: update_axis(state, 0.5, 0))
    return plot


def update_axis(state, x, y):
    if state.COEFF is None:
        ui.notify(
            "No X-Axis coefficient has been established yet.\n Update the PAC Classifier at least once."
        )
        return
    P = cart_to_tril(x, y)
    P = P / np.sum(P)
    state.META["axis"] = {"a": P[0], "b": P[1], "c": P[2]}
    ternary_plot.refresh(state)
    update_data_positions(
        state.DATA,
        *update_projection(
            state,
            state.COEFF,
            state.INTERCEPTS,
            project_X=False,
            run_pca=False,
        ),
    )

    update_plot(state)
