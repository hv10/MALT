#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#    "scikit-learn>=1.8,<2",
#    "transformers[torch]>=5.7,<6",
#    "torchvision==0.26.0",
#    "matplotlib>=3.10,<4",
#    "tqdm<5",
#    "pillow>=12.2,<13",
#    "numpy",
#    "pandas>=3.0,<4",
#    "nicegui>=3,<4",
#    "plotly>=6.7,<7",
#    "pywebview",
#    "tables",
#    "pyarrow",
# ]
# ///

import argparse as ap
import json
import logging
from functools import partial, wraps
from pathlib import Path
from time import perf_counter

import plotly.io as pio
from nicegui import ElementFilter, app, binding, run, ui
from tqdm import tqdm

tqdm.pandas()
logging.basicConfig(level=logging.WARN)


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__}: {perf_counter() - start:.3f}s")
        return result

    return wrapper


# ==================== STATE Mgmt ====================
from components.state import State  # noqa: E402

STATE = None

from components.embeddings import apply_emb_to_df  # noqa: E402
from components.history import redo, undo  # noqa: E402
from components.utils import (  # noqa: E402
    refresh,
    register_refresh,
    with_loading_overlay,
)


# =================== DATA Mgmt ====================
from components.data_mgmt import (  # noqa: E402
    save_callback,
    add_cls_from_file,
    add_new_cls,
    clear_class_for_selected,
    load_folder,
    load_prior_state,
    remove_class,
    set_class_for_selected,
)
from components.pa_clf import update_pa_clf  # noqa: E402



async def save_async_cb(state, btn, unique=False):
    ui.notify("Saving...")
    btn.disable()
    ui.query("body").classes("loading")
    fpth = await run.io_bound(save_callback, state, unique)
    ui.page_title(f"MALT - {fpth.stem}")
    btn.enable()
    ui.query("body").classes(remove="loading")
    ui.notify("Save Successful")


def remove_file(state, i):
    state.DATA.drop(index=i, inplace=True)
    refresh(state)


# ==================== GUI-Components ====================
from components.ui.emb_plot import emb_plot, update_plot  # noqa: E402

from components.ui.force_sim_plot import force_similarity_plot  # noqa: E402
from components.ui.info_chip import make_info_chip  # noqa: E402
from components.ui.ternary_plot import ternary_plot  # noqa: E402

from components.ui.class_hist import class_hist  # noqa: E402
from components.ui.data_preview import data_preview  # noqa: E402
from components.ui.data_table import data_table, update_data_table  # noqa: E402

# ==================== Embedder + Preview Functions ====================
from components.embedder.hf_image_emb import embed_image  # noqa: E402
from components.ui.previews.image_preview import (  # noqa: E402
    make_image_detail_preview,
    make_image_sample_preview,
)

apply_emb_to_df.embed_func = embed_image  # type: ignore
data_preview.make_sample_preview = make_image_sample_preview  # type: ignore
data_preview.make_detail_preview = make_image_detail_preview  # type: ignore


# =================== GUI-Setup ====================
async def handle_file_upload(state, dialog, e):
    content = await e.file.text()
    add_cls_from_file(state, content)
    dialog.close()


def import_classes_dialog(state):
    with ui.dialog() as dialog, ui.card():
        ui.label("Import classes from file. \nEach line is one class.")
        ui.upload(on_upload=partial(handle_file_upload, state, dialog))
    ui.button("Import", icon="upload", on_click=lambda: dialog.open())


async def update_clf_cb(state, val, btn):
    btn.disable()
    ui.query("body").classes("loading")
    await run.io_bound(update_pa_clf, state, val)
    ui.query("body").classes(remove="loading")
    btn.enable()


@ui.refreshable
def plot_controls(state):
    with ui.row(align_items="baseline").classes("w-full"):
        make_info_chip("""
                    Select the observed set — the classifier(s) driving the x-axis (task fittingness).</br>
                    When multiple are selected, their classifiers act jointly.</br>

                    - MIN_MARGIN: signed distance to the nearest selected decision boundary.
                    - Think: *Weakest confidence across the observed set — surfaces uncertainty.*
                    - PROJECTION: aggregated alignment across the selected classifiers' span.
                    - Think: *Collective consistency across the observed set — how strongly the selected labels jointly support the point.*
        """)
        ui.label("X=").classes("text-lg")
        x_sel = (
            ui.select(
                options=state.META["classes"],
                with_input=True,
                multiple=True,
            )
            .props("use-chips")
            .classes("w-48")
        )
        with ui.dropdown_button(
            "UPDATE",
            # icon="update",
            split=True,
            on_click=lambda e: update_clf_cb(state, x_sel.value, e.sender),
        ):
            with ui.row().classes("p-4 pb-8 items-center min-w-xs overflow-y-visible"):
                ui.slider(min=-1, max=1, step=0.05).bind_value(
                    state, "CLS_THRESHOLD"
                ).props("label-always marker switch-label-side").classes("w-full")
                ui.toggle(["MIN_MARGIN", "PROJECTION"]).bind_value(state, "CLS_TYPE")
        ui.separator()
    with ui.row(align_items="center").classes("w-full"):
        ternary_plot(state)
        ui.separator()
    with ui.row(align_items="baseline").classes("w-full"):
        with ui.row(align_items="baseline").classes("w-full"):
            cls_sel = (
                ui.select(
                    options=state.META["classes"],
                    with_input=True,
                    multiple=state.META["multilabel"],
                )
                .props("use-chips")
                .classes("w-48")
            )
            ui.button(
                "Assign to Selected",
                on_click=lambda: set_class_for_selected(state, cls_sel.value),
            )
        with ui.row(align_items="baseline").classes("w-full"):
            ui.label("Labels are ")
            ui.toggle({True: "ADDED", False: "REPLACED"}).bind_value(
                state, "ADD_LABELS"
            )
            ui.button(
                "CLEARED", on_click=lambda: clear_class_for_selected(state)
            ).props("flat")
    ElementFilter(kind=ui.input).props("dense")
    ElementFilter(kind=ui.select).props("dense options-dense")


@ui.refreshable
def label_controls(state):
    with ui.row(align_items="baseline").classes("w-full"):
        inp = ui.input(
            label="Add new class",
            placeholder="class_name",
            validation={"Input too long": lambda value: len(value) < 20},
        ).props("clearable")
        with ui.dropdown_button(
            "Add", split=True, on_click=lambda: add_new_cls(state, inp)
        ):
            import_classes_dialog(state)
    with ui.row().classes("w-full"):
        table = ui.table(
            columns=[
                {"name": "name", "label": "Label", "field": "name", "align": "left"},
                {"name": "action", "label": "Del.", "align": "center"},
            ],
            rows=[{"name": lbl} for lbl in state.META["classes"]],
            row_key="name",
        ).classes("w-full")
        table.add_slot(
            "body-cell-action",
            """
            <q-td :props="props">
                <q-btn icon="delete" @click="() => $parent.$emit('del_label', props.row)" flat />
            </q-td>
        """,
        )
        table.on("del_label", lambda e: remove_class(state, e.args["name"]))


def make_overlay():
    overlay = ui.element("div").classes(
        "fixed inset-0 z-[9999] flex items-center justify-center bg-black/30"
    )
    with overlay:
        with ui.column():
            ui.label("Processing...").classes("text-white text-lg mb-4")
            ui.spinner(size="xl", color="white")
    overlay.set_visibility(False)
    return overlay


def progress_info(state):
    def get_progress(d):
        return d["annot"].value_counts().get("h", 0) / len(d)

    with ui.row(wrap=False).classes("w-fit items-center"):
        ui.slider(min=0, max=1).bind_value_from(
            state, "DATA", backward=get_progress
        ).props("disable flat dense").classes("w-48")
        ui.label().bind_text_from(
            state, "DATA", backward=lambda d: f"{get_progress(d) * 100:.1f}% labeled"
        )
        ui.switch("Hide Labeled", on_change=lambda: refresh(state)).bind_value(
            state, "HIDE_LABELED"
        ).classes("ml-auto")


def make_gui(state):
    """
    This function draws the complete GUI.
    """
    if state.THEME == "dark":
        ui.dark_mode().enable()
        pio.templates.default = "plotly_dark"
    ui.add_css("body.loading, body.loading * { cursor: wait !important; }")
    app.add_static_files("/samples", state.OUT_DIR)
    with ui.grid(columns="3fr 7fr").classes("w-full"):
        with ui.column().classes("w-full"):
            with ui.tabs() as tabs:
                ui.button("Save").on(
                    "click",
                    lambda e: save_async_cb(state, e.sender, e.args["shiftKey"]),
                    args=["shiftKey"],
                ).classes("m-[1em]")
                ui.tab("ctrl", label="CTRL").props('indicator-color="blue-6"')
                ui.tab("labels", label="Labels")
                ui.tab("info", label="Info")
            with ui.tab_panels(tabs, value="ctrl").classes("w-full"):
                with ui.tab_panel("ctrl"):
                    plot_controls(state)
                    progress_info(state)
                    data_table(state)
                with ui.tab_panel("labels"):
                    label_controls(state)
                with ui.tab_panel("info"):
                    ui.editor().bind_value(state.META, "note").classes(
                        "w-full max-w-[600px]"
                    )
                    class_hist(state)
                    force_similarity_plot(state, sign_invariant=True)
                    ui.label("Meta:")
                    ui.code(language="json").bind_content_from(
                        state,
                        "META",
                        backward=lambda d: json.dumps(
                            {k: v for k, v in d.items() if k != "note"}, indent=2
                        ),
                    )
        with ui.column().classes("w-full min-h-svh"):
            emb_plot(state)
            dprv = data_preview(state)
            binding.bind_from(
                self_obj=dprv,
                self_name="selected_rows",
                other_obj=state,
                other_name="SELECTED_ROWS",
            )
    with (
        ui.card()
        .classes("fixed top-4 left-1/2 -translate-x-1/2 z-50 p-0")
        .props("flat")
    ):
        with ui.row().classes("gap-0"):
            ui.button(icon="undo", on_click=lambda: undo(state)).props("flat").tooltip(
                "Undo (Ctrl+Z)"
            )
            ui.button(icon="redo", on_click=lambda: redo(state)).props("flat").tooltip(
                "Redo (Ctrl+Shift+Z)"
            )
    ElementFilter(kind=ui.input).props("dense")
    ElementFilter(kind=ui.select).props("dense options-dense")
    with_loading_overlay.overlay = make_overlay()  # type: ignore
    register_refresh(
        elements=[
            update_plot,
            update_data_table,
            force_similarity_plot.refresh,
            class_hist.refresh,
            dprv.refresh,
        ]
    )  # type: ignore
    ui.keyboard(on_key=global_handle_key)


# ==================== MAIN ====================
def setup_state(model, directory, color_blind, prior, theme, task):
    global STATE

    if STATE is None:
        state = State()
        # setup model for embedding
        state.META["model"] = model
        state.COLORBLIND = color_blind
        state.THEME = theme
        if task == "multilabel":
            state.META["multilabel"] = True
        directory = Path(directory)
        if not directory.is_dir():
            directory = directory.parent
        state.OUT_DIR = directory.resolve().absolute()
        if not prior:
            # load_model(state, model)
            print("Model Loaded")
            load_folder(state, directory)
        print("OUTDIR:", state.OUT_DIR)
        STATE = state


def global_handle_key(e):
    if e.action.keydown and not e.action.repeat:
        if (
            e.key == "z"
            and (e.modifiers.ctrl or e.modifiers.meta)
            and not e.modifiers.shift
        ):
            undo(STATE)
        elif (e.key == "Z" or (e.key == "z" and e.modifiers.shift)) and (
            e.modifiers.ctrl or e.modifiers.meta
        ):
            redo(STATE)


@ui.page("/select")
async def select_page():
    global prior_path, STATE
    opts = sorted(list(STATE.OUT_DIR.glob("*.malt")))

    if not opts:
        ui.notify("No state files found.")
        return

    if len(opts) == 1:
        o = opts[0]
        prior_path = o
        load_prior_state(STATE, o)
        ui.page_title("MALT")
        ui.navigate.to("/")
        return

    with ui.card().classes("absolute-center"):
        ui.label("Select a state to load:").classes("text-h6")
        for opt in opts:

            def choose(o=opt):
                global prior_path
                prior_path = o
                load_prior_state(STATE, prior_path)
                ui.page_title(f"MALT - {o.stem}")
                ui.navigate.to("/")

            with (
                ui.button(on_click=choose).props("flat").classes("w-full justify-start")
            ):
                ui.label(opt.stem).classes("text-left w-full")


prior_path = None


@ui.page("/")
async def index():
    global STATE, prior_path
    if prior_path is None:
        ui.navigate.to("/select")
        return
    make_gui(STATE)


# Needs to be unguarded to work.
parser = ap.ArgumentParser()
parser.add_argument(
    "--model",
    default="microsoft/resnet-50",
    help="Which model to use for embedding.",
)
parser.add_argument(
    "-t",
    "--task",
    choices=["classification", "regression", "multilabel"],
    default="classification",
)
parser.add_argument("-d", "--directory", type=Path, default=Path.cwd())
parser.add_argument("--color-blind", action="store_true")
parser.add_argument("--prior", action="store_true")
parser.add_argument("--theme", choices=["dark", "light"], default="dark")
parser.add_argument(
    "--prepare", action="store_true", help="Run data preparation, save and exit."
)
args = parser.parse_args()

setup_cb = partial(
    setup_state,
    args.model,
    args.directory,
    args.color_blind,
    args.prior,
    args.theme,
    args.task,
)

app.on_startup(setup_cb)

if args.prepare:
    setup_cb()
    save_callback(STATE)
    print("Data prepared and saved. Exiting.")
    exit(0)

ui.run(
    index,
    native=False,
    favicon="🚀",
    title="MALT",
    reload=True,
    show_welcome_message=False,
)  # set reload to False for prod
