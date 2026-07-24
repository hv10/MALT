import ast
import json
import tempfile
import tomllib
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from nicegui import ui

from .embeddings import apply_emb_to_df
from .history import push_undo, undoable
from .projection import dir_of_max_variance, project_and_remove_direction
from .state import State
from .utils import refresh, with_loading_overlay


def update_data_positions(df, x, y):
    df["pos_x"] = x
    df["pos_y"] = y
    return df


def update_data_cls(df, indices, y_hat):
    df.loc[indices, "annot"] = "m"
    idx = df.index[indices]
    for i, yh in zip(idx, y_hat):
        df.at[i, "cls"] = yh
    return df


def load_folder(state, folder_pth):
    print(f"Loading Data from: {folder_pth}")
    file_paths = (
        p.resolve().relative_to(state.OUT_DIR)
        for p in folder_pth.glob("*")
        if p.suffix.lower() in state.META["cfg"]["embeddings"]["file_types"]
    )
    for pth in file_paths:
        state.DATA.loc[len(state.DATA.index)] = State.NEW_ROW(pth)
    if not state.DATA.empty:
        state.DATA = apply_emb_to_df(state.OUT_DIR, state.DATA, state)
        data_x, data_y = initialize_pos(state.DATA)
        state.DATA = update_data_positions(state.DATA, data_x, data_y)


def load_build_cfg(state, cfg):
    if cfg is None:
        cfg = tomllib.load(open(Path(__file__).parent / "build.toml", "rb"))
    state.META["cfg"] = cfg


def safe_load_cls(val):
    try:
        if pd.isna(val):
            return ()
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return ()


def load_prior_state(state, state_pth):
    print(f"Loading prior state from {state_pth}.")
    with zipfile.ZipFile(state_pth, "r") as zipf:
        with zipf.open("data.parquet") as f:
            with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
                tmp.write(f.read())
                tmp.flush()
                data_loaded = pd.read_parquet(
                    tmp.name
                )  # , converters={"cls": safe_load_cls})
        with zipf.open("emb.parquet") as f:
            with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
                tmp.write(f.read())
                tmp.flush()
                emb_loaded = pd.read_parquet(
                    tmp.name
                )  # , converters={"cls": safe_load_cls})
        emb_loaded["emb"] = emb_loaded["emb"].apply(lambda x: np.array(x))
        data_loaded["cls"] = data_loaded["cls"].apply(lambda x: tuple(x))
        state.DATA = pd.merge(data_loaded, emb_loaded)
        with zipf.open("meta.json") as fp:
            state.META = state.META | json.load(fp)
            load_build_cfg(state, state.META.get("cfg", None))
    print("Labels:", state.META["classes"])
    print(state.DATA.info())
    print(state.DATA.head())


@with_loading_overlay
def save_callback(state, unique=False):
    if unique:
        suffix = datetime.now().replace(microsecond=0).isoformat().replace(":", "-")
        fpth = state.OUT_DIR / f"state{suffix}.malt"
    else:
        fpth = state.OUT_DIR / "latest.malt"
    with zipfile.ZipFile(fpth, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            data_to_save = state.DATA.copy()
            data_to_save.drop(columns=["emb"], inplace=True)
            data_to_save.to_parquet(tmp.name)
            zipf.write(tmp.name, arcname="data.parquet")
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            data_to_save = state.DATA[["fpth", "emb"]].copy()
            data_to_save["emb"] = data_to_save["emb"].apply(lambda x: x.tolist())
            data_to_save.to_parquet(tmp.name)
            zipf.write(tmp.name, arcname="emb.parquet")
        zipf.writestr("meta.json", json.dumps(state.META, indent=2))
        if state.PAC_MODEL is not None:
            pac_model_dict = {
                l: {
                    "coeff": np.squeeze(
                        state.PAC_MODEL["model"].estimators_[i].coef_
                    ).tolist(),
                    "intercept": np.squeeze(
                        state.PAC_MODEL["model"].estimators_[i].intercept_
                    ).item(),
                }
                for i, l in enumerate(state.PAC_MODEL["labels"])
            }
            zipf.writestr("pac_model.json", json.dumps(pac_model_dict, indent=2))
    return fpth


def initialize_pos(df):
    X = np.vstack(df["emb"].values)
    w, data_x = dir_of_max_variance(X)
    projX = project_and_remove_direction(X, w)
    w2, data_y = dir_of_max_variance(projX)
    return data_x, data_y


@undoable
def update_row_cls(state, rowId, cls):
    if cls in state.META["classes"]:
        state.DATA.at[rowId, "cls"] = sorted(cls)
        state.DATA.at[rowId, "annot"] = "h"
        refresh(state)
    else:
        ui.notify(f"Class '{cls}' does not exist.")


@undoable
def set_class_for_selected(state, cls):
    if isinstance(cls, str):  # only a single label selected from dropdown
        cls = [cls]
    cls = tuple(sorted(cls))
    indices_to_update = state.DATA.index[state.DATA["fpth"].isin(state.SELECTED_ROWS)]
    for idx in indices_to_update:
        if state.ADD_LABELS:
            state.DATA.at[idx, "cls"] = tuple(
                sorted(set(state.DATA.at[idx, "cls"]) | set(cls))
            )
        else:
            state.DATA.at[idx, "cls"] = cls
        state.DATA.at[idx, "annot"] = "h"

    print(f"set class {cls} for selected rows.")
    refresh(state)


async def clear_class_for_selected(state):
    indices_to_update = state.DATA.index[state.DATA["fpth"].isin(state.SELECTED_ROWS)]
    if indices_to_update.empty:
        ui.notify("No rows selected.")
        return
    with ui.dialog() as dialog, ui.card():
        ui.label(f"Are you sure you want to clear {len(indices_to_update)} labels?")
        with ui.row():
            ui.button("Yes", on_click=lambda: dialog.submit("yes")).classes(
                "bg-red-700"
            )
            ui.button("Cancel", on_click=lambda: dialog.submit("cancel"))

    result = await dialog
    if result == "yes":
        push_undo(state)
        for idx in indices_to_update:
            state.DATA.at[idx, "cls"] = ()
            state.DATA.at[idx, "annot"] = "i"
        refresh(state)


def add_cls_from_file(state, content):
    classes = [l.strip() for l in content.split("\n")]
    lcls = len(state.META["classes"])
    for cls in classes:
        if cls and cls not in state.META["classes"]:
            state.META["classes"].append(cls)
    ui.notify(f"""Added {len(state.META["classes"]) - lcls} new classes.""")
    refresh(state)


def add_new_cls(state, inp):
    cls = inp.value
    if cls and cls not in state.META["classes"]:
        state.META["classes"].append(cls)
        inp.value = ""
        refresh(state)


def remove_class(state, cls):
    if cls and cls in state.META["classes"]:
        state.META["classes"].remove(cls)
        refresh(state)
    else:
        ui.notify(f"'{cls}' not in classes.")
