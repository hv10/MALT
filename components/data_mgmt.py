import ast
import json
import tempfile
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd

from .embeddings import apply_emb_to_df
from .state import State
from .utils import with_loading_overlay, refresh
from .history import undoable


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
    print(f"Loading Images from: {folder_pth}")
    file_paths = (
        p.resolve().relative_to(state.OUT_DIR)
        for p in folder_pth.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    for pth in file_paths:
        state.DATA.loc[len(state.DATA.index)] = State.NEW_ROW(pth)
    if not state.DATA.empty:
        state.DATA = apply_emb_to_df(state.OUT_DIR, state.DATA)
        data_x, data_y = initialize_pos(state.DATA)
        state.DATA = update_data_positions(state.DATA, data_x, data_y)


def safe_load_cls(val):
    try:
        if pd.isna(val):
            return ()
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return ()


def load_prior_state(state, state_pth):
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
            print(state.META["classes"])
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
