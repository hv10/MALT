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
#     "nicegui==2.17.0",

import argparse as ap
import ast
import asyncio
from glob import glob
import itertools
import json
import logging
import os
import tempfile
import zipfile
from datetime import datetime
from functools import partial, wraps
from pathlib import Path
from time import perf_counter, time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import torch
from nicegui import ElementFilter, app, run, ui
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

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


# ==================== STATE Data ====================


class State:
    DATA = pd.DataFrame(
        columns=["fpth", "emb", "pos_x", "pos_y", "cls", "annot", "pr_cls"]
    )
    MODEL = None
    PAC_MODEL = None
    PCA_MODEL = None
    META = {
        "classes": [],
        "model": None,
        "axis": {"a": 0, "b": 0, "c": 1},
        "multilabel": False,
        "note": "",
    }
    NEW_ROW = lambda fpth: (str(fpth), None, None, None, "", "i", [])  # noqa: E731
    SELECTED_ROWS = set()
    OUT_DIR = None
    COLORBLIND = False
    COEFF = None
    INTERCEPTS = None
    PLOT = None
    TABLE = None
    CLS_THRESHOLD = 0.0
    CLS_TYPE = "MIN_MARGIN"  # options: "MIN_MARGIN", "PROJECTION"
    HIDE_LABELED = False
    ADD_LABELS = True
    # History Management
    UNDO_STACK = []
    REDO_STACK = []
    MAX_HISTORY = 50


STATE = None


# ==================== History Management ====================


def with_loading_overlay(func):
    debounce = {"timer": None}

    @wraps(func)
    def wrapper(*args, **kwargs):
        overlay = getattr(with_loading_overlay, "overlay", None)
        if overlay is not None:
            if debounce["timer"] is not None:
                debounce["timer"].cancel()
            debounce["timer"] = app.timer(
                0.3,
                once=True,
                callback=lambda: (
                    overlay.set_visibility(True) if overlay is not None else None
                ),
            )
        try:
            result = func(*args, **kwargs)
        finally:
            if overlay is not None:
                debounce["timer"].cancel()
                debounce["timer"] = None
                overlay.set_visibility(False)
        return result

    return wrapper


def _snapshot(state):
    return {
        "DATA": state.DATA.copy(),
        "META": json.loads(json.dumps(state.META)),  # deep copy
        "COEFF": [c.copy() for c in state.COEFF] if state.COEFF is not None else None,
        "INTERCEPTS": list(state.INTERCEPTS) if state.INTERCEPTS is not None else None,
    }


def _restore(state, snapshot):
    state.DATA = snapshot["DATA"]
    state.META = snapshot["META"]
    state.COEFF = snapshot["COEFF"]
    state.INTERCEPTS = snapshot["INTERCEPTS"]


def push_undo(state):
    state.UNDO_STACK.append(_snapshot(state))
    if len(state.UNDO_STACK) > state.MAX_HISTORY:
        state.UNDO_STACK.pop(0)
    state.REDO_STACK.clear()


@with_loading_overlay
def undo(state):
    if not state.UNDO_STACK:
        ui.notify("Nothing to undo.")
        return
    state.REDO_STACK.append(_snapshot(state))
    _restore(state, state.UNDO_STACK.pop())
    refresh(state)


@with_loading_overlay
def redo(state):
    if not state.REDO_STACK:
        ui.notify("Nothing to redo.")
        return
    state.UNDO_STACK.append(_snapshot(state))
    _restore(state, state.REDO_STACK.pop())
    refresh(state)


def undoable(func):
    @wraps(func)
    def wrapper(state, *args, **kwargs):
        push_undo(state)
        result = func(state, *args, **kwargs)
        return result

    return wrapper


# ==================== HALFSPACE Projection ====================


def project_and_remove_direction(X, v):
    """
    Project each vector in X onto vector v and then remove this component from X.
    """
    v = v / np.linalg.norm(v)  # ensure |v| = 1
    projection_onto_v = np.dot(X, v)[:, np.newaxis] * v
    print(projection_onto_v.shape)
    X_residual = X - np.squeeze(projection_onto_v)  # remove projection
    return X_residual


def dir_of_max_variance(X, count=1):
    """
    Perform PCA to find the `count` first principal component (nD PCA).
    Optimized using Singular Value Decomposition (SVD).
    """
    n = len(X)
    fit_samples = min(n, max(2000, int(0.05 * n)))

    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=fit_samples, replace=False)

    X_mean = X.mean(axis=0)
    X_centered = X - X_mean

    _, _, w = randomized_svd(
        X_centered[idx], n_components=count, n_iter=2, random_state=42
    )
    Vt = w.T
    X_projected = X_centered @ Vt
    return Vt, X_projected


def orthonormalize(vectors, return_indices=False):
    """
    Orthonormalize a set of vectors (columns of a matrix) using Gram-Schmidt.
    """
    Q = []
    keep = []
    for i, v in enumerate(vectors):
        v = v.copy()
        for q in Q:
            v -= np.dot(q, v) * q
        if np.linalg.norm(v) > 1e-10:
            keep.append(i)
            Q.append(v / np.linalg.norm(v))
    if return_indices:
        return np.array(Q), np.array(keep)
    return np.array(Q)


def project_onto_orthogonal_complement(X, coeffs):
    """
    Removes the components of X along the directions in coeffs simultaneously.
    """
    # Orthonormalize first
    C = orthonormalize(coeffs)  # shape (num_coeffs, n_features)
    C = C.T  # now shape (n_features, num_coeffs)

    # Projection matrix onto subspace spanned by C: P = C @ C.T
    # Projection onto orthogonal complement: I - P
    X_proj = (X @ C) @ C.T
    X_residual = X - X_proj
    return X_residual


# ==================== Embedding & Huggingface ====================


def load_model(state, model_name="microsoft/resnet50"):
    if state.MODEL is None:
        state.MODEL = {
            # Load the model and processor (which will handle image pre-processing)
            "processor": AutoProcessor.from_pretrained(model_name),
            "model": AutoModel.from_pretrained(model_name),
        }
    return state.MODEL


def embed_image(filepath, model):
    """
    Embeds an image as a vector using a Hugging Face model specified by its name.
    """
    image = Image.open(filepath).convert("RGB")
    inputs = model["processor"](images=image, return_tensors="pt")
    with torch.no_grad():  # forward pass
        outputs = model["model"](**inputs)

    if hasattr(outputs, "pooler_output"):  # extr. emb
        embedding = outputs.pooler_output.squeeze()
    else:
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

    return embedding.numpy()


def apply_emb_to_df(out_dir, df, model):
    tqdm.pandas(desc="Embedding Images")

    def helper(row):
        row["emb"] = embed_image(out_dir / row["fpth"], model)
        return row

    return df.progress_apply(helper, axis=1)


# ==================== PA Classifier ====================
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


async def save_async_cb(state, btn, unique=False):
    ui.notify("Saving...")
    btn.disable()
    ui.query("body").classes("loading")
    fpth = await run.io_bound(save_callback, state, unique)
    ui.page_title(f"MALT - {fpth.stem}")
    btn.enable()
    ui.query("body").classes(remove="loading")
    ui.notify("Save Successful")


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


def initialize_pos(df):
    X = np.vstack(df["emb"].values)
    w, data_x = dir_of_max_variance(X)
    projX = project_and_remove_direction(X, w)
    w2, data_y = dir_of_max_variance(projX)
    return data_x, data_y


def remove_file(state, i):
    state.DATA.drop(index=i, inplace=True)
    refresh(state)


def prepare_data_exact_match(state):
    h_annot = state.DATA["annot"] == "h"
    X = np.stack(state.DATA.loc[h_annot, "emb"])
    y, labels = pd.factorize(state.DATA.loc[h_annot, "cls"])
    return X, y, labels


def prepare_data(state):
    h_annot = state.DATA["annot"] == "h"
    df = state.DATA.loc[h_annot, ["emb", "cls"]].copy()

    # explode list-valued 'cls' so each class label gets its own row
    df = df.explode("cls", ignore_index=True)

    # stack embeddings and factorize labels
    X = np.stack(df["emb"])
    y, labels = pd.factorize(df["cls"])
    return X, y, labels


@with_loading_overlay
@undoable
def update_pa_clf(state, sel_cls):
    h_annot = state.DATA["annot"] == "h"
    if state.DATA.loc[h_annot].empty:
        ui.notify("You need to annotate at least one element, in two distinct classes.")
        return
    X, y, labels = prepare_data(state)
    print("Fit PAC Clf.")
    # now we fit a PAC Clf for each _single_ label
    model = OneVsRestClassifier(
        SGD(
            random_state=0,
            loss="hinge",
            penalty=None,
            learning_rate="pa1",
            eta0=1.0,
            fit_intercept=True,
        )
    )
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="warn")
    y_enc = encoder.fit_transform(y.reshape(-1, 1))
    model.fit(X, y_enc)
    state.PAC_MODEL = {
        "model": model,
        "labels": labels,
    }
    # this will now have one classifier per each label

    print("PCA & Plotting Prep")  # TODO: make this faster
    # now we want to do the following:
    # for all selected classes we want to know the relevant classifier(s)
    # then we need to use the full thing and calculate the coefficients
    # we will need to modify the update_projection function
    if sel_cls:
        label_to_pos = {lbl: i for i, lbl in enumerate(labels)}
        missing = set(sel_cls) - label_to_pos.keys()
        if missing:
            print(f"Skipping: no datapoints yet for {missing}")
            return
        sel_cls_ind = [label_to_pos[c] for c in sel_cls]
    else:
        sel_cls_ind = [0]
    coeffs = [np.squeeze(model.estimators_[sidx].coef_) for sidx in sel_cls_ind]
    intercepts = [
        np.squeeze(model.estimators_[sidx].intercept_) for sidx in sel_cls_ind
    ]
    # fix directionality by ensuring the mean of the positive class is in the positive direction
    update_projection(state, coeffs, intercepts, project_X=True, run_pca=True)

    print("Update Machine Predictions")
    # these discern between exact and inclusive matching...
    X_full = np.stack(state.DATA["emb"])
    preds = model.decision_function(X_full)
    binary_preds = (
        preds > state.CLS_THRESHOLD
    )  # this should probably guard against multilabel
    no_pred = np.sum(binary_preds, axis=1) == 0
    binary_preds = binary_preds[~no_pred & ~h_annot]
    state.DATA = update_data_cls(
        state.DATA,
        (~h_annot & ~no_pred),
        [tuple(sorted(labels[pred].to_list())) for pred in binary_preds],
    )
    refresh(state)


def update_projection(state, coeffs, intercepts, project_X=True, run_pca=True):
    # project our data onto the (d-1)-simplex defined by coeffs[]
    # either:
    ## data_x will be the closest value to zero of X@coeff (minimal magnitude)
    ## or data_x will be the magnitude of the projection of X onto
    ## the span of coeffs negated (so that higher is more similar)
    # then remove its influence
    if project_X:
        X = np.stack(state.DATA["emb"])
        if state.CLS_TYPE == "MIN_MARGIN":
            coeffs_mat = np.array(coeffs)  # (k, d)
            norms = np.linalg.norm(coeffs_mat, axis=1, keepdims=True)  # (k, 1)
            px = np.transpose(
                (X @ coeffs_mat.T + np.array(intercepts)) / norms.T
            )  # (k, n) in one BLAS call
            min_idx = np.argmin(np.abs(px), axis=0)
            pos_x = px[min_idx, np.arange(px.shape[1])]
        else:  # PROJECTION
            Q, b_keep = orthonormalize(coeffs, return_indices=True)  # orthonormal span
            X_d = X @ Q.T + np.array(intercepts)[b_keep]  # project onto span
            pos_x = 1 / (1 + np.exp(-np.sum(X_d, axis=1)))
            pos_x = 2 * (pos_x - 0.5)
        projX = project_onto_orthogonal_complement(X, coeffs)
        state.PROJ_X = projX
    else:
        pos_x = state.DATA["pos_x"].to_numpy()

    # now we will reproject onto a 3D PCA Projection
    # which we will then reweigh according to our ternary.
    if run_pca:
        w, _ = dir_of_max_variance(state.PROJ_X, count=3)
        state.PCA_MODEL = w
    else:
        w = state.PCA_MODEL
    # w are our principal components
    # we now construct the axis by doing a linear combination of the columns of w
    P_ax = (
        state.META["axis"]["c"] * w[:, 0]
        + state.META["axis"]["a"] * w[:, 1]
        + state.META["axis"]["b"] * w[:, 2]
    )
    data_y = np.dot(state.PROJ_X, P_ax)
    state.DATA = update_data_positions(state.DATA, pos_x, data_y)
    state.COEFF = coeffs
    state.INTERCEPTS = intercepts


@ui.refreshable
def force_similarity_plot(
    state,
    sign_invariant=False,
    n_iter=500,
    repulsion_strength=1.0,
    attraction_strength=1.0,
    seed=42,
):
    if state.PAC_MODEL is None:
        return go.Figure()  # no model, no plot
    W = np.stack(
        [np.squeeze(est.coef_) for est in state.PAC_MODEL["model"].estimators_]
    )
    labels = state.PAC_MODEL["labels"]

    np.random.seed(seed)
    n = W.shape[0]
    # ---- Normalize vectors ----
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    # ---- Cosine similarity ----
    S = W @ W.T
    if sign_invariant:
        S = S**2
    np.fill_diagonal(S, 0.0)
    # ---- Initialize random 2D positions ----
    pos = np.random.randn(n, 2)
    # ---- Force-directed layout ----
    for _ in range(n_iter):
        delta = pos[:, None, :] - pos[None, :, :]  # pairwise differences
        distance = np.linalg.norm(delta, axis=2) + 1e-6
        # Repulsion (Coulomb-like)
        repulsion = repulsion_strength / distance**2
        rep_force = (delta / distance[:, :, None]) * repulsion[:, :, None]
        # Attraction weighted by similarity
        attraction = attraction_strength * S
        attr_force = -(delta) * attraction[:, :, None]
        total_force = rep_force.sum(axis=1) + attr_force.sum(axis=1)
        pos += 0.01 * total_force  # small step size
    # ---- Build Plotly figure ----
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pos[:, 0],
            y=pos[:, 1],
            mode="markers+text",
            marker=dict(size=12),
            text=labels,
            textposition="top center",
        )
    )
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=5, r=5, t=15, b=5),
    )
    ui.label(
        "Classifier Similarity (Sign-Invariant)"
        if sign_invariant
        else "Classifier Similarity"
    )
    ui.plotly(fig).classes("w-full")


def add_cls_from_file(state, content):
    classes = [l.strip() for l in content.split("\n")]
    lcls = len(state.META["classes"])
    for cls in classes:
        if cls and cls not in state.META["classes"]:
            state.META["classes"].append(cls)
    ui.notify(f"""Added {len(state.META["classes"]) - lcls} new classes.""")
    plot_controls.refresh()
    label_controls.refresh()


def add_new_cls(state, inp):
    cls = inp.value
    if cls and cls not in state.META["classes"]:
        state.META["classes"].append(cls)
        inp.value = ""
    plot_controls.refresh()
    label_controls.refresh()


def remove_class(state, cls):
    if cls and cls in state.META["classes"]:
        state.META["classes"].remove(cls)
        plot_controls.refresh()
        label_controls.refresh()
    else:
        ui.notify(f"'{cls}' not in classes.")


# ==================== DATA Mgmt ====================


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


def load_folder(state, folder_pth: Path):
    print(f"Loading Images from: {folder_pth}")
    file_paths = (
        p.resolve().relative_to(state.OUT_DIR)
        for p in folder_pth.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    for pth in file_paths:
        state.DATA.loc[len(state.DATA.index)] = State.NEW_ROW(pth)
    if not state.DATA.empty:
        state.DATA = apply_emb_to_df(state.OUT_DIR, state.DATA, load_model(state))
        data_x, data_y = initialize_pos(state.DATA)
        state.DATA = update_data_positions(state.DATA, data_x, data_y)
        # refresh(state)


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


# ==================== GUI ====================


def get_color(number: int) -> str:
    palette = plt.get_cmap("Spectral").colors
    return plt.colors.to_hex(palette[hash(number) % len(palette)])


def refresh(state):
    data_preview.refresh(state)
    update_plot(state)
    class_hist.refresh(state)
    update_data_table(state)
    force_similarity_plot.refresh(state, sign_invariant=True)


def get_table_data(state):
    sdf = state.DATA[["fpth", "cls", "annot"]].copy()
    sdf["cls"] = sdf["cls"].map(format_cls_label)
    data = sdf.to_dict(orient="records")
    return data


def data_table(state):
    ag_grid = ui.aggrid(
        {
            "columnDefs": [
                {
                    "headerName": "Filename",
                    "field": "fpth",
                    "filter": "agTextColumnFilter",
                    "floatingFilter": True,
                    # "flex": 1,
                },
                {
                    "headerName": "Class",
                    "field": "cls",
                    # "flex": 1,
                    "floatingFilter": True,
                    "filter": "agTextColumnFilter",
                },
                {
                    "headerName": "Annotator",
                    "field": "annot",
                    # "flex": 1,
                    "floatingFilter": True,
                    "filter": "agTextColumnFilter",
                },
            ],
            "rowData": get_table_data(state),
            "rowSelection": "multiple",
            "readOnlyEdit": True,
        },
        theme="balham",
    )
    ag_grid.classes("h-[66svh] w-100")
    # ag_grid.on(
    #     "cellEditRequest",
    #     lambda event: update_row_cls(
    #         state, int(event.args["rowId"]), event.args["newValue"]
    #     ),
    # )
    ag_grid.on("selectionChanged", lambda: update_selected_rows_table(state, ag_grid))
    state.TABLE = ag_grid


def update_data_table(state):
    if state.TABLE:
        print("Updating the table...")
        state.TABLE.options |= {"rowData": get_table_data(state)}


@undoable
def update_row_cls(state, rowId, cls):
    if cls in state.META["classes"]:
        state.DATA.at[rowId, "cls"] = sorted(cls)
        state.DATA.at[rowId, "annot"] = "h"
        refresh(state)
        # update_plot(state)
    else:
        ui.notify(f"Class '{cls}' does not exist.")


async def update_selected_rows_table(state, grid):
    rows = await grid.get_selected_rows()
    rows = [r["fpth"] for r in rows]
    state.SELECTED_ROWS = rows
    data_preview.refresh(state)


def format_cls_label(cls):
    if not cls:
        return "<unlabeled>"
    return ", ".join(cls)


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


def update_selected_rows(state, rows):
    state.SELECTED_ROWS = [state.DATA.iloc[r].fpth for r in rows]
    data_preview.refresh(state)
    # update_data_table(state)


@ui.refreshable
def data_preview(state):
    sources = (
        [str(state.OUT_DIR / r) for r in state.SELECTED_ROWS]
        if state.SELECTED_ROWS
        else []
    )
    mask = state.DATA.fpth.isin(state.SELECTED_ROWS)
    subset = state.DATA[mask].set_index("fpth")
    labels = [
        [r, format_cls_label(subset.loc[r, "cls"]), subset.loc[r, "annot"]]
        for r in state.SELECTED_ROWS
    ]

    def make_lbl(el):
        with ui.label(el[0]).classes("font-bold"):
            ui.tooltip(el[0])
        ui.label(str(el[1]))
        ui.chip(
            str(el[2]).capitalize(),
            color="green" if el[2] == "h" else "blue" if el[2] == "m" else "default",
        ).classes("absolute right-2 bottom-2 z-[10]")

    current_idx = {"value": 0}  # mutable container so closure can update it

    with ui.dialog() as dialog:
        with ui.card().classes("p-0 overflow-scroll max-w-[95vw] max-h-[95vh]"):
            zoomed = ui.image("").classes("h-[90vh] w-[90vh] object-contain")
            modal_label = ui.row().classes(
                "p-2 text-sm absolute bottom-0 left-0 bg-black bg-opacity-50 text-white w-full"
            )
            ui.button(icon="close", on_click=dialog.close).classes(
                "absolute top-2 right-2"
            )

    def open_zoom(idx: int):
        current_idx["value"] = idx
        zoomed.set_source(sources[idx])
        modal_label.clear()
        with modal_label:
            make_lbl(labels[idx])
        dialog.open()

    def on_key(e):
        if not dialog.value:  # dialog not open
            return
        if not e.action.keydown or e.action.repeat:
            return
        if e.key == "ArrowRight":
            current_idx["value"] = (current_idx["value"] + 1) % len(sources)
        elif e.key == "ArrowLeft":
            current_idx["value"] = (current_idx["value"] - 1) % len(sources)
        zoomed.set_source(sources[current_idx["value"]])
        modal_label.clear()
        with modal_label:
            make_lbl(labels[current_idx["value"]])

    ui.keyboard(on_key=on_key).on(
        "key",
        js_handler="""(e) => {
        if ((e.key === 'ArrowRight' || e.key === 'ArrowLeft') && e.action === 'keydown') {
            emit(e);
            e.event.preventDefault();
        }
        }""",
    )

    PAGINATION_STATE = {"ps": 25, "p": 1}

    @ui.refreshable
    @with_loading_overlay
    def card_grid():
        p = PAGINATION_STATE["p"]
        ps = PAGINATION_STATE["ps"]
        with ui.row().classes("w-full"):
            for i in range((p - 1) * ps, min((p - 1) * ps + ps, len(sources))):
                with ui.card().classes("w-2/12 overflow-hidden"):
                    make_lbl(labels[i])
                    ui.image(sources[i]).classes(
                        "cursor-pointer hover:opacity-90 transition-opacity max-w-full"
                    ).on("click", lambda idx=i: open_zoom(idx))

    @ui.refreshable
    def pagination():
        ui.pagination(
            1,
            np.ceil(len(state.SELECTED_ROWS) / PAGINATION_STATE["ps"]),
            on_change=card_grid.refresh,
            direction_links=True,
        ).bind_value(PAGINATION_STATE, "p")

    if not state.SELECTED_ROWS:
        ui.label("Selected Data will be previewed here.").classes("italic")
    else:
        ui.label("Selected Data").classes("center")
        with ui.row().classes("w-full items-center justify-between"):
            pagination()
            ui.radio(
                [25, 50, 100],
                on_change=lambda: (card_grid.refresh(), pagination.refresh()),
            ).props("inline").bind_value(PAGINATION_STATE, "ps")
        card_grid()
        with ui.row().classes("w-full items-center justify-between"):
            pagination()


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
    update_projection(
        state,
        state.COEFF,
        state.INTERCEPTS,
        project_X=False,
        run_pca=False,
    )
    update_plot(state)


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


def make_info_chip(text):
    with ui.icon("o_info", size="xs").classes("text-gray-500 cursor-default"):
        with ui.tooltip():
            ui.markdown(text)


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
            rows=[{"name": l} for l in state.META["classes"]],
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


@ui.refreshable
def class_hist(state):
    # Explode 'cls' so each label becomes a separate row
    cls_exploded = state.DATA["cls"].explode().astype(str)

    # Count occurrences of each label
    label_counts = (
        cls_exploded.value_counts()
        .reindex(state.META["classes"], fill_value=0)
        .reset_index()
    )
    label_counts.columns = ["label", "count"]

    # Plot using Plotly Express
    fig = px.bar(label_counts, x="count", y="label", orientation="h")
    ui.label("Class Frequency")
    ui.plotly(fig).classes("w-full")


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
    get_progress = lambda d: d["annot"].value_counts().get("h", 0) / len(d)
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
    app.add_static_files("/images", state.OUT_DIR)
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
            data_preview(state)
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
    with_loading_overlay.overlay = make_overlay()
    ui.keyboard(on_key=global_handle_key)


# ==================== MAIN ====================
def setup_state(model, directory, color_blind, prior, fullscreen, theme, task):
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
            load_model(state, model)
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
        ui.page_title(f"MALT")
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
parser.add_argument("-f", "--fullscreen", action="store_true")
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
    args.fullscreen,
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
    native=True,
    favicon="🚀",
    title="MALT",
    reload=True,
    show_welcome_message=False,
)  # set reload to False for prod
