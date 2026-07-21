import numpy as np
import pandas as pd
from nicegui import ui
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder

from .data_mgmt import update_data_cls, update_data_positions
from .history import undoable
from .projection import update_projection
from .utils import refresh, with_loading_overlay


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
    pos_x, data_y = update_projection(
        state, coeffs, intercepts, project_X=True, run_pca=True
    )
    state.DATA = update_data_positions(state.DATA, pos_x, data_y)
    state.COEFF = coeffs
    state.INTERCEPTS = intercepts

    print("Update Machine Predictions")
    # these discern between exact and inclusive matching...
    X_full = np.stack(state.DATA["emb"])
    preds = model.decision_function(X_full)
    state.PREDS = preds
    state.SEL_LABEL_INDCS = sel_cls_ind
    binary_preds = (
        preds > state.CLS_THRESHOLD
    )  # this should probably guard against multilabel
    no_pred = np.sum(binary_preds, axis=1) == 0
    if state.CLS_ARGMAX:
        indices = np.argmax(binary_preds * preds, axis=1)
        mask = np.zeros_like(binary_preds, dtype=bool)
        mask[np.arange(len(binary_preds)), indices] = True
        binary_preds[~mask] = 0
    binary_preds = binary_preds[~no_pred & ~h_annot]
    state.DATA = update_data_cls(
        state.DATA,
        (~h_annot & ~no_pred),
        [tuple(sorted(labels[pred].to_list())) for pred in binary_preds],
    )
    refresh(state)


def get_sample_prediction(state, idx):
    preds = getattr(state, "PREDS", [])
    if len(preds) == 0:
        return []
    if isinstance(idx, str):
        idx = state.DATA.index[state.DATA["fpth"] == idx][0]
    return preds[idx]
