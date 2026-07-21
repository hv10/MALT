import json
from pathlib import Path

import pandas as pd
from nicegui import binding


class State:
    DATA = pd.DataFrame(
        columns=["fpth", "emb", "pos_x", "pos_y", "cls", "annot", "pr_cls"]
    )
    MODEL = None
    PAC_MODEL = None
    SEL_LABEL_INDCS = None
    PCA_MODEL = None
    META = {
        "classes": [],
        "cfg": None,
        "axis": {"a": 0, "b": 0, "c": 1},
        "multilabel": False,
        "note": "",
    }
    NEW_ROW = lambda fpth: (str(fpth), None, None, None, "", "i", [])  # noqa: E731
    SELECTED_ROWS = set()
    OUT_DIR = Path.cwd()
    COLORBLIND = False
    THEME = "dark"
    COEFF = None
    INTERCEPTS = None
    PLOT = None
    TABLE = None
    CLS_THRESHOLD = 0.0
    CLS_TYPE = "MIN_MARGIN"  # options: "MIN_MARGIN", "PROJECTION"
    CLS_ARGMAX = False
    HIDE_LABELED = False
    ADD_LABELS = True
    # History Management
    UNDO_STACK = []
    REDO_STACK = []
    MAX_HISTORY = 50
    PREVIEW_FUNC = None


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


async def update_selected_rows_table(state, grid):
    rows = await grid.get_selected_rows()
    rows = [r["fpth"] for r in rows]
    state.SELECTED_ROWS = rows


def update_selected_rows(state, rows):
    state.SELECTED_ROWS = [state.DATA.iloc[r].fpth for r in rows]
