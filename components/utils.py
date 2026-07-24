import csv
from functools import wraps
from pathlib import Path

from nicegui import app


def is_numeric(s):
    try:
        float(s)
    except ValueError:  # Failed
        return False
    else:  # Succeeded
        return True

def has_header(path):
    preview = load_table(
        path,
        header=None,
        nrows=10,
    )

    if len(preview) < 2:
        return False

    first_row = preview.iloc[0]
    remaining = preview.iloc[1:]

    first_numeric_ratio = first_row.map(is_numeric).mean()
    remaining_numeric_ratio = remaining.map(is_numeric).mean().mean()

    return 0 if first_numeric_ratio < remaining_numeric_ratio else None

def load_table(source, **kwargs):
    import pandas as pd
    source = Path(source)
    suffix = source.suffix.lower()

    match suffix:
        case ".csv":
            return pd.read_csv(source, **kwargs)

        case ".tsv":
            return pd.read_csv(source, sep="\t", **kwargs)

        case ".xls" | ".xlsx" | ".xlsm" | ".xlsb":
            return pd.read_excel(source, **kwargs)

        case _:
            raise ValueError(f"Unsupported file type: {suffix}")

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


def refresh(state, group=None):
    to_update = getattr(refresh, "to_update", set())
    for up_func in to_update:
        if group is None or up_func[1] == group:
            up_func[0](state)


def register_refresh(elements, group=None):
    to_update = getattr(refresh, "to_update", set())
    elements = [(e, group) for e in elements]
    refresh.to_update = to_update.union(set(elements))  # type: ignore


def format_cls_label(cls):
    if not cls:
        return "<unlabeled>"
    return ", ".join(cls)
