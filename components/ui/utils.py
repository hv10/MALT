from nicegui import app
from functools import wraps
from .data_preview import data_preview
from .class_hist import class_hist
from .force_similarity_plot import force_similarity_plot
from .data_table import update_data_table
from .plot import update_plot

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

def refresh(state):
    data_preview.refresh(state)
    update_plot(state)
    class_hist.refresh(state)
    update_data_table(state)
    force_similarity_plot.refresh(state, sign_invariant=True)