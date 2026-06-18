from functools import wraps

from nicegui import app


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
