from functools import wraps
from state import _snapshot, _restore
from ui.utils import with_loading_overlay, refresh

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