from nicegui import ui


def prediction_viz(pred_arr, state, all=False):
    if len(pred_arr) == 0:
        ui.label("No predictions").classes("italic")
        return
    pred_arr = list(pred_arr)
    labels = state.PAC_MODEL["labels"] if state.PAC_MODEL is not None else []
    if state.SEL_LABEL_INDCS and not all:
        pred_arr = [pred_arr[i] for i in state.SEL_LABEL_INDCS]
        labels = [labels[i] for i in state.SEL_LABEL_INDCS]
    with ui.column().classes("w-full m-0 p-0"):
        ui.echart(
            {
                "yAxis": {"type": "category", "data": list(labels)},
                "xAxis": {
                    "type": "value",
                    "min": min(-1, min(pred_arr)),
                    "max": max(1, max(pred_arr)),
                },
                "series": [{"data": pred_arr, "type": "bar", "color": pred_arr}],
                "grid": {"left": 0, "top": 0, "right": 0, "bottom": 0},
                "visualMap": {
                    "show": False,
                    "min": -1,
                    "max": 1,
                    "dimension": 0,
                    "inRange": {
                        "color": [
                            "#FD665F",
                            "#CCCCCC",
                            "#58D9F9",
                        ]  # red to gray to blue
                    },
                },
            },
            renderer="svg",
        ).classes(f"w-full max-h-{8 * len(labels)}")
