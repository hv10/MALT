from ..utils import format_cls_label
from ..state import update_selected_rows_table
from nicegui import ui

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
    ag_grid.on("selectionChanged", lambda: update_selected_rows_table(state, ag_grid))
    state.TABLE = ag_grid


def update_data_table(state):
    if state.TABLE:
        print("Updating the table...")
        state.TABLE.options |= {"rowData": get_table_data(state)}
