import numpy as np
from nicegui import ui, binding
from ..utils import format_cls_label, with_loading_overlay

class data_preview(ui.element):
    selected_rows = binding.BindableProperty(on_change=lambda sender, value: sender.update_selected_rows(value))
    sources = []

    def __init__(self, state):
        super().__init__()
        self.state_ref = state
        self.selected_rows = set()
        self.pagination_state = {"ps": 25, "p": 1}
        self.current_idx = 0
        self.ui()
    
    def update_selected_rows(self, rows):
        self.selected_rows = rows
        self.sources = (
            [str(self.state_ref.OUT_DIR / r) for r in self.selected_rows]
            if self.selected_rows
            else []
        )
        mask = self.state_ref.DATA.fpth.isin(self.selected_rows)
        subset = self.state_ref.DATA[mask].set_index("fpth")
        self.labels = [
            [r, format_cls_label(subset.loc[r, "cls"]), subset.loc[r, "annot"]]
            for r in self.selected_rows
        ]
        self.ui.refresh()

    def make_lbl(self, el):
        with ui.label(el[0]).classes("font-bold"):
            ui.tooltip(el[0])
        ui.label(str(el[1]))
        ui.chip(
            str(el[2]).capitalize(),
            color="green" if el[2] == "h" else "blue" if el[2] == "m" else "default",
        ).classes("absolute right-2 bottom-2 z-[10]")

    def open_zoom(self,idx):
        self.current_idx = idx
        self.zoomed.set_source(self.sources[idx])
        self.modal_label.clear()
        with self.modal_label:
            self.make_lbl(self.labels[idx])
        self.dialog.open()

    def on_key_dialog(self,e):
        if not self.dialog.value:  # dialog not open
            return
        if not e.action.keydown or e.action.repeat:
            return
        if e.key == "ArrowRight":
            self.current_idx = self.current_idx + 1 % len(self.sources)
        elif e.key == "ArrowLeft":
            self.current_idx = self.current_idx - 1 % len(self.sources)
        self.zoomed.set_source(self.sources[self.current_idx])
        self.modal_label.clear()
        with self.modal_label:
            self.make_lbl(self.labels[self.current_idx])

    @ui.refreshable
    @with_loading_overlay
    def card_grid(self):
        p = self.pagination_state["p"]
        ps = self.pagination_state["ps"]
        with ui.row().classes("w-full"):
            for i in range((p - 1) * ps, min((p - 1) * ps + ps, len(self.sources))):
                with ui.card().classes("w-2/12 overflow-hidden"):
                    self.make_lbl(self.labels[i])
                    ui.image(self.sources[i]).classes(
                        "cursor-pointer hover:opacity-90 transition-opacity max-w-full"
                    ).on("click", lambda idx=i: self.open_zoom(idx))

    @ui.refreshable
    def pagination(self):
        ui.pagination(
            1,
            np.ceil(len(self.selected_rows) / self.pagination_state["ps"]),
            on_change=self.card_grid.refresh,
            direction_links=True,
        ).bind_value(self.pagination_state, "p")

    @ui.refreshable_method
    def ui(self):
        self.dialog = ui.dialog()
        with self.dialog:
            with ui.card().classes("p-0 overflow-scroll max-w-[95vw] max-h-[95vh]"):
                self.zoomed = ui.image("").classes("h-[90vh] w-[90vh] object-contain")
                self.modal_label = ui.row().classes(
                    "p-2 text-sm absolute bottom-0 left-0 bg-black bg-opacity-50 text-white w-full"
                )
                ui.button(icon="close", on_click=self.dialog.close).classes(
                    "absolute top-2 right-2"
                )

        ui.keyboard(on_key=self.on_key_dialog).on(
            "key",
            js_handler="""(e) => {
            if ((e.key === 'ArrowRight' || e.key === 'ArrowLeft') && e.action === 'keydown') {
                emit(e);
                e.event.preventDefault();
            }
            }""",
        )

        if not self.selected_rows:
            ui.label("Selected Data will be previewed here.").classes("italic")
        else:
            ui.label("Selected Data").classes("center")
            with ui.row().classes("w-full items-center justify-between"):
                self.pagination()
                ui.radio(
                    [25, 50, 100],
                    on_change=lambda: (self.card_grid.refresh(), self.pagination.refresh()),
                ).props("inline").bind_value(self.pagination_state, "ps")
            self.card_grid()
            with ui.row().classes("w-full items-center justify-between"):
                self.pagination()
