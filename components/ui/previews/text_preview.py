from pathlib import Path
from nicegui import ui
from functools import partial


def make_text_sample_preview(obj, source):
    source = Path(source)
    text = source.read_text(encoding="utf-8").strip()[1:137]
    text += "…"
    ui.label(text).classes("overflow-hidden font-mono w-full text-ellipsis").style("white-space: pre-line")


def make_text_detail_preview(obj, source):
    source = Path(source)
    text = source.read_text(encoding="utf-8").strip()
    ui.label(text).classes("p-2 overflow-scroll font-mono max-w-2xl").style("white-space: pre-line")


if __name__ in {"__main__", "__mp_main__"}:
    ui.element("hr").classes("border-1 border-black w-64 center")
    with ui.card().classes("w-32 h-32"):
        make_text_sample_preview(None, "PLAN.md")
    ui.element("hr").classes("border-1 border-black border-dashed w-32 center")
    with ui.card().classes("w-32 h-32"):
        make_text_sample_preview(None, "build.toml")
    ui.element("hr").classes("border-1 border-black w-64 center")
    make_text_detail_preview(None, "PLAN.md")
    ui.element("hr").classes("border-1 border-black border-dashed w-32 center")
    make_text_detail_preview(None, "build.toml")
    ui.run()
