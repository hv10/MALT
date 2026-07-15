from nicegui import ui

def make_image_sample_preview(obj, source):
    return ui.image(source).classes(
        "cursor-pointer hover:opacity-90 transition-opacity max-w-full"
    )

def make_image_detail_preview(obj, source):
    return ui.image(source).classes("h-full w-full object-contain")
