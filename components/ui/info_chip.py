from nicegui import ui


def make_info_chip(text):
    with ui.icon("o_info", size="xs").classes("text-gray-500 cursor-default"):
        with ui.tooltip():
            ui.markdown(text)
