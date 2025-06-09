from . import sg


class GUIWidget:
    def __init__(self, host):
        self.host_editor = host

    def get_layout(self):
        return sg.Col([])

    @property
    def visible(self) -> bool:
        return self.get_layout().visible

    @property
    def window(self) -> sg.Window:
        return self.host_editor.window

    @property
    def tomi_project(self):
        return self.host_editor.tomi_project

    @property
    def tomi_engine(self):
        return self.host_editor.tomi_engine

    def handle_event(self, event, values):...
