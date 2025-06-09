from . import sg, GUIWidget


class GUIProjectSetting(GUIWidget):
    def __init__(self, host):
        super().__init__(host)
        self.configs = self.tomi_project.gui_configs
        self.tomi_project.update_gui_configs()
        self.configs.set_key_prefix('ProjectSetting@ProjectSetting')
        self.configs_layout = sg.Frame("", self.configs.get_config_layout(), visible=False, expand_x=True, expand_y=True)
        self.project_settings_layout = sg.pin(self.configs_layout, expand_x=True)

    def get_layout(self):
        return self.project_settings_layout

    def handle_event(self, event, values):
        if event.endswith('+Clicked'):
            elem = self.window[event.split('+')[0]]
            if not (isinstance(elem, sg.InputText) and elem.Widget.focus_get()):
                elem.set_focus(True)
                self.configs.update_entire_gui(self.window)
        elif event.endswith('+Returned'):
            self.configs.update_entire_gui(self.window)
        else:
            param_value = values[event] if event in values else None
            self.configs.handle_value_changes(event, param_value, self.window)