from tomi import Project, REAPERController, TOMIEngine
from . import GUIProjectSetting, GUIProjectEditPane, sg


class TOMIEditor:
    def __init__(self, host: Project | TOMIEngine):
        self.default_font = ("IBM Plex Mono", 16)  # Damn this is elegant
        sg.theme('Reddit')
        sg.set_options(font=self.default_font)
        if isinstance(host, TOMIEngine):
            self.tomi_engine = host
            self.tomi_project = host.project
        elif isinstance(host, Project):
            self.tomi_engine = None
            self.tomi_project = host
        else:
            raise ValueError("'host' must be a Project instance or a TOMIEngine instance.")
        self.project_settings = GUIProjectSetting(self)
        self.project_edit_pane = GUIProjectEditPane(self)
        self.main_layout = [
            [
                sg.Button('Song Settings', enable_events=True, key='-SONGSETTINGS-', button_color='orange'),
                sg.Push(),
                sg.Button('Open Reaper', enable_events=True, key='-OPENREAPER-'),
                sg.Button('Sync Project', enable_events=True, key='-SYNCPROJECT-'),
                sg.Button('Settings', enable_events=True, key='-SETTINGS-')
            ],
            [self.project_settings.get_layout()],
            [
                sg.TabGroup([[
                    sg.Tab('Project', self.project_edit_pane.get_layout()),
                    # sg.Tab('LLM', [[]]),
                ]], expand_x=True, expand_y=True),
            ]
        ]
        self.window = sg.Window(title="TOMI Editor (Demo)", layout=self.main_layout, margins=(20, 20), finalize=True, resizable=True)

    def frequent_check_handler(self):
        self.project_edit_pane.monitor_project_pane_size_change_handler()

    def run(self):
        self.project_settings.configs.bind_configs_to_button()
        self.project_edit_pane.init_project()
        while True:
            event, values = self.window.read(timeout=100)
            if event != '__TIMEOUT__':
                print(event)
            if event != 'Graph@':
                self.project_edit_pane.node_graph.last_graph_mouse_pos = None
            if event == sg.WIN_CLOSED:
                break
            elif event == '-SONGSETTINGS-':
                self.project_settings.configs_layout.update(visible=not self.project_settings.configs_layout.visible)
            elif event == '-SETTINGS-':
                sg.main_global_pysimplegui_settings()
            elif event == '-OPENREAPER-':
                REAPERController.open_reaper()
            elif event == '-SYNCPROJECT-':
                if self.tomi_engine is not None:
                    self.tomi_engine.sync_project()
            elif event.startswith('Graph@'):
                self.project_edit_pane.node_graph.handle_event(event, values)
            elif event.startswith('Arrangement@'):
                self.project_edit_pane.node_arrangement.handle_event(event, values)
            elif event.startswith('ProjectSetting@'):
                self.project_settings.handle_event(event, values)
            elif event.startswith('NodeSetting@'):
                self.project_edit_pane.node_settings.handle_event(event, values)
            elif event.startswith('ProjectArrangement@'):
                self.project_edit_pane.project_arrangement.handle_event(event, values)
            elif event == '__TIMEOUT__':
                self.frequent_check_handler()

