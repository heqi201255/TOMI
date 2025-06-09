from tomi import NodeBase, NodeType
from . import GUIWidget, sg
from .gui_utils import *


class GUINodeSetting(GUIWidget):
    def __init__(self, host):
        super().__init__(host)
        self.node_settings = sg.Frame(title='Node Settings', layout=[[sg.Push(), sg.Button('X', enable_events=True, key='NodeSetting@CLOSE')]], expand_x=True, expand_y=True, visible=False)
        self.current_setting_layout = None
        self.current_configs = None

    def get_layout(self):
        return self.node_settings

    def show_node_info(self, node: NodeBase = None):
        if node is None:
            self.node_settings.update(visible=False)
            self.host_editor.change_pane_handle_position()
            return
        if not self.node_settings.visible:
            self.node_settings.update(visible=True)
            self.host_editor.change_pane_handle_position(0.5)
        node_id = node.id
        layout_name = f'&{node_id}'
        if self.current_setting_layout is not None and self.current_setting_layout.key == layout_name:
            self.node_settings.Widget.update()
            self.current_setting_layout.contents_changed()
            return
        if self.current_setting_layout is not None: self.current_setting_layout.update(visible=False)
        remove_element(self.current_setting_layout, self.window)
        if self.current_configs is not None: self.current_configs.current_layout = None
        if not node.node_type in (NodeType.GeneralTransform, NodeType.FillTransform, NodeType.DrumTransform, NodeType.FxTransform):
            self.current_configs = node.gui_configs
            node.update_gui_configs()
            self.current_configs.set_key_prefix(f'NodeSetting@{node_id}')
            node_layout = self.current_configs.get_config_layout()
        else:
            node_info = node.__getstate__()
            node_layout = []
            for k, v in node_info.items():
                setting_key = f'NodeSetting{layout_name}@{k}'
                node_layout.append([sg.Text(k), sg.Push(), sg.InputText(str(v), enable_events=True, key=setting_key)])
        self.current_setting_layout = sg.Col(node_layout, scrollable=True, key=layout_name, expand_x=True, expand_y=True)
        self.window.extend_layout(self.node_settings, [[self.current_setting_layout]])
        try:
            self.current_configs.bind_configs_to_button()
        except:
            pass

    def handle_event(self, event, values):
        if event == 'NodeSetting@CLOSE':
            self.show_node_info()
        elif event.endswith('+Clicked'):
            elem = self.window[event.split('+')[0]]
            if not (isinstance(elem, sg.InputText) and elem.Widget.focus_get()):
                elem.set_focus(True)
                self.current_configs.update_entire_gui(self.window)
        elif event.endswith('+Returned'):
            self.current_configs.update_entire_gui(self.window)
        else:
            param_value = values[event] if event in values else None
            self.current_configs.handle_value_changes(event, param_value, self.window)
