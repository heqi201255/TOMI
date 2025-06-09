from . import sg, GUIWidget, GUINodeGraph, GUIProjectArrangement, GUINodeSetting, GUINodeArrangement
from tomi import NodeBase

class GUIProjectEditPane(GUIWidget):
    def __init__(self, host):
        super().__init__(host)
        self.node_graph = GUINodeGraph(self)
        self.node_arrangement = GUINodeArrangement(self)

        self.node_settings = GUINodeSetting(self)
        self.project_arrangement = GUIProjectArrangement(self)
        self.project_node_tab_group = sg.TabGroup(layout=[[
            sg.Tab('NodeArrangement', [[self.node_arrangement.get_layout()]]),
            sg.Tab('NodeGraph', [[self.node_graph.get_layout()]])
        ]], expand_x=True, expand_y=True)
        self.project_node_pane = sg.Pane([
            sg.Col([[self.project_node_tab_group]], expand_x=True, expand_y=True),
            sg.Col([[self.node_settings.get_layout()]], expand_x=True, expand_y=True),
        ], show_handle=False, orientation='h', expand_x=True, expand_y=True)
        self.project_layout = [
            [sg.T('Song Arrangement'), self.project_arrangement.get_layout()],
            [self.project_node_pane]
        ]
        self.pane_size = None

    def show_node_info(self, node: NodeBase = None):
        self.node_settings.show_node_info(node)

    @property
    def active_tab(self):
        return self.project_node_tab_group.find_currently_active_tab_key()

    def monitor_project_pane_size_change_handler(self):
        if self.active_tab == 'NodeGraph':
            self.node_graph.get_layout().contents_changed()
            self.node_graph.monitor_graph_size_change_handler()
        elif self.active_tab == 'NodeArrangement':
            self.node_arrangement.get_layout().contents_changed()
        p_size = self.project_node_pane.get_size()
        if p_size != self.pane_size:
            self.pane_size = p_size
            if not self.node_settings.visible:
                self.change_pane_handle_position()
            else:
                self.change_pane_handle_position(0.5)

    def get_layout(self):
        return self.project_layout

    def init_project(self):
        self.node_arrangement.init_node_arrangement()
        self.project_arrangement.init_project_arrangement()
        self.node_graph.init_project_graph()

    def change_pane_handle_position(self, percentage: float = 1):
        self.project_node_pane.Widget.sash_place(0, int(self.project_node_pane.get_size()[0] * percentage), 1)