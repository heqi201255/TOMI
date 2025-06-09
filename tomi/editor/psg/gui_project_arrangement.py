from . import GUIWidget, GUINode, sg


class GUIProjectArrangement(GUIWidget):
    def __init__(self, host):
        super().__init__(host)
        self.base_frame_size = (1000, 50)
        self.graph_nodes = {}
        self.graph_canvas_size = None
        self.last_graph_mouse_pos = None
        self.dragging = False
        self.selected_node = None
        self.graph = sg.Graph(canvas_size=self.base_frame_size, graph_bottom_left=(0, 0),
                              graph_top_right=self.base_frame_size,
                              drag_submits=True, key='ProjectArrangement@', expand_x=True, expand_y=True, enable_events=True,
                              background_color='#F0F0F0', motion_events=True)
        self.graph_layout = sg.Col([[self.graph]], scrollable=False,
                                    expand_x=True, expand_y=True)

    def get_layout(self):
        return self.graph_layout

    def init_project_arrangement(self):
        position = (0, 25)
        left = 0
        for node in self.tomi_project.arrangement:
            g_node = GUINode(self, node, False, False)
            if node not in self.graph_nodes:
                self.graph_nodes[node] = []
            self.graph_nodes[node].append(g_node)
            g_node.draw(position)
            l, t, r, b = g_node.rect_box
            offset = left - l
            g_node.shift_to(offset, 0)
            left = r + offset

    def get_graph_widget_on_mouse(self, x, y, click: bool = True):
        widgets = self.graph.get_figures_at_location((x, y))
        found_widget = None
        for gui_nodes in self.graph_nodes.values():
            for gui_node in gui_nodes:
                if found_widget is None and gui_node.on_mouse(widgets, click):
                    found_widget = gui_node
                else:
                    gui_node.reset_state()
        return found_widget

    def handle_event(self, event, values):
        x, y = values['ProjectArrangement@']
        if event == 'ProjectArrangement@':
            self.selected_node = self.get_graph_widget_on_mouse(x, y, True)
            self.host_editor.show_node_info(
                self.selected_node.node if self.selected_node is not None else None)
            self.last_graph_mouse_pos = (x, y)
        elif event == 'ProjectArrangement@+UP':
            self.dragging = False
        elif event == 'ProjectArrangement@+MOVE':
            if self.selected_node is None:
                self.get_graph_widget_on_mouse(x, y, False)