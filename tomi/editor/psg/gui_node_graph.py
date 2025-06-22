from tomi import get_random_rgb
from . import GUILink, GUINode, GUIWidget, sg
from .gui_utils import rgb_to_hex


class LinkComp:
    def __init__(self, link_id, color='#000000'):
        self.link_id = link_id
        self.color = color
        self.links = []

    def __hash__(self):
        return hash(self.link_id)

    def __eq__(self, other):
        return self.link_id == other.link_id

    def __repr__(self):
        return f"LinkID: {self.link_id}, Color: {self.color}, Links: {[l.link for l in self.links]}"


class GUINodeGraph(GUIWidget):
    def __init__(self, host):
        super().__init__(host)
        self.base_frame_size = (1000, 1000)
        self.graph_nodes = {}
        self.graph_links = {}
        self.link_comps = []
        self.latest_link_id = 0
        self.graph_canvas_size = None
        self.last_graph_mouse_pos = None
        self.dragging = False
        self.selected_node = None
        self.drawing_link = None
        self.graph = sg.Graph(canvas_size=self.base_frame_size, graph_bottom_left=(0, 0),
                              graph_top_right=self.base_frame_size,
                              drag_submits=True, key='Graph@', expand_x=True, expand_y=True, enable_events=True,
                              background_color='#F0F0F0', motion_events=True)
        self.graph_layout = sg.Col([[self.graph]], size=self.base_frame_size, scrollable=True, vertical_scroll_only=False, expand_x=True, expand_y=True)

    def get_layout(self):
        return self.graph_layout

    def init_project_graph(self):
        position = (0, 0)
        v_gap = 100
        h_gap = 300
        composition_links = self.tomi_project.node_graph.arrangement_links.get_links()
        for node_type in ['section_nodes', 'transformation_nodes', 'clip_nodes', 'track_nodes']:
            node_list = self.host_editor.tomi_project.__getattribute__(node_type)
            if len(node_list) == 0:
                continue
            for node in node_list:
                g_node = GUINode(self, node, node_type != 'section_nodes', node_type != 'track_nodes')
                g_node.draw(position)
                position = (position[0], position[1] - v_gap)
                self.graph_nodes[node] = g_node
            position = (position[0] + h_gap, 0)
        colors = [rgb_to_hex(rgb) for rgb in get_random_rgb(num_output=len(composition_links))]
        self.link_comps = [LinkComp(i, colors[i]) for i in range(len(composition_links))]
        for link_i in range(len(composition_links)):
            link = composition_links[link_i]
            for node_i in range(len(link)-1):
                GUILink(self).establish_link(from_node=self.graph_nodes[link[node_i]], to_node=self.graph_nodes[link[node_i+1]], link_comp=self.link_comps[link_i])
        self.latest_link_id = len(composition_links)
        self.adjust_graph_size()

    def shift_nodes(self, shift_x, shift_y):
        for node in self.graph_nodes.values():
            node.shift_to(shift_x, shift_y)

    def _change_graph_size(self, x, y):
        self.graph.set_size((x, y))
        self.graph.change_coordinates((0, 0), (x, y))
        self.graph_layout.contents_changed()

    def adjust_graph_size(self):
        if self.graph.get_size() == (1, 1):
            return
        element_positions = [n.position for n in self.graph_nodes.values()]
        left = min(p[0] for p in element_positions) - 200
        top = max(p[1] for p in element_positions) + 200
        right = max(p[0] for p in element_positions) + 200
        bottom = min(p[1] for p in element_positions) - 200
        new_w, new_h = right - left, top - bottom
        canvas_w, canvas_h = self.graph_layout.get_size()
        base_w, base_h = self.base_frame_size
        new_w = max((new_w, base_w, canvas_w))
        new_h = max((new_h, base_h, canvas_h))
        graph_size = self.graph.get_size()
        y_offset = new_h - graph_size[1]
        if (new_w, new_h) != graph_size:
            if y_offset != 0:
                for n in self.graph_nodes.values():
                    n.position = (n.position[0], n.position[1] + y_offset)
                top += y_offset
                bottom += y_offset
            self._change_graph_size(new_w, new_h)
        shift_x, shift_y = 0, 0
        if left < 0: shift_x += -left
        if bottom < 0: shift_y += -bottom
        if top > new_h: shift_y += -(top-new_h)
        if right > new_w: shift_x += -(right-new_w)
        if shift_x != 0 or shift_y != 0:
            self.shift_nodes(shift_x, shift_y)

    def get_graph_widget_on_mouse(self, x, y, click: bool = True):
        widgets = self.graph.get_figures_at_location((x, y))
        found_widget = None
        for gui_node in self.graph_nodes.values():
            if found_widget is None and gui_node.on_mouse(widgets, click):
                found_widget = gui_node
            else:
                gui_node.reset_state()
        if found_widget is None and self.drawing_link is None:
            for gui_link in self.graph_links:
                if found_widget is None and gui_link.on_mouse(widgets, click):
                    found_widget = gui_link
                    break
        else:
            for gui_link in self.graph_links:
                gui_link.reset_state()
        return found_widget

    def hovering_on_graph(self, x, y):
        if self.selected_node is not None:
            return None
        widgets = self.graph.get_figures_at_location((x, y))
        found_widget = None
        for gui_node in self.graph_nodes.values():
            if found_widget is None and gui_node.on_mouse(widgets, False):
                found_widget = gui_node
            else:
                gui_node.reset_state()
        return found_widget

    def reset_state_for_all(self):
        for gui_node in self.graph_nodes.values():
            gui_node.reset_state()
        for gui_link in self.graph_links:
            gui_link.reset_state()

    def monitor_graph_size_change_handler(self):
        canvas_size = self.graph_layout.get_size()
        if canvas_size != self.graph_canvas_size:
            self.adjust_graph_size()
            self.graph_canvas_size = canvas_size

    def register_link(self, link: GUILink, link_comp: LinkComp = None):
        if link_comp is None:
            lc = LinkComp(self.latest_link_id)
            self.latest_link_id += 1
        else:
            lc = link_comp
        self.graph_links[link] = lc
        lc.links.append(link)
        if lc not in self.link_comps:
            self.link_comps.append(lc)

    def handle_event(self, event, values):
        x, y = values['Graph@']
        if event == 'Graph@':
            if self.dragging:
                if self.last_graph_mouse_pos == (x, y):
                    return
                if self.selected_node.output_selected or self.drawing_link is not None:
                    self.get_graph_widget_on_mouse(x, y, False)
                    # if self.drawing_link is None:
                    #     self.drawing_link = GUILink(self)
                    #     self.drawing_link.draw(self.selected_node, (x, y))
                    # else:
                    #     self.drawing_link.draw(to_pos=(x, y))
                else:
                    self.selected_node.move_to(x, y)
            else:
                self.selected_node = self.get_graph_widget_on_mouse(x, y, True)
                # self.selected_node = found_widget if isinstance(found_widget, GUINode) else None
                if self.selected_node is not None and isinstance(self.selected_node, GUINode): self.dragging = True
            self.last_graph_mouse_pos = (x, y)
        elif event == 'Graph@+UP':
            if self.drawing_link is not None:
                to_node = self.get_graph_widget_on_mouse(x, y, False)
                if to_node is None or not isinstance(to_node, GUINode) or to_node.output_selected or not to_node.have_input:
                    self.drawing_link.delete()
                else:
                    self.drawing_link.establish_link(to_node=to_node)
                if isinstance(to_node, GUINode):
                    to_node.reset_state()
            if self.dragging:
                self.host_editor.show_node_info(self.selected_node.node if self.selected_node is not None and isinstance(self.selected_node, GUINode) else None)
            self.graph_layout.Widget.hookMouseWheel(None)
            self.dragging = False
            # self.selected_node = None
            self.drawing_link = None
            self.adjust_graph_size()
        elif event == 'Graph@+MOVE':
            self.hovering_on_graph(x, y)