from . import GUIWidget, sg
from .gui_utils import *


class GUILink(GUIWidget):
    def __init__(self, host):
        super().__init__(host)
        self.graph: sg.Graph = self.host_editor.graph
        self.from_node = self.to_node = None
        self.link = None
        self.link_established = False
        self.link_comp = None
        self.from_position = None
        self.to_position = None
        self.state = WidgetState.Normal

    def set_link_comp(self, link_comp):
        self.link_comp = link_comp

    def draw(self, from_pos = None, to_pos = None):
        if self.link is not None:
            self.graph.delete_figure(self.link)
        if from_pos is None:
            f_pos = self.from_position
        elif type(from_pos).__name__ == 'GUINode':
            f_pos = self.from_position = from_pos.output_center
            self.from_node = from_pos
        else:
            f_pos = from_pos
        if to_pos is None:
            t_pos = self.to_position
        elif type(to_pos).__name__ == 'GUINode':
            t_pos = self.to_position = to_pos.input_center
            self.to_node = to_pos
        else:
            t_pos = to_pos
        assert f_pos is not None, '"from_pos" cannot be None.'
        assert t_pos is not None, '"to_pos" cannot be None.'
        color = self.link_comp.color if self.link_comp is not None else 'black'
        # self.link = self.graph.draw_arc(f_pos, t_pos, extent=5, start_angle=5, line_width=3, arc_color=color, style='bevel')
        self.link = self.graph.draw_line(f_pos, t_pos, width=3, color=color)
        self.bring_to_front()

    def bring_to_front(self):
        if self.link is not None:
            self.graph.bring_figure_to_front(self.link)

    def dim(self):
        self.graph.Widget.itemconfigure(self.link, fill=dim_color(self.link_comp.color))

    def undim(self):
        self.graph.Widget.itemconfigure(self.link, fill=self.link_comp.color)

    def delete(self):
        self.graph.delete_figure(self.link)
        if self.link_established:
            self.from_node.output_links.remove(self)
            self.to_node.input_links.remove(self)

    def establish_link(self, from_node = None, to_node = None, link_comp = None):
        assert from_node is not None or self.from_node is not None
        assert to_node is not None or self.to_node is not None
        if from_node is not None: self.from_node = from_node
        if to_node is not None: self.to_node = to_node
        self.from_node.output_links.append(self)
        self.to_node.input_links.append(self)
        self.link_established = True
        self.host_editor.register_link(self, link_comp)
        self.set_link_comp(self.host_editor.graph_links[self])
        self.draw(from_node, to_node)

    def update_state(self, state: WidgetState = WidgetState.Normal):
        if state == WidgetState.Normal:
            self.undim()
        elif state == WidgetState.Dim:
            self.dim()
        elif state == WidgetState.Highlight:
            self.undim()
            self.bring_to_front()
        elif state == WidgetState.Selected:
            self.undim()
            self.bring_to_front()
            self.highlight_link_comp()
        else:
            raise ValueError(f'Wrong state: {state}')
        self.state = state

    def reset_state(self):
        self.update_state(WidgetState.Normal)

    def highlight_link_comp(self):
        for gui_node in self.host_editor.graph_nodes.values():
            if not (any(l.link_comp == self.link_comp for l in gui_node.input_links) or any(
                    l.link_comp == self.link_comp for l in gui_node.output_links)):
                gui_node.update_state(WidgetState.Dim)
        for link in self.host_editor.graph_links:
            if link == self:
                continue
            if link.link_comp != self.link_comp:
                link.update_state(WidgetState.Dim)
            else:
                link.update_state(WidgetState.Highlight)

    def on_mouse(self, fig_ids: tuple, click: bool = True):
        link_s = False
        for fig_id in fig_ids:
            if fig_id == self.link:
                link_s = True
                break
        if link_s:
            self.update_state(WidgetState.Selected)
        else:
            self.update_state(WidgetState.Normal)
        return link_s
