from tomi import NodeBase, Dotdict
from . import GUIWidget
from .gui_utils import *


class GUINode(GUIWidget):
    def __init__(self, host, node: NodeBase, input_port: bool = True, output_port: bool = True):
        super().__init__(host)
        self.graph = self.host_editor.graph
        self.node = node
        self.default_font = ("IBM Plex Mono", 16, 'bold')
        self.node_color = NodeColors[self.node.node_type]
        self.dim_color = Dotdict({k: dim_color(v) for k,v in self.node_color.items()})
        self.have_input = input_port
        self.have_output = output_port
        self.text = self.rect = self.input = self.output = None
        self.position = None
        self.selected_item = SelectedItem.Empty
        self.state = WidgetState.Normal
        self.input_links = []
        self.output_links = []

    def draw(self, position: tuple = None):
        if position is None:
            position = self.position
        if self.text is not None: self.graph.Widget.delete(self.text)
        if self.rect is not None: self.graph.Widget.delete(self.rect)
        if self.have_input and self.input is not None: self.graph.Widget.delete(self.input)
        if self.have_output and self.output is not None: self.graph.Widget.delete(self.output)
        self.text = self.graph.draw_text(self.node.name, position, font=self.default_font, color='white')
        left, top, right, bottom = self.get_bbox(self.text)
        rect_tl = (left - 20, top + 10)
        rect_br = (right + 20, bottom - 10)
        self.rect = self.graph.draw_rectangle(rect_tl, rect_br, fill_color=self.node_color.rect_color, line_width=4)
        port_y = rect_br[1] + (rect_tl[1] - rect_br[1]) / 2
        if self.have_input: self.input = self.graph.draw_circle((rect_tl[0], port_y), radius=10, fill_color=self.node_color.input_color, line_color='white')
        if self.have_output: self.output = self.graph.draw_circle((rect_br[0], port_y), radius=10, fill_color=self.node_color.output_color, line_color='white')
        self.graph.bring_figure_to_front(self.text)
        self.position = position

    @property
    def input_center(self):
        return self.get_item_center(self.input) if self.have_input else None

    @property
    def output_center(self):
        return self.get_item_center(self.output) if self.have_output else None

    @property
    def input_selected(self):
        return self.selected_item == SelectedItem.Input

    @property
    def output_selected(self):
        return self.selected_item == SelectedItem.Output

    @property
    def rect_selected(self):
        return self.selected_item == SelectedItem.Rect

    @property
    def nothing_selected(self):
        return self.selected_item == SelectedItem.Empty

    def get_item_center(self, sid):
        l, t, r, b = self.get_bbox(sid)
        return l+(r-l)/2, b+(t-b)/2

    def get_current_position(self):
        l, t, r, b = self.get_bbox(self.text)
        return l + (r-l)/2, b + (t-b)/2

    def get_bbox(self, sid):
        (l, t), (r, b) = self.graph.get_bounding_box(sid)
        return l, t, r, b

    @property
    def rect_box(self):
        return self.get_bbox(self.rect)

    @property
    def input_box(self):
        return self.get_bbox(self.input) if self.have_input else None

    @property
    def output_box(self):
        return self.get_bbox(self.output) if self.have_output else None

    def move_to(self, x, y):
        delta_x, delta_y = x-self.position[0], y-self.position[1]
        self.shift_to(delta_x, delta_y)

    def shift_to(self, shift_x, shift_y):
        x, y = self.position
        self.position = (x + shift_x, y + shift_y)
        self.graph.move_figure(self.text, shift_x, shift_y)
        self.graph.move_figure(self.rect, shift_x, shift_y)
        if self.have_input: self.graph.move_figure(self.input, shift_x, shift_y)
        if self.have_output: self.graph.move_figure(self.output, shift_x, shift_y)
        self.update_links()

    def update_links(self):
        for i_link in self.input_links:
            i_link.to_position = self.input_center
            i_link.draw()
        for o_link in self.output_links:
            o_link.from_position = self.output_center
            o_link.draw()

    def update_state(self, state: WidgetState = WidgetState.Normal, figure_id: int = None):
        if state == WidgetState.Normal:
            self.dehighlight()
            self.selected_item = SelectedItem.Empty
        elif state == WidgetState.Dim:
            self.dehighlight()
            self.dim()
            self.selected_item = SelectedItem.Empty
        elif state == WidgetState.Highlight:
            if figure_id is not None:
                self.dehighlight()
                self.highlight(figure_id)
            else:
                self.highlight_all()
            self.selected_item = SelectedItem.Empty
            self.bring_to_front()
        elif state == WidgetState.Selected:
            self.dehighlight()
            if figure_id is not None:
                self.highlight(figure_id)
                if figure_id == self.rect:
                    self.selected_item = SelectedItem.Rect
                elif figure_id == self.input:
                    self.selected_item = SelectedItem.Input
                elif figure_id == self.output:
                    self.selected_item = SelectedItem.Output
                else:
                    self.selected_item = SelectedItem.Empty
            else:
                self.highlight_all()
                self.selected_item = SelectedItem.Rect
            self.bring_to_front()
        else:
            raise ValueError(f'Wrong state: {state}')
        self.state = state

    def reset_state(self):
        self.update_state(WidgetState.Normal)

    def on_mouse(self, fig_ids: tuple, click: bool = True):
        rect_s = input_s = output_s = False
        for fig_id in fig_ids:
            if fig_id == self.rect:
                rect_s = True
            elif self.have_input and fig_id == self.input:
                input_s = True
            elif self.have_output and fig_id == self.output:
                output_s = True
        new_state = WidgetState.Selected if click else WidgetState.Highlight
        if input_s:
            self.update_state(new_state, self.input)
        elif output_s:
            self.update_state(new_state, self.output)
        elif rect_s:
            self.update_state(new_state, self.rect)
        else:
            self.update_state(WidgetState.Normal)
        return any((rect_s, input_s, output_s))

    def bring_to_front(self):
        self.graph.bring_figure_to_front(self.rect)
        if self.have_input: self.graph.bring_figure_to_front(self.input)
        if self.have_output: self.graph.bring_figure_to_front(self.output)
        for link in self.input_links:
            link.bring_to_front()
        for link in self.output_links:
            link.bring_to_front()
        self.graph.bring_figure_to_front(self.text)

    def dim(self):
        self.graph.Widget.itemconfigure(self.rect, fill=self.dim_color.rect_color)
        if self.have_input: self.graph.Widget.itemconfigure(self.input, fill=self.dim_color.input_color)
        if self.have_output: self.graph.Widget.itemconfigure(self.output, fill=self.dim_color.output_color)

    def undim(self):
        self.graph.Widget.itemconfigure(self.rect, fill=self.node_color.rect_color)
        if self.have_input: self.graph.Widget.itemconfigure(self.input, fill=self.node_color.input_color)
        if self.have_output: self.graph.Widget.itemconfigure(self.output, fill=self.node_color.output_color)

    def highlight(self, select_id):
        if self.rect == select_id:
            self.highlight_all()
        if self.have_input and self.input == select_id:
            self.graph.Widget.itemconfigure(self.input, outline="#000001")
        elif self.have_output and self.output == select_id:
            self.graph.Widget.itemconfigure(self.output, outline="#000001")

    def highlight_all(self):
        self.graph.Widget.itemconfigure(self.text, fill=self.node_color.rect_color)
        self.graph.Widget.itemconfigure(self.rect, fill='white')
        self.graph.Widget.itemconfigure(self.rect, outline="#000001")
        if self.have_input: self.graph.Widget.itemconfigure(self.input, outline="#000001")
        if self.have_output: self.graph.Widget.itemconfigure(self.output, outline="#000001")

    def dehighlight(self):
        self.graph.Widget.itemconfigure(self.text, fill='white')
        self.graph.Widget.itemconfigure(self.rect, fill=self.node_color.rect_color)
        self.graph.Widget.itemconfigure(self.rect, outline="white")
        if self.have_input: self.graph.Widget.itemconfigure(self.input, outline="white")
        if self.have_output: self.graph.Widget.itemconfigure(self.output, outline="white")