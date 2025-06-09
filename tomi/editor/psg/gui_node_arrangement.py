from . import GUIWidget, sg
from tomi import NodeBase, OrderedDotdict, NodeType, Dotdict
from .gui_utils import *


class GUINodeArrangement(GUIWidget):
    def __init__(self, host):
        super().__init__(host)
        self.base_frame_size = (1000, 1000)
        self.base_cell_size = (100, 50)
        self.default_font = ("IBM Plex Mono", 16, 'bold')
        self.available_node_types = ['AllTransformations', 'AllClips', 'ProjectArrangement'] + [nt for nt in NodeType.option_list() if not nt.startswith('Effect')]
        self.current_selected_row_type = 'Track'
        self.current_selected_col_type = 'ProjectArrangement'
        self.arrangement_layout = sg.Col([], scrollable=True, vertical_scroll_only=False, expand_x=True, expand_y=True)
        self.table_canvas = None
        self.table_layout = OrderedDotdict()
        self.node_name_map = {}
        self.row_index_nodes = []
        self.col_index_nodes = []
        self.row_index_names = ['HEADER']
        self.col_index_names = ['INDEX']
        self.btn_names = []
        self.current_col_width = {}
        self.current_row_height = {}

    def get_cell_at(self, row_name, col_name):
        return self.table_layout[row_name].row_cells[col_name]

    def insert_to_cell(self, row_name, col_name, layout):
        cell = self.get_cell_at(row_name, col_name)
        cell.buttons.extend(layout)
        self.window.extend_layout(cell.frame, layout)

    def iter_row_cells(self, index_name):
        for cell in self.table_layout[index_name].row_cells.values():
            yield cell

    def iter_col_cells(self, index_name):
        for row in self.table_layout:
            yield self.table_layout[row].row_cells[index_name]

    def get_node_layout(self, node: NodeBase):
        self.node_name_map[str(node.id)] = node
        return sg.Button(node.name, button_color=NodeColors[node.node_type].rect_color, font=self.default_font,
                         enable_events=True, key=self._get_btn_name(f'Arrangement@Node&{node.id}'), expand_x=True, expand_y=True)

    def _get_btn_name(self, name):
        if name in self.btn_names:
            if '#' in name:
                n, r = name.split('#')
                name = f"{n}#{int(r)+1}"
                return self._get_btn_name(name)
            else:
                return self._get_btn_name(f"{name}#2")
        else:
            self.btn_names.append(name)
            return name

    def set_row_height(self, index_node_name: str, height: int):
        for cell in self.iter_row_cells(index_node_name):
            cell.frame.set_size((None, height))

    def set_col_width(self, index_node_name: str, width: int):
        for cell in self.iter_col_cells(index_node_name):
            cell.frame.set_size((width, None))

    def resize_table_cells(self):
        for col in self.col_index_names:
            col_width = self.base_cell_size[0] + 20
            for cell in self.iter_col_cells(col):
                for cell_btn_row in cell.buttons:
                    col_width = max(col_width, sum(measure_text_size(button.get_text() if hasattr(button, 'get_text') else button.get(), self.default_font)+20 for button in cell_btn_row))
            self.current_col_width[col] = col_width
            self.set_col_width(col, col_width)
        for row in self.row_index_names:
            if row == 'HEADER':
                row_height = 65
            else:
                row_height = self.base_cell_size[1]
                for cell in self.iter_row_cells(row):
                    row_height = max(row_height, len(cell.buttons) * self.base_cell_size[1])
            self.current_row_height[row] = row_height
            self.set_row_height(row, row_height)

    def get_header_layout(self, col_nodes):
        index_items = [
            [sg.T('Row Type', font=self.default_font),
            sg.Combo(self.get_valid_axis_types(self.current_selected_col_type), self.current_selected_row_type,
                     enable_events=True, key='Arrangement@RowType', expand_x=True, readonly=True)],
            [sg.T('Column Type', font=self.default_font),
            sg.Combo(self.get_valid_axis_types(self.current_selected_row_type), self.current_selected_col_type,
                     enable_events=True, key='Arrangement@ColType', expand_x=True, readonly=True)],
        ]
        index_cell = create_frame(index_items, size=self.base_cell_size)
        row_cells = OrderedDotdict(INDEX=Dotdict(frame=index_cell, buttons=index_items))
        for col_node in col_nodes:
            btn = self.get_node_layout(col_node)
            row_cells[self._get_col_index_name(col_node.id)] = Dotdict(frame=create_frame([[btn]], size=self.base_cell_size), buttons=[[btn]])
        row_ele = sg.Col([[cell.frame for cell in row_cells.values()]])
        self.table_layout['HEADER'] = Dotdict(row_ele=row_ele, row_cells=row_cells)
        return row_ele

    def get_row_layout(self, index_node: NodeBase):
        node_btn = self.get_node_layout(index_node)
        row_cells = OrderedDotdict(INDEX=Dotdict(frame=create_frame([[node_btn]], size=self.base_cell_size), buttons=[[node_btn]]))
        for col_name in self.col_index_names:
            if col_name == 'INDEX':
                continue
            row_cells[col_name] = Dotdict(frame=create_frame([], size=self.base_cell_size), buttons=[])
        row_ele = sg.Col([[cell.frame for cell in row_cells.values()]])
        self.table_layout[self._get_row_index_name(index_node.id)] = Dotdict(row_ele=row_ele, row_cells=row_cells)
        return row_ele

    def _get_row_index_name(self, node_id: int | str):
        name = str(node_id) if isinstance(node_id, int) else node_id
        if name in self.row_index_names:
            if '#' in name:
                n, r = name.split('#')
                name = f"{n}#{int(r)+1}"
                return self._get_row_index_name(name)
            else:
                return self._get_row_index_name(f"{name}#2")
        else:
            self.row_index_names.append(name)
            return name

    def _get_col_index_name(self, node_id: int | str):
        name = str(node_id) if isinstance(node_id, int) else node_id
        if name in self.col_index_names:
            if '#' in name:
                n, r = name.split('#')
                name = f"{n}#{int(r)+1}"
                return self._get_col_index_name(name)
            else:
                return self._get_col_index_name(f"{name}#2")
        else:
            self.col_index_names.append(name)
            return name

    def _init_project_links(self):
        for link in self.tomi_project.node_graph.arrangement_links.get_links():
            other_nodes = []
            col_node = row_node = None
            for node in link:
                if node in self.col_index_nodes:
                    col_node = node
                elif node in self.row_index_nodes:
                    row_node = node
                else:
                    other_nodes.append(node)
            if col_node is None or row_node is None:
                continue
            for col_name in self.col_index_names:
                if col_name.split('#')[0] == str(col_node.id):
                    for row_name in self.row_index_names:
                        if row_name.split('#')[0] == str(row_node.id):
                            self.insert_to_cell(row_name, col_name, [[self.get_node_layout(node) for node in other_nodes]])

    def get_axis_nodes(self, axis_type: str):
        if axis_type == 'ProjectArrangement':
            return self.tomi_project.arrangement
        elif axis_type == 'AllTransformations':
            return self.tomi_project.transformation_nodes
        elif axis_type == 'AllClips':
            return self.tomi_project.clip_nodes
        elif axis_type == 'Section':
            return self.tomi_project.section_nodes
        elif axis_type == 'Track':
            return self.tomi_project.track_nodes
        elif axis_type == 'GeneralTransform':
            return self.tomi_project.general_transformation_nodes
        elif axis_type == 'FxTransform':
            return self.tomi_project.fx_transformation_nodes
        elif axis_type == 'FillTransform':
            return self.tomi_project.fill_transformation_nodes
        elif axis_type == 'DrumTransform':
            return self.tomi_project.drum_transformation_nodes
        elif axis_type == 'MidiClip':
            return self.tomi_project.midi_nodes
        elif axis_type == 'AudioClip':
            return self.tomi_project.audio_nodes
        return None

    def init_node_arrangement(self):
        self.col_index_nodes = self.get_axis_nodes(self.current_selected_col_type)
        self.row_index_nodes = self.get_axis_nodes(self.current_selected_row_type)
        header = self.get_header_layout(self.col_index_nodes)
        table_layout = [[header]] + [[self.get_row_layout(r_i)] for r_i in self.row_index_nodes]
        self.table_canvas = sg.Col(layout=table_layout, background_color='white', expand_x=True, expand_y=True,
                                   justification='left')
        self.window.extend_layout(self.arrangement_layout, [[self.table_canvas]])
        self._init_project_links()
        self.resize_table_cells()

    def remove_table(self):
        remove_element(self.table_canvas, self.window)
        for row_name in self.row_index_names:
            for cell in self.iter_row_cells(row_name):
                for btn_groups in cell.buttons:
                    for btn in btn_groups:
                        remove_key_from_window(self.window, btn.key)
                        del btn
                del cell.frame
            del self.table_layout[row_name].row_ele
        self.table_canvas = None
        self.table_layout = OrderedDotdict()
        self.node_name_map = {}
        self.row_index_nodes = []
        self.col_index_nodes = []
        self.row_index_names = ['HEADER']
        self.col_index_names = ['INDEX']
        self.btn_names = []
        self.current_col_width = {}
        self.current_row_height = {}


    def get_layout(self):
        return self.arrangement_layout

    def get_valid_axis_types(self, axis_type: str):
        exclude = [axis_type]
        if axis_type == 'ProjectArrangement':
            exclude.append('Section')
        elif axis_type == 'AllTransformations':
            exclude.extend(['GeneralTransform', 'DrumTransform', 'FxTransform', 'FillTransform'])
        elif axis_type == 'AllClips':
            exclude.extend(['MidiClip', 'AudioClip'])
        elif axis_type == 'Section':
            exclude.append('ProjectArrangement')
        elif axis_type == 'GeneralTransform':
            exclude.extend(['AllTransformations', 'DrumTransform', 'FxTransform', 'FillTransform'])
        elif axis_type == 'FxTransform':
            exclude.extend(['AllTransformations', 'GeneralTransform', 'DrumTransform', 'FillTransform'])
        elif axis_type == 'FillTransform':
            exclude.extend(['AllTransformations', 'GeneralTransform', 'DrumTransform', 'FxTransform'])
        elif axis_type == 'DrumTransform':
            exclude.extend(['AllTransformations', 'GeneralTransform', 'FxTransform', 'FillTransform'])
        elif axis_type == 'MidiClip':
            exclude.extend(['AllClips', 'AudioClip'])
        elif axis_type == 'AudioClip':
            exclude.extend(['MidiClip', 'AllClips'])
        return [nt for nt in self.available_node_types if nt not in exclude]

    def handle_event(self, event, values):
        if event == 'Arrangement@RowType':
            value = values[event]
            if value in self.available_node_types:
                if value != self.current_selected_row_type:
                    self.current_selected_row_type = value
                    col_val = values['Arrangement@ColType']
                    new_col_vals = self.get_valid_axis_types(value)
                    col_val = new_col_vals[0] if col_val == value else col_val
                    self.window['Arrangement@ColType'].update(value=col_val, values=new_col_vals)
                    self.remove_table()
                    self.init_node_arrangement()
            else:
                self.window['Arrangement@RowType'].update(value=self.current_selected_row_type)
        elif event == 'Arrangement@ColType':
            value = values[event]
            if value in self.available_node_types:
                if value != self.current_selected_col_type:
                    self.current_selected_col_type = value
                    row_val = values['Arrangement@RowType']
                    new_row_vals = self.get_valid_axis_types(value)
                    row_val = new_row_vals[0] if row_val == value else row_val
                    self.window['Arrangement@RowType'].update(value=row_val, values=new_row_vals)
                    self.remove_table()
                    self.init_node_arrangement()
            else:
                self.window['Arrangement@ColType'].update(value=self.current_selected_col_type)
        elif event.startswith('Arrangement@Node&'):
            node_id = event.split('&')[1].split('#')[0]
            self.host_editor.node_settings.show_node_info(self.node_name_map[node_id])
