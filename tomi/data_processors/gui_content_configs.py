import FreeSimpleGUI as sg
from . import MIDIProcessor
from tomi import OrderedDotdict, Dotdict, TOMIEnum, ItemSelector, BarStepTick, TimeSignature, Groove
import ast


class GUIContentConfigs:
    def __init__(self, host):
        self.host = host
        self.configs = OrderedDotdict()
        self.key_prefix = None
        self.key_element_map = {}
        self.last_modified_param = self.last_modified_value = None
        self.current_layout = None

    def set_key_prefix(self, prefix: str):
        self.key_prefix = prefix

    def update_config(self, param_name: str, gui_name, param_type: type, value, mutable: bool, special_entry: str = None):
        self.configs[param_name] = Dotdict(gui_name=gui_name, type=param_type, value=value, mutable=mutable, special_entry=special_entry)

    def handle_value_changes(self, event: str, value: str, window: sg.Window):
        ele_type, value_type = self.key_element_map[event]
        param = event.split('@')[-1]
        if ele_type is sg.InputText:
            if value_type in (int, float):
                try:
                    processed = value_type(value) if value else 0
                    self.last_modified_param = param
                    self.last_modified_value = processed
                except ValueError:
                    window[event].update(value=value[:-1])
            else:
                self.last_modified_param = param
                self.last_modified_value = value
        elif ele_type is sg.Combo and value_type in (Groove, MIDIProcessor):
            window[event].update(value=window[event].Values[0])
        else:
            self.last_modified_param = param
            self.last_modified_value = value
            self.update_entire_gui(window)

    def convert_to_type(self, param_name: str, value: str):
        param_type = self.configs[param_name].type
        special_entry = self.configs[param_name].special_entry
        if param_type in (set, list, tuple):
            if special_entry == 'min_max_range':
                processed = float(value) if value else 0
            elif special_entry == 'string_container':
                processed = param_type([x for x in [v.strip() for v in value.split(',')] if x])
            else:
                value = f"{{{value}}}" if param_type is set else f"[{value}]" if param_type is list else f"({value},)"
                processed = ast.literal_eval(value)
        elif param_type is BarStepTick:
            processed = int(value) if value != "" else 0
        elif param_type is ItemSelector:
            processed = value
        elif param_type is TimeSignature:
            processed = int(value) if value else 0
        else:
            processed = param_type(value)
        return processed

    def bind_configs_to_button(self):
        if self.current_layout is None:
            raise ValueError('Layout is not created yet.')
        for row in self.current_layout:
            for ele in row:
                ele.bind('<Button-1>', "+Clicked")
        for row in self.current_layout[0][0].Rows + self.current_layout[0][1].Rows:
            for ele in row:
                if isinstance(ele, sg.Col):
                    for row2 in ele.Rows:
                        for ele2 in row2:
                            if isinstance(ele2, sg.InputText):
                                ele2.bind('<Return>', "+Returned")
                            ele2.bind('<Button-1>', "+Clicked")
                else:
                    if isinstance(ele, sg.InputText):
                        ele.bind('<Return>', "+Returned")
                    ele.bind('<Button-1>', "+Clicked")

    def get_config_layout(self):
        layout_left = []
        layout_right = []
        for k, attrs in self.items():
            field_key = f'{self.key_prefix}@{k}'
            field_value = attrs.value
            if attrs.type in (str, int, float):
                value_field = sg.InputText(field_value, enable_events=True, key=field_key, disabled=not attrs.mutable,
                                           disabled_readonly_background_color='white', expand_x=True, expand_y=True)
                self.key_element_map[field_key] = sg.InputText, attrs.type
            elif attrs.type is bool:
                value_field = sg.Checkbox('', default=field_value, enable_events=True, key=field_key,
                                        disabled=not attrs.mutable, expand_x=True, expand_y=True, pad=0)
                self.key_element_map[field_key] = sg.Checkbox, attrs.type
            elif issubclass(attrs.type, TOMIEnum):
                value_field = sg.Combo(values=attrs.type.option_list(), default_value=field_value.name,
                                       enable_events=True, key=field_key, readonly=True, disabled=not attrs.mutable,
                                       expand_x=True, expand_y=True)
                self.key_element_map[field_key] = sg.Combo, TOMIEnum
            elif attrs.type is ItemSelector:
                value_field = sg.Col([[
                    sg.Combo(values=field_value.option_list(), default_value=field_value.current_key, enable_events=True,
                            key=field_key, readonly=True, auto_size_text=False, disabled=not attrs.mutable, expand_y=True),
                    sg.Button('random', enable_events=True, key=f"{field_key}!r", pad=0)
                ]], pad=0, expand_y=True)
                self.key_element_map[field_key] = sg.Combo, str
                self.key_element_map[f"{field_key}!r"] = sg.Button, None
            elif attrs.type is TimeSignature:
                value_field = sg.Col([[
                    sg.T('Numerator', expand_y=True),
                    sg.InputText(field_value.numerator, size=(3, 3), enable_events=True, key=f"{field_key}$0",
                                 disabled=not attrs.mutable,
                                 disabled_readonly_background_color='white', pad=0, expand_y=True),
                    sg.T('Denominator', expand_y=True),
                    sg.InputText(field_value.denominator, size=(3, 3), enable_events=True, key=f"{field_key}$1",
                                 disabled=not attrs.mutable,
                                 disabled_readonly_background_color='white', pad=0, expand_y=True)
                ]], pad=0, expand_y=True)
                self.key_element_map[f"{field_key}$0"] = sg.InputText, int
                self.key_element_map[f"{field_key}$1"] = sg.InputText, int
            elif attrs.type is Groove:
                if field_value is not None:
                    field_value: Groove
                    values = [f'Type: {field_value.midi_type.name}', f'Length: {field_value.bar_length}',
                              f"Speed: {field_value.speed.name}", f'ProgCount: {field_value.progression_count}',
                              f"Groove: {field_value.groove.tolist()}"]
                else:
                    values = ['Unavailable']
                value_field = sg.Drop(values, default_value=values[0], enable_events=True, key=field_key,
                                      readonly=True, auto_size_text=False, disabled=field_value is None, expand_y=True)
                self.key_element_map[field_key] = sg.Combo, Groove
            elif attrs.type is MIDIProcessor:
                if field_value is not None:
                    field_value: MIDIProcessor
                    values = [f'Name: {field_value.name}', f'Type: {field_value.midi_type.name}', f'Length: {field_value.ceil_bst}',
                              f'NoteCount: {len(field_value.get_notelist())}', f'Key: {field_value.key}',
                              f'RootProgression: {field_value.progression_nums['major']}']
                else:
                    values = ['Unavailable']
                value_field = sg.Drop(values, default_value=values[0], enable_events=True, key=field_key,
                              readonly=True, auto_size_text=False, disabled=field_value is None, expand_y=True)
                self.key_element_map[field_key] = sg.Combo, MIDIProcessor
            elif attrs.type in (set, list, tuple):
                if attrs.special_entry == 'min_max_range':
                    value_field = sg.Col([[
                        sg.T('Min', expand_y=True),
                        sg.InputText(field_value[0], size=(3, 3), enable_events=True, key=f"{field_key}$0", disabled=not attrs.mutable,
                                     disabled_readonly_background_color='white', expand_y=True),
                        sg.T('Max', expand_y=True),
                        sg.InputText(field_value[1], size=(3, 3), enable_events=True, key=f"{field_key}$1", disabled=not attrs.mutable,
                                     disabled_readonly_background_color='white', expand_y=True)
                    ]], pad=0, expand_y=True)
                    self.key_element_map[f"{field_key}$0"] = sg.InputText, int
                    self.key_element_map[f"{field_key}$1"] = sg.InputText, int
                elif attrs.special_entry == 'string_container':
                    value_field = sg.InputText(",".join(field_value) if field_value else '', enable_events=True,
                                               key=field_key, disabled=not attrs.mutable,
                                               disabled_readonly_background_color='white', expand_x=True, expand_y=True)
                    self.key_element_map[field_key] = sg.InputText, str
                else:
                    value_field = sg.InputText(str(field_value)[1:-1] if field_value else '', enable_events=True,
                                               key=field_key, disabled=not attrs.mutable,
                                               disabled_readonly_background_color='white', expand_x=True, expand_y=True)
                    self.key_element_map[field_key] = sg.InputText, str
            elif attrs.type is BarStepTick:
                b, s, t = (field_value.bar, field_value.step, field_value.tick) if field_value is not None else (0, 0, 0)
                value_field = sg.Col([[
                    sg.InputText(b, size=(3, 3), enable_events=True, key=f"{field_key}$0", disabled=not attrs.mutable,
                                 disabled_readonly_background_color='white', expand_y=True),
                    sg.T('Bar', pad=0, expand_y=True),
                    sg.InputText(s, size=(3, 3), enable_events=True, key=f"{field_key}$1", disabled=not attrs.mutable,
                                 disabled_readonly_background_color='white', expand_y=True),
                    sg.T('Step', pad=0, expand_y=True),
                    sg.InputText(t, size=(3, 3), enable_events=True, key=f"{field_key}$2",
                                 disabled=not attrs.mutable,
                                 disabled_readonly_background_color='white', expand_y=True),
                    sg.T('Tick', pad=0, expand_y=True)
                ]], pad=0, expand_y=True)
                self.key_element_map[f"{field_key}$0"] = sg.InputText, int
                self.key_element_map[f"{field_key}$1"] = sg.InputText, int
                self.key_element_map[f"{field_key}$2"] = sg.InputText, int
            else:
                value_field = sg.Text(field_value if field_value else '', key=field_key, expand_x=True, expand_y=True)
                self.key_element_map[field_key] = sg.Text, str
            layout_left.append([sg.Text(attrs.gui_name, expand_y=True, key=f"{field_key}$TEXT")])
            layout_right.append(value_field if isinstance(value_field, list) else [value_field])
        self.current_layout = [[sg.Col(layout_left, expand_x=True, expand_y=True, key=f'{self.key_prefix}$LEFT'), sg.Col(layout_right, expand_x=True, expand_y=True, key=f'{self.key_prefix}$RIGHT')]]
        return self.current_layout

    def update_entire_gui(self, window: sg.Window):
        if self.last_modified_param is not None:
            window.set_cursor('watch')
            try:
                if "!" in self.last_modified_param:
                    self.host.config_update(self.last_modified_param, None)
                else:
                    processed = self.convert_to_type(self.last_modified_param.split('$')[0], self.last_modified_value)
                    self.host.config_update(self.last_modified_param, processed)
            except:
                print(f"Cannot update value: {self.last_modified_value} to {self.last_modified_param}")
            finally:
                self.host.update_gui_configs()
            for k, attrs in self.items():
                field_key = f'{self.key_prefix}@{k}'
                field_value = attrs.value
                if attrs.type in (str, int, float):
                    window[field_key].update(value=field_value, disabled=not attrs.mutable)
                elif attrs.type is bool:
                    window[field_key].update(value=field_value, disabled=not attrs.mutable)
                elif issubclass(attrs.type, TOMIEnum):
                    window[field_key].update(value=field_value.name, disabled=not attrs.mutable)
                elif attrs.type is ItemSelector:
                    window[field_key].update(values=field_value.option_list(), value=field_value.current_key, disabled=not attrs.mutable)
                elif attrs.type in (set, list, tuple):
                    if attrs.special_entry == 'min_max_range':
                        window[f"{field_key}$0"].update(value=field_value[0], disabled=not attrs.mutable)
                        window[f"{field_key}$1"].update(value=field_value[1], disabled=not attrs.mutable)
                    elif attrs.special_entry == 'string_container':
                        window[field_key].update(value=",".join(field_value) if field_value else '', disabled=not attrs.mutable)
                    else:
                        window[field_key].update(value=str(field_value)[1:-1] if field_value else '', disabled=not attrs.mutable)
                elif attrs.type is TimeSignature:
                    window[f"{field_key}$0"].update(value=field_value.numerator, disabled=not attrs.mutable)
                    window[f"{field_key}$1"].update(value=field_value.denominator, disabled=not attrs.mutable)
                elif attrs.type is BarStepTick:
                    b, s, t = (field_value.bar, field_value.step, field_value.tick) if field_value is not None else (0, 0, 0)
                    window[f"{field_key}$0"].update(value=b, disabled=not attrs.mutable)
                    window[f"{field_key}$1"].update(value=s, disabled=not attrs.mutable)
                    window[f"{field_key}$2"].update(value=t, disabled=not attrs.mutable)
                elif attrs.type is Groove:
                    if field_value is not None:
                        field_value: Groove
                        values = [f'Type: {field_value.midi_type.name}', f'Length: {field_value.bar_length}',
                                  f"Speed: {field_value.speed.name}", f'ProgCount: {field_value.progression_count}',
                                  f"Groove: {field_value.groove.tolist()}"]
                    else:
                        values = ['Unavailable']
                    window[field_key].update(values=values, value=values[0], disabled=field_value is None)
                elif attrs.type is MIDIProcessor:
                    if field_value is not None:
                        field_value: MIDIProcessor
                        values = [f'Name: {field_value.name}', f'Type: {field_value.midi_type.name}', f'Length: {field_value.ceil_bst}',
                                  f'NoteCount: {len(field_value.get_notelist())}', f'Key: {field_value.key}',
                                  f'RootProgression: {field_value.progression_nums['major']}']
                    else:
                        values = ['Unavailable']
                    window[field_key].update(values=values, value=values[0], disabled=field_value is None)
                else:
                    window[field_key].update(value=field_value if field_value else '')
        window.set_cursor('arrow')
        self.last_modified_param = self.last_modified_value = None

    def items(self):
        return self.configs.items()

    def keys(self):
        return self.configs.keys()

    def values(self):
        return self.configs.values()

    def __getitem__(self, item):
        return self.configs[item]

    def __iter__(self):
        return iter(self.configs)