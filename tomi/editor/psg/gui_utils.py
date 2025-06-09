from tomi import NodeType, Dotdict, TOMIEnum, TOMIEnumDescriptor
from tkinter import Tk, font
from . import sg

__all__ = ["measure_text_size", "SelectedItem", "WidgetState", "get_max_size_of_all_child_element_if_available",
           "get_all_child_element_keys_if_available", "get_input_info_from_event", "create_column",
           "create_frame", "remove_element", "remove_key_from_window", "rgb_to_hex", "hex_to_rgb", "dim_color", "NodeColors"]

def measure_text_size(text: str | list, font_properties):
    root = Tk()
    custom_font = font.Font(root, font_properties)
    if isinstance(text, str):
        width = custom_font.measure(text)
    else:
        width = [custom_font.measure(i) for i in text]
    root.destroy()
    return width


class SelectedItem(TOMIEnum):
    Empty = TOMIEnumDescriptor('Empty')
    Rect = TOMIEnumDescriptor('Rect')
    Input = TOMIEnumDescriptor('Input')
    Output = TOMIEnumDescriptor('Output')


class WidgetState(TOMIEnum):
    Dim = TOMIEnumDescriptor('Dim')
    Normal = TOMIEnumDescriptor('Normal')
    Highlight = TOMIEnumDescriptor('Highlight')
    Selected = TOMIEnumDescriptor('Selected')


def get_all_child_element_keys_if_available(element):
    keys = []
    for row in element.Rows:
        for ele in row:
            if ele.key is not None:
                keys.append(ele.key)
            if hasattr(ele, 'Rows'):
                keys += get_all_child_element_keys_if_available(ele)
    return keys

def get_max_size_of_all_child_element_if_available(element):
    height = width = 0
    for row in element.Rows:
        for ele in row:
            if hasattr(ele, 'get_size'):
                w, h = ele.get_size()
                width, height = max(width, w), max(height, h)
            if hasattr(ele, 'Rows'):
                w, h = get_all_child_element_keys_if_available(ele.Rows)
                width, height = max(width, w), max(height, h)
    return width, height

def create_frame(layout: list[list], size: tuple[int, int] = None, key: str = None, title: str = '', color: str = None, visible: bool = True):
    return sg.Frame(title, layout, size=size, background_color=color, key=key, pad=0,
                    expand_x=True, expand_y=True, visible=visible)

def create_column(layout: list[list], size: tuple[int, int] = None, key: str = None, color: str = None):
    return sg.Col(layout, background_color=color, size=size, key=key, expand_x=True, expand_y=True)


def remove_key_from_window(window: sg.Window, key: str):
    if key in window.AllKeysDict:
        del window.AllKeysDict[key]

def remove_element(element, window):
    if element is not None:
        child_keys = get_all_child_element_keys_if_available(element)
        element.Widget.master.destroy()
        remove_key_from_window(window, element.key)
        for child_ele in child_keys:
            remove_key_from_window(window, child_ele)

def get_input_info_from_event(configs, param_name, param_value):
    try:
        processed = configs[param_name]['type'](param_value)
        configs[param_name]['value'] = param_value
        return processed
    except ValueError:
        return None

def rgb_to_hex(rgb):
    """Convert an RGB tuple (255, 255, 255) to a hex string #RRGGBB."""
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def hex_to_rgb(hex_color: str) -> tuple:
    # Remove the hash symbol if present
    hex_color = hex_color.lstrip('#')
    # Convert hex to RGB
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def dim_color(hex_color: str, factor: float = 0.5) -> str:
    """
    Dim the given RGB color toward white.

    Parameters:
    rgb (tuple): A tuple of (R, G, B) values ranging from 0 to 255.
    factor (float): A float between 0 and 1, indicating how much to blend toward white.
                    A factor of 0 returns the original color, while a factor of 1 returns white.

    Returns:
    tuple: A new dimmed RGB tuple.
    """
    r, g, b = hex_to_rgb(hex_color)
    dimmed_r = int(r + (255 - r) * factor)
    dimmed_g = int(g + (255 - g) * factor)
    dimmed_b = int(b + (255 - b) * factor)
    return rgb_to_hex((dimmed_r, dimmed_g, dimmed_b))

NodeColors = {
    NodeType.AudioClip: Dotdict({
        "rect_color": rgb_to_hex([124, 194, 112]),
        'input_color': '#53FF50',
        'output_color': '#FF221C',
    }),
    NodeType.MidiClip: Dotdict({
        "rect_color": rgb_to_hex([124, 194, 112]),
        'input_color': '#53FF50',
        'output_color': '#FF221C',
    }),
    NodeType.Section: Dotdict({
        "rect_color": rgb_to_hex([118, 194, 241]),
        'input_color': '#53FF50',
        'output_color': '#FF221C',
    }),
    NodeType.GeneralTransform: Dotdict({
        "rect_color": rgb_to_hex([240, 207, 96]),
        'input_color': '#53FF50',
        'output_color': '#FF221C',
    }),
    NodeType.DrumTransform: Dotdict({
        "rect_color": rgb_to_hex([240, 207, 96]),
        'input_color': '#53FF50',
        'output_color': '#FF221C',
    }),
    NodeType.FxTransform: Dotdict({
        "rect_color": rgb_to_hex([240, 207, 96]),
        'input_color': '#53FF50',
        'output_color': '#FF221C',
    }),
    NodeType.FillTransform: Dotdict({
        "rect_color": rgb_to_hex([240, 207, 96]),
        'input_color': '#53FF50',
        'output_color': '#FF221C',
    }),
    NodeType.Track: Dotdict({
        "rect_color": rgb_to_hex([225, 145, 51]),
        'input_color': '#53FF50',
        'output_color': '#FF221C',
    })
}


