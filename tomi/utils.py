import os
from collections import OrderedDict
import sys
import numpy as np
import random
from typing import Union, Sequence, Any, Callable
import hashlib


__all__ = ["ArrayLike", "ADSRSampleKeyMapping", "is_array_like", "doc_func", "suppress_stderr", "OrderedDotdict", "Dotdict",
           "HashableArray", "smart_div", "get_random_rgb", "to_json", "compute_md5"]

ArrayLike = Union[Sequence[Any], np.ndarray]

ADSRSampleKeyMapping = {0: "Unknown", 1: "C", 2: "CMaj", 3: "CMin", 4: "C#", 5: 'C#Maj', 6: 'C#Min', 7: 'D', 8: 'DMaj', 9: 'DMin',
                        10: 'D#', 11: 'D#Maj', 12: 'D#Min', 13: "E", 14: 'EMaj', 15: 'EMin', 16: 'F', 17: 'FMaj', 18: 'Fmin',
                        19: 'F#', 20: 'F#Maj', 21: 'F#Min', 22: 'G', 23: 'GMaj', 24: 'GMin', 25: 'G#', 26: 'G#Maj', 27: 'G#Min',
                        28: 'A', 29: 'AMaj', 30: 'AMin', 31: 'A#', 32: 'A#Maj', 33: 'A#Min', 34: 'B', 35: 'BMaj', 36: 'BMin'}


def compute_md5(text) -> int:
    md5 = hashlib.md5(text.encode('utf-8')).hexdigest()
    return int(md5[:15], 16)


def is_array_like(obj):
    return isinstance(obj, (Sequence, np.ndarray))


def doc_func(func: Callable, doc: str):
    func.__doc__ = doc
    return func


class suppress_stderr:
    def __enter__(self):
        self.errnull_file = open(os.devnull, 'w')
        self.old_stderr_fileno_undup    = sys.stderr.fileno()
        self.old_stderr_fileno = os.dup ( sys.stderr.fileno() )
        self.old_stderr = sys.stderr
        os.dup2 ( self.errnull_file.fileno(), self.old_stderr_fileno_undup )
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stderr = self.old_stderr
        os.dup2 ( self.old_stderr_fileno, self.old_stderr_fileno_undup )
        os.close ( self.old_stderr_fileno )
        self.errnull_file.close()


class OrderedDotdict(OrderedDict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = OrderedDict.get
    __setattr__ = OrderedDict.__setitem__
    __delattr__ = OrderedDict.__delitem__

    def __repr__(self):
        if self.keys():
            return "\n".join([f"    {k}: {v}" for k, v in self.items()])
        return '{}'


class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            return "\n".join([f"    {k}: {v}" for k, v in self.items()])
        return '{}'


class HashableArray:
    def __init__(self, arr_like, dtype=None, *args, **kwargs):
        arr = arr_like.arr if isinstance(arr_like, HashableArray) else arr_like
        self.arr = np.array(arr, dtype, *args, **kwargs)

    @property
    def shape(self):
        return self.arr.shape

    @property
    def size(self):
        return self.arr.size

    def __getitem__(self, item):
        a = self.arr.__getitem__(item)
        if isinstance(a, np.ndarray):
            return HashableArray(a)
        else:
            return a

    def __iter__(self):
        return self.arr.__iter__()

    def __next__(self):
        return self.arr.__next__()

    def __setitem__(self, key, value):
        self.arr.__setitem__(key, value)

    def __str__(self):
        return "h" + self.arr.__str__()

    def __repr__(self):
        return "h" + self.arr.__repr__()

    def __hash__(self):
        return hash((self.shape, self.arr.tobytes()))

    def __eq__(self, other):
        if isinstance(other, HashableArray):
            return np.array_equal(self.arr, other.arr)
        elif isinstance(other, np.ndarray):
            return np.array_equal(self.arr, other)
        return NotImplemented


def smart_div(a: int | float, b: int | float) -> int | float:
    """
    Division that returns an integer if the result should be an integer, otherwise a float.
    :param a: dividend
    :param b: divisor
    :return: integer if the decimal part of the result is 0, float otherwise.
    """
    result = a / b
    return int(result) if result.is_integer() else result


def get_random_rgb(r_range: tuple = None, g_range: tuple = None, b_range: tuple = None,
                   taken_colors: list = None, num_output: int = 1) -> list:
    def get_range(r):
        if not r or len(r) != 2 or r[0] < 0 or r[1] > 255 or r[1] <= r[0]:
            return 50, 200
        return r

    def random_color():
        nonlocal r_range, g_range, b_range
        return random.randint(*r_range), random.randint(*g_range), random.randint(*b_range)
    r_range, g_range, b_range = get_range(r_range), get_range(g_range), get_range(b_range)
    colors = set()
    if taken_colors is not None:
        # Generate a list of all possible colors
        possible_colors = [(r, g, b) for r in range(*r_range) for g in range(*g_range) for b in range(*b_range)]
        # Remove the input colors from the list of possible colors
        possible_colors = [color for color in possible_colors if color not in taken_colors]
        for i in range(num_output):
            color = random.choice(possible_colors)
            possible_colors.remove(color)
            colors.add(color)
    else:
        while len(colors) < num_output:
            colors.add(random_color())
    return list(colors)


def to_json(state: dict, indent: int = 4, save_path: str = None) -> str:
    """
    Custom json string encoder, the difference between this function and json.dump() is that this function renders list of
    strings and numbers in a single line, avoiding massive line counts where each value in the list occupies a line,
    enhancing the readability.
    @param save_path: if set, save to the json file.
    @param state: a dictionary, try the "__getstate__()" function of different variables.
    @param indent: the indentation format of the output.
    @return: the output json string.
    """
    def _encode(_state, _indent: int = 0, last_ele: bool = False):
        nonlocal jstr

        def process_value(value, _last_ele: bool = False):
            nonlocal jstr, indent
            if isinstance(value, dict):
                _encode(value, _indent + indent, _last_ele)
            elif isinstance(value, list):
                if all(isinstance(x, (int, float)) for x in value):
                    jstr += f'[{', '.join(map(str, value))}]{'' if _last_ele else ','}\n'
                elif all(isinstance(x, str) for x in value):
                    sstr = f'{'", "'.join(map(str, value))}'
                    jstr += f'["{sstr}"]{'' if _last_ele else ','}\n'
                else:
                    _encode(value, _indent + indent, _last_ele)
            else:
                if isinstance(value, bool):
                    jstr += f'{str(value).lower()}'
                elif isinstance(value, str):
                    jstr += f'"{value}"'
                else:
                    jstr += f'{value}'
                jstr += f'{'' if _last_ele else ','}\n'

        if isinstance(_state, dict):
            jstr += "{\n"
            for i, (k, v) in enumerate(_state.items()):
                jstr += f'{(_indent + indent) * " "}"{k}": '
                process_value(v, i == len(_state) - 1)
            jstr += _indent * " " + "}" + f"{'' if last_ele else ','}\n"
        elif isinstance(_state, list):
            jstr += "[\n"
            for i, v in enumerate(_state):
                jstr += f'{(_indent + indent) * " "}'
                process_value(v, i == len(_state) - 1)
            jstr += _indent * " " + f"]{'' if last_ele else ','}\n"
    jstr = ""
    _encode(state, last_ele=True)
    if save_path:
        with open(save_path.rstrip('.json')+".json", 'w') as f:
            f.write(jstr)
    return jstr

