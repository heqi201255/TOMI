from tomi import OrderedDotdict
from tomi import config
import random


class ItemSelector(object):
    def __init__(self, host, name, item_type):
        self.host = host
        self.name = name
        self.item_type = item_type
        self.items = OrderedDotdict()
        self.current_key = None

    def get_current_value(self):
        return None if self.current_key is None else self.items[self.current_key]

    def __bool__(self) -> bool:
        return bool(self.items)

    def select(self, key = None):
        if key is None:
            self.current_key = list(self.items.keys())[0]
        elif key in self.items:
            self.current_key = key
        else:
            raise ValueError(f"Key '{key}' is not in items.")

    def gui_select(self, key):
        self.current_key = key
        self.host.on_selector_update(self.name)

    def random_select(self):
        if config.FIX_SEED is not None and isinstance(config.FIX_SEED, int):
            random.seed(config.FIX_SEED)
        if self.items:
            self.current_key = random.choice(list(self.items.keys()))

    def gui_random_select(self):
        self.random_select()
        self.host.on_selector_update(self.name)

    def option_list(self):
        return list(self.keys())

    def __len__(self):
        return len(self.items)

    def __setitem__(self, key, value):
        assert isinstance(value, self.item_type)
        self.items[key] = value

    def __getitem__(self, key):
        return self.items[key]

    def clear(self):
        self.items = OrderedDotdict()
        self.current_key = None

    def values(self):
        return self.items.values()

    def keys(self):
        return self.items.keys()
