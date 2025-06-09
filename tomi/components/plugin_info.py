from . import PluginFormat


class PluginInfo:
    def __init__(self, host_node=None, plugin_name: str = None, preset_path: str = None, plugin_format: PluginFormat = None):
        self.host_node = host_node
        self.plugin_name = plugin_name
        self.preset_path = preset_path
        self.path = ""
        self.format = plugin_format

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['host_node']
        return state

    def set_plugin(self, plugin_name: str = None, preset_path: str = None, plugin_format: PluginFormat = None):
        self.__init__(self.host_node, plugin_name, preset_path, plugin_format)

    def set_preset(self, preset_path: str):
        self.preset_path = preset_path

    def __eq__(self, other):
        return isinstance(other, PluginInfo) and self.format == other.format and self.path == other.path and self.plugin_name == other.plugin_name and self.preset_path == other.preset_path

    def __repr__(self):
        if self.format:
            return f"<{self.format.name}Plugin>{self.plugin_name} - {self.preset_path}" if self.preset_path else f"<{self.format.name}Plugin>{self.plugin_name}"
        else:
            return f"<Plugin>{self.plugin_name} - Uninitialized"


if __name__ == '__main__':
    p = PluginInfo()
    print(p)
