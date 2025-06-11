from .. import NodeBase
from tomi import NodeType, PluginInfo, TrackType, AddInstrumentOnAudioTrackError, PluginFormat, ItemSelector


class TrackNode(NodeBase):
    def __init__(self,
                 project,
                 name: str,
                 track_type: TrackType = TrackType.Midi,
                 volume: int = 100,
                 pan: int = 0,
                 plugin_name: str = None,
                 plugin_preset: str = None,
                 plugin_format: PluginFormat = None):
        super().__init__(NodeType.Track, project, name,
                         parent_accept=[NodeType.MidiClip, NodeType.AudioClip])
        assert -100 <= pan <= 100, "Pan value must be between -100 and 100"
        assert 0 <= volume <= 100, "Volume value must be between 0 and 100"
        self.track_type = track_type
        if self.track_type == TrackType.Midi:
            self.parent_accept.remove(NodeType.AudioClip)
        elif self.track_type == TrackType.Audio:
            self.parent_accept.remove(NodeType.MidiClip)
        self.volume = volume
        self.pan = pan
        self.plugin = PluginInfo(host_node=self)
        self.plugin_candidates = ItemSelector(self, 'plugin_candidates', str)
        self.preset_candidates = ItemSelector(self, 'preset_candidates', str)
        if self.track_type == TrackType.Midi:
            self.set_plugin(plugin_name, plugin_preset, plugin_format)
        self.parameters = []

    def update_gui_configs(self):
        self.gui_configs.update_config(param_name='id', gui_name='Node ID', param_type=int, value=self.id, mutable=False)
        self.gui_configs.update_config(param_name='name', gui_name='Node Name', param_type=str, value=self.gui_name, mutable=True)
        self.gui_configs.update_config(param_name='_parents', gui_name='Parent Nodes', param_type=set, value={n.gui_name for n in self._parents}, mutable=False, special_entry='string_container')
        self.gui_configs.update_config(param_name='track_type', gui_name='Track Type', param_type=TrackType, value=self.track_type, mutable=True)
        self.gui_configs.update_config(param_name='volume', gui_name='Volume', param_type=int, value=self.volume, mutable=True)
        self.gui_configs.update_config(param_name='pan', gui_name='Pan', param_type=int, value=self.pan, mutable=True)
        self.gui_configs.update_config(param_name='plugin', gui_name='Instrument', param_type=str, value=self.plugin.plugin_name, mutable=self.track_type == TrackType.Midi)
        self.gui_configs.update_config(param_name='preset', gui_name='Preset', param_type=str, value=self.plugin.preset_path, mutable=self.track_type == TrackType.Midi)

    def config_update(self, param_name: str, param_value):
        name, loc = self._get_param_name_and_loc(param_name)
        match name:
            case 'name': self.name = param_value
            case 'track_type':  self.set_track_type(param_value)
            case 'volume': self.set_volume(param_value)
            case 'pan': self.set_pan(param_value)
        self.need_sync = True

    def set_volume(self, param_value: int):
        assert isinstance(param_value, int) and 0 <= param_value <= 100, "Volume value must be between 0 and 100"
        self.volume = param_value

    def set_pan(self, pan: int):
        assert isinstance(pan, int) and -100 <= pan <= 100, "Pan value must be between -100 and 100"
        self.pan = pan

    def set_params(self, parameters: list):
        self.parameters = parameters

    def print_params(self):
        for param in self.parameters:
            self.log(f"{param['index']} {param['name']} {param['text']}{param['label']}")

    def set_track_type(self, track_type: TrackType):
        assert isinstance(track_type, TrackType)
        if self.track_type != track_type:
            if track_type == TrackType.Midi:
                for node in self.parents:
                    if node.node_type == NodeType.AudioClip:
                        self.remove_parent(node)
            elif track_type == TrackType.Audio:
                for node in self.parents:
                    if node.node_type == NodeType.MidiClip:
                        self.remove_parent(node)
            self.track_type = track_type

    def set_plugin(self, plugin_name: str = None, plugin_preset: str = None, plugin_format: PluginFormat = None):
        if self.track_type != TrackType.Midi:
            raise AddInstrumentOnAudioTrackError(f"Instrument is not allowed on {self.track_type.name} tracks.")
        if plugin_name is not None:
            self.plugin.set_plugin(plugin_name, plugin_preset, plugin_format)

    def equals(self, other):
        return isinstance(other, TrackNode) and self.plugin == other.plugin and self.track_type == other.track_type and self.volume == other.volume and self.pan == other.pan
