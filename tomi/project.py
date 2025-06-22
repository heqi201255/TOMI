from tomi import (Key,
                  Mode,
                  KeyMode,
                  SongGenre,
                  MIDI_DB_ADDRESS,
                  SAMPLE_DB_ADDRESS,
                  printer,
                  ARRANGEMENT_FRONT_PADDING_BARS,
                  GUIContentConfigs,
                  NodeGraph,
                  ClipNode,
                  TrackNode,
                  MIDINode,
                  AudioNode, TransformationNode,
                  GeneralTransformNode, SectionNode, DrumTransformNode, FxTransformNode, FillTransformNode,
                  TimeSignature
                  )
import sqlite3


class Project:
    def __init__(self, project_name: str = None, bpm: float = 120., time_signature: TimeSignature = None, key: Key = Key.C, mode: Mode = Mode.Major, genre: SongGenre = SongGenre.Pop):
        if project_name is None:
            project_name = "Untitled"
        time_signature = TimeSignature.default() if time_signature is None else time_signature
        assert isinstance(project_name, str)
        assert isinstance(bpm, (int, float))
        assert isinstance(time_signature, TimeSignature)
        assert isinstance(key, Key)
        assert isinstance(mode, Mode)
        assert isinstance(genre, SongGenre)
        self.song_name = project_name
        self.bpm = bpm
        self.time_signature = time_signature
        self.key = key
        self.mode = mode
        self.genre = genre
        self.midi_lib = {}
        self.sample_lib = {}
        self.arrangement = []
        self.nodes = set()
        self._last_node_id = 0
        self.node_graph = NodeGraph()
        self.databases = {}
        self.gui_configs = GUIContentConfigs(self)
        self.project_settings_updated = False
        self.init_data_source()

    @property
    def key_mode(self):
        return KeyMode(self.key, self.mode)

    def get_new_node_id(self):
        node_id = self._last_node_id
        self._last_node_id += 1
        return node_id

    def update_gui_configs(self):
        self.gui_configs.update_config(param_name='song_name', gui_name='Song Name', param_type=str, value=self.song_name, mutable=True)
        self.gui_configs.update_config(param_name='bpm', gui_name='BPM', param_type=float, value=self.bpm, mutable=True)
        self.gui_configs.update_config(param_name='time_signature', gui_name='Time Signature', param_type=TimeSignature, value=self.time_signature, mutable=True)
        self.gui_configs.update_config(param_name='key', gui_name='Key', param_type=Key, value=self.key, mutable=True)
        self.gui_configs.update_config(param_name='mode', gui_name='Mode', param_type=Mode, value=self.mode, mutable=True)
        self.gui_configs.update_config(param_name='genre', gui_name='Genre', param_type=SongGenre, value=self.genre, mutable=True)
        self.gui_configs.update_config(param_name='node_count', gui_name='Node Count', param_type=int, value=len(self.nodes), mutable=False)

    def config_update(self, param_name: str, param_value):
        self.__setattr__(param_name, param_value)
        self.project_settings_updated = True

    def run(self):
        self.node_graph.sync_nodes()
        clip_render_order = []
        for node in self.clip_nodes:
            if node in clip_render_order:
                continue
            anode = node
            dependent_l = [anode]
            while anode.dependent_node:
                anode = anode.dependent_node
                if anode not in clip_render_order:
                    dependent_l.insert(0, anode)
            clip_render_order.extend(dependent_l)
        for node in printer.progress(clip_render_order, desc="Running Clip Nodes", unit="nodes"):
            node.clear()
            node.run()
        for node in printer.progress(self.transformation_nodes, desc='Running Transformation Nodes', unit="nodes"):
            node.clear()
            node.run()
        self.project_settings_updated = False

    @property
    def song_beat_length(self):
        return ARRANGEMENT_FRONT_PADDING_BARS * 4 + sum([arrangement.length.to_beats() for arrangement in self.arrangement])

    def set_arrangement(self, arrangements: list[SectionNode]):
        self.arrangement = arrangements

    def rank(self, nodes: list = None, with_usage: bool = False):
        if nodes is None:
            nodes = self.nodes
        return sorted(nodes, key=lambda n: n.usage, reverse=True) if not with_usage else sorted([(node, node.usage) for node in nodes], key=lambda n: n[1], reverse=True)

    @property
    def section_nodes(self):
        return [node for node in self.nodes if isinstance(node, SectionNode)]

    @property
    def transformation_nodes(self):
        return [node for node in self.nodes if isinstance(node, TransformationNode)]

    @property
    def clip_nodes(self):
        return [node for node in self.nodes if isinstance(node, ClipNode)]

    @property
    def general_transformation_nodes(self):
        return [node for node in self.nodes if isinstance(node, GeneralTransformNode)]

    @property
    def drum_transformation_nodes(self):
        return [node for node in self.nodes if isinstance(node, DrumTransformNode)]

    @property
    def fx_transformation_nodes(self):
        return [node for node in self.nodes if isinstance(node, FxTransformNode)]

    @property
    def fill_transformation_nodes(self):
        return [node for node in self.nodes if isinstance(node, FillTransformNode)]

    @property
    def track_nodes(self):
        nodes = [node for node in self.nodes if isinstance(node, TrackNode)]
        nodes.sort(key=lambda node: node.name)
        return nodes

    @property
    def midi_nodes(self):
        return [node for node in self.clip_nodes if isinstance(node, MIDINode)]

    @property
    def audio_nodes(self):
        return [node for node in self.clip_nodes if isinstance(node, AudioNode)]

    @property
    def unused_nodes(self):
        return [node for node in self.nodes if node not in self.node_graph.registered_nodes]

    def remove_unused_nodes(self):
        for node in self.unused_nodes:
            self.nodes.remove(node)
            if node in self.arrangement:
                self.arrangement = [n for n in self.arrangement if n != node]

    def init_data_source(self):
        self.databases["midiDB"] = sqlite3.connect(MIDI_DB_ADDRESS)
        self.databases["midiCur"] = self.databases["midiDB"].cursor()
        self.databases["audioDB"] = sqlite3.connect(SAMPLE_DB_ADDRESS)
        self.databases["audioCur"] = self.databases["audioDB"].cursor()

    def node_auto_naming(self, name: str = None):
        current_names = [node.name for node in self.nodes]
        new_name = name
        ind = 1
        while new_name in current_names:
            new_name = f"{name}_#{ind}"
            ind += 1
        return new_name

