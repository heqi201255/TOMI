from tomi import (SectionType,
                   MIDIType, BarStepTick, AudioType, TrackType,
                   MIDIProcessor, PluginFormat, GrooveSpeed, Groove)
from . import (SectionNode, ClipNode, AudioNode, MIDINode,
               GeneralTransformNode, DrumTransformNode, FxTransformNode, TrackNode, FillTransformNode)
import numpy as np


class NodeFactory:
    def __init__(self, project):
        self.project = project

    def create_node(self, node_name, node_class, prefix, *args, **kwargs):
        if node_name is None:
            node_name = self.project.node_auto_naming(f"{prefix}_{node_class.__name__}")
        else:
            node_name = self.project.node_auto_naming(f"{prefix}_{node_name}")
        node = node_class(self.project, node_name, *args, **kwargs)
        self.project.node_graph.register_node(node)
        return node

    def section(self, section_type: SectionType, section_length: BarStepTick, node_name: str = None):
        return self.create_node(node_name, SectionNode, 'sec', section_type, section_length)

    def track(self, track_type: TrackType = TrackType.Midi,
              volume: int = 100, pan: int = 0, plugin_name: str = None, plugin_preset: str = None,
              plugin_format: PluginFormat = None, node_name: str = None
              ):
        return self.create_node(node_name, TrackNode, 'tr', track_type,volume, pan, plugin_name, plugin_preset, plugin_format)

    def midi(self, midi_type: MIDIType, length: BarStepTick = None, dynamic=None, complexity=None,
             groove_speed: GrooveSpeed = GrooveSpeed.Normal, groove: Groove = None,
             midi: MIDIProcessor | str = None,
             dependent_node: 'ClipNode' = None,
             dependent_type: MIDIType = None,
             root_progression: list = None,
             min_progression_count: int = 4,
             node_name: str = None):
        return self.create_node(node_name, MIDINode, "m", midi_type, length, dynamic, complexity,
                                groove_speed, groove, midi, dependent_node, dependent_type, root_progression,
                                min_progression_count)

    def audio(self, audio_type: AudioType, query: list[str] = None, loop: bool = True,
              bpm_range: tuple[int | float, int | float] = (0, 999),
              audio_file_path: str = None, minlen: BarStepTick = None, maxlen: BarStepTick = None,
              reverse: bool = False,
              fit_key: bool = False, transpose_steps: int = 0, fit_tempo: bool = False, node_name: str = None):
        return self.create_node(node_name, AudioNode, "a", audio_type, query, loop, bpm_range, audio_file_path, minlen, maxlen,
                                reverse, fit_key, transpose_steps, fit_tempo)

    def general_transform(self, action_sequence: list | np.ndarray[np.int8] = GeneralTransformNode.DEFAULT_ACTION_SEQUENCE, loop: bool = True,
                          node_name: str = None):
        return self.create_node(node_name, GeneralTransformNode, "general", action_sequence, loop)

    def drum_transform(self, action_sequence: list | np.ndarray[np.int8] = GeneralTransformNode.DEFAULT_ACTION_SEQUENCE, loop: bool = False,
                       node_name: str = None):
        return self.create_node(node_name, DrumTransformNode, "drum", action_sequence, loop)

    def fx_transform(self, action_sequence: list | np.ndarray[np.int8] = None, offset: int = None,
                     is_faller: bool = False, node_name: str = None):
        return self.create_node(node_name, FxTransformNode, "fx", action_sequence, offset, is_faller)

    def fill_transform(self, action_sequence: list | np.ndarray[np.int8] = None, loop_bars: float = None, offset: int = None, node_name: str = None):
        return self.create_node(node_name, FillTransformNode, "fill", action_sequence, loop_bars, offset)
