import os.path
from tomi import (Project, TOMIEngine, SongGenre, SectionNode, TrackType, MIDIType,
                   AudioType, BarStepTick, GrooveSpeed, NodeFactory, DEFAULT_INSTRUMENT_PLUGIN, DEFAULT_INSTRUMENT_PRESET, Key, KeyMode)
from .tomi_llm_request import TOMILLMRequest
import json


class TOMISong:
    def __init__(self,
                 song_name: str = "tomi_song",
                 song_genre: SongGenre = SongGenre.Pop,
                 song_blocks: dict | str = None,
                 key: Key | KeyMode = Key.C):
        self.song_genre = song_genre
        key = KeyMode(key) if isinstance(key, Key) else key
        self.project = Project(song_name, key=key.key, mode=key.mode, genre=self.song_genre)
        self.engine = TOMIEngine(self.project)
        if song_blocks is None:
            self.song_blocks = TOMILLMRequest().generate_song_blocks(self.song_genre)
        elif isinstance(song_blocks, str):
            if not os.path.exists(song_blocks):
                raise FileNotFoundError(song_blocks)
            self.song_blocks = json.load(open(song_blocks))
        elif isinstance(song_blocks, dict):
            self.song_blocks = song_blocks
        else:
            raise TypeError("song_blocks must be a str or dict")
        self.arrangement = self.song_blocks['Structure']
        self.sections = self.song_blocks['Sections']
        self.tracks = self.song_blocks['Tracks']
        self.clips = self.song_blocks['Clips']
        self.transformations = self.song_blocks['Transformations']
        self.nodelinks = self.song_blocks['Links']
        self.factory = NodeFactory(self.project)
        self.gui = None
        self.track_nodes = {}
        self.section_nodes = {}
        self.clip_nodes = {}
        self.transform_nodes = {}

    def cast_sections(self):
        sections = SectionNode.get_arrangement_nodes(self.project, self.sections)
        for i in range(len(self.sections)):
            self.section_nodes[self.sections[i][0]] = sections[i]
        self.arrangement = [self.section_nodes[x] for x in self.arrangement]

    def cast_track(self, track_attrs: tuple):
        track_name, track_type = track_attrs
        track_type = TrackType.get_object_by_name(track_type)
        if track_type == TrackType.Midi:
            self.track_nodes[track_name] = self.factory.track(track_type, plugin_name=DEFAULT_INSTRUMENT_PLUGIN, plugin_preset=DEFAULT_INSTRUMENT_PRESET, node_name=track_name)
        else:
            self.track_nodes[track_name] = self.factory.track(track_type, node_name=track_name)

    def cast_clips(self, clip_attrs: tuple):
        if clip_attrs[1] == 'Midi':
            clip_name, _, midi_type, midi_length_in_bars, midi_groove_speed, dependent_midi, root_progression = clip_attrs
            midi_type, midi_groove_speed = MIDIType.get_object_by_name(midi_type), GrooveSpeed.get_object_by_name(midi_groove_speed)
            self.clip_nodes[clip_name] = self.factory.midi(midi_type, BarStepTick(midi_length_in_bars), groove_speed=midi_groove_speed, dependent_node=self.clip_nodes[dependent_midi] if dependent_midi is not None else None, dependent_type=MIDIType.Bass if dependent_midi is not None else None, root_progression=root_progression, node_name=clip_name)
        else:
            clip_name, _, audio_type, query, loop, reverse = clip_attrs
            audio_type = AudioType.get_object_by_name(audio_type)
            fit_tempo = loop
            fit_key = loop
            self.clip_nodes[clip_name] = self.factory.audio(audio_type, query, loop, (self.project.bpm-20, self.project.bpm+20), reverse=reverse, fit_key=fit_key, fit_tempo=fit_tempo, node_name=clip_name)

    def cast_transformations(self, transform_attrs: tuple):
        attrs = []
        for x in transform_attrs:
            if isinstance(x, tuple):
                attrs.extend(x)
            else:
                attrs.append(x)
        transform_type = attrs[1]
        if transform_type == 'general_transform':
            transform_name, transform_type, action_sequence = attrs
            self.transform_nodes[transform_name] = self.factory.general_transform(action_sequence, node_name=transform_name, loop=True)
        elif transform_type == 'drum_transform':
            transform_name, transform_type, action_sequence = attrs
            self.transform_nodes[transform_name] = self.factory.drum_transform(action_sequence, node_name=transform_name, loop=True)
        elif transform_type == 'fill_transform':
            transform_name, transform_type = attrs
            self.transform_nodes[transform_name] = self.factory.fill_transform(node_name=transform_name, loop_bars=8)
            # self.transform_nodes[transform_name] = self.factory.fx_transform(front=False, node_name=transform_name)
        else:
            transform_name, transform_type, is_faller = attrs
            self.transform_nodes[transform_name] = self.factory.fx_transform(is_faller=is_faller, node_name=transform_name)

    def cast_nodelink(self, link: str):
        s, f, c, t = link.split('->')
        return self.section_nodes[s] >> self.transform_nodes[f] >> self.clip_nodes[c] >> self.track_nodes[t]

    def print_step(self):
        TOMIEngine(self.project).run_reaper(True)

    def gen(self, stream_output: bool = False, open_editor: bool = True):
        self.cast_sections()
        self.project.set_arrangement(self.arrangement)
        for t in self.tracks:
            self.cast_track(t)
        ordered_clips = []
        for c in self.clips:
            if c[1] == 'Midi' and c[5] is not None:
                ordered_clips.append(c)
            else:
                ordered_clips.insert(0, c)
        for c in ordered_clips:
            self.cast_clips(c)
        for f in self.transformations:
            self.cast_transformations(f)
        for nl in self.nodelinks:
            self.project.node_graph.add(self.cast_nodelink(nl))
        self.project.remove_unused_nodes()
        self.project.node_graph.print_link("Arrangement Structure")
        self.engine.run_reaper(stream_output=stream_output)
        if open_editor:
            from tomi.editor import TOMIEditor
            self.gui = TOMIEditor(self.engine)
            self.gui.run()
