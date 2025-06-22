import reapy
from reapy import reascript_api as RPR
from tomi import Project, ARRANGEMENT_FRONT_PADDING_BARS, BarStepTick, TrackType, MIDINoteList, PluginInfo, REAPER_PATH, SectionNode, TrackNode, AudioNode, MIDINode, TransformationNode
from contextlib import nullcontext
import psutil
import os
from collections import defaultdict


class REAPERController:
    def __init__(self, tomi_project: Project, stream_output: bool = False):
        if not self.reaper_is_running():
            raise ProcessLookupError('Reaper is not running, please start the program first.')
        self.project = reapy.Project()
        self.tomi_project = tomi_project
        self.stream_output = stream_output
        self.project.bpm = self.tomi_project.bpm
        self.node_bind = defaultdict(list)
        self.reaper_bind = {}

    @staticmethod
    def reaper_is_running():
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] in ('REAPER', 'reaper.exe'):
                return True
        return False

    @staticmethod
    def open_reaper():
        if not REAPERController.reaper_is_running():
            os.popen(f"open '{REAPER_PATH}'")
            return True
        return False

    def clear_project(self, project: reapy.Project):
        with reapy.inside_reaper() if not self.stream_output else nullcontext():
            for sec in project.regions:
                sec.delete()
            for track in project.tracks:
                track.delete()
        self.node_bind = defaultdict(list)
        self.reaper_bind = {}

    def bind(self, node, reaper_ele):
        self.node_bind[node].append(reaper_ele)
        self.reaper_bind[repr(reaper_ele)] = node

    @staticmethod
    def add_midi_to_track(track: reapy.Track, midi_nl: MIDINoteList, start_s: float, end_s: float):
        t_item = track.add_midi_item(start_s, end_s, quantize=False)
        t_take = t_item.active_take
        for note in midi_nl.get_note_list():
            t_take.add_note(note[1], note[2], note[0], note[3])
        return t_item

    @staticmethod
    def add_track(project: reapy.Project, name: str):
        return project.add_track(0, name)

    @staticmethod
    def add_instrument(track: reapy.Track, plugin_info: PluginInfo):
        t_inst = track.add_fx(plugin_info.plugin_name, even_if_exists=True)
        t_inst.preset = plugin_info.preset_path
        return t_inst

    @staticmethod
    def add_audio_to_track(track: reapy.Track, audio_path: str, insert_mode:int = 0, start_s: float = 0,
                           start_offset: float = None, playback_rate: float = None, pitch_shift: float = None,
                           item_length: float = None):
        track.make_only_selected_track()
        track.project.cursor_position = start_s
        RPR.InsertMedia(audio_path, insert_mode)
        inserted_item = None
        for media_item in track.items:
            if abs(media_item.position - start_s) <= 0.0001:
                inserted_item = media_item
                break
        if inserted_item:
            inserted_item.set_info_value('B_LOOPSRC', False)
            if playback_rate is not None:
                inserted_item.takes[0].set_info_value('D_PLAYRATE', playback_rate)
            if pitch_shift is not None:
                inserted_item.takes[0].set_info_value('D_PITCH', pitch_shift)
            if start_offset is not None:
                inserted_item.takes[0].set_info_value('D_STARTOFFS', start_offset)
            if item_length is not None:
                inserted_item.length = item_length
            return inserted_item
        return None

    def render_arrangement_view(self, track_arrangement: dict):
        self.project.stop()
        for track, track_content in track_arrangement.items():
            if not track_content:
                continue
            with reapy.inside_reaper() if not self.stream_output else nullcontext():
                t = self.add_track(self.project, track.name)
                self.bind(track, t)
                if track.track_type == TrackType.Midi:
                    if track.plugin.plugin_name is not None:
                        self.add_instrument(t, track.plugin)
                    for midi, midi_data in track_content.items():
                        self._add_midi_node(midi, midi_data, t)
                elif track.track_type == TrackType.Audio:
                    for audio, positions in track_content.items():
                        self._add_audio_node(audio, positions, t)
        self.project.cursor_position = 0
        self.project.play()

    def add_section_regions(self):
        start = ARRANGEMENT_FRONT_PADDING_BARS if ARRANGEMENT_FRONT_PADDING_BARS else 0
        start = BarStepTick.beat2sec(start*4, self.project.bpm)
        sections = [{'node': x, 'name': x.name, 'type': x.section_type.name, 'length': x.length.to_seconds(self.project.bpm)} for x in self.tomi_project.arrangement]
        for sec in sections:
            end = start + sec['length']
            region = self.project.add_region(start, end, sec['name'])
            self.bind(sec['node'], region)
            start = end

    def clear_section_regions(self):
        remove_nodes = []
        for node in self.node_bind:
            if isinstance(node, SectionNode):
                for region in self.node_bind[node]:
                    self.reaper_bind.pop(repr(region), None)
                    region.delete()
                remove_nodes.append(node)
        for node in remove_nodes:
            self.node_bind.pop(node, None)

    def add_curated_section_regions(self, section_pattern: list):
        """This method is used in StandaloneLLM experiment only."""
        start = ARRANGEMENT_FRONT_PADDING_BARS if ARRANGEMENT_FRONT_PADDING_BARS else 0
        start = BarStepTick.beat2sec(start*4, self.project.bpm)
        for sec in section_pattern:
            end = start + BarStepTick(sec[2]).to_seconds(self.project.bpm)
            self.project.add_region(start, end, sec[0])
            start = end

    def update_audio_node(self, track_arrangement: dict):
        with reapy.inside_reaper() if not self.stream_output else nullcontext():
            for track, track_content in track_arrangement.items():
                t = self.node_bind[track][0]
                if track.track_type == TrackType.Audio:
                    for audio, positions in track_content.items():
                        for audio_item in self.node_bind[audio]:
                            self.reaper_bind.pop(repr(audio_item), None)
                            audio_item.delete()
                        self.node_bind.pop(audio, None)
                        self._add_audio_node(audio, positions, t)

    def update_midi_node(self, track_arrangement: dict):
        with reapy.inside_reaper() if not self.stream_output else nullcontext():
            for track, track_content in track_arrangement.items():
                t = self.node_bind[track][0]
                if track.track_type == TrackType.Midi:
                    for midi, midi_data in track_content.items():
                        for midi_item in self.node_bind[midi]:
                            self.reaper_bind.pop(repr(midi_item), None)
                            midi_item.delete()
                        self.node_bind.pop(midi, None)
                        self._add_midi_node(midi, midi_data, t)

    def _add_audio_node(self, audio: AudioNode, positions: list, track: reapy.Track):
        insert_mode = 0 | 8192 if audio.reverse else 0
        for pos in [[BarStepTick.beat2sec(b, self.project.bpm) for b in list(x)] for x in positions]:
            if len(pos) == 0:
                continue
            start_s, end_s, offset_s = pos
            playback_rate = self.project.bpm / audio.current_sample.tempo if audio.loop else None
            pitch_shift = audio.transpose_steps if audio.fit_key else None
            start_offset = offset_s if offset_s != 0 else None
            item_length = end_s - start_s
            audio_item = self.add_audio_to_track(track, audio.current_sample.path, insert_mode, start_s,
                                                 start_offset,
                                                 playback_rate, pitch_shift, item_length)
            self.bind(audio, audio_item)

    def _add_midi_node(self, midi: MIDINode, midi_data: list, track: reapy.Track):
        for midi_nl, st, ed in midi_data:
            midi_item = self.add_midi_to_track(track, midi_nl, st, ed)
            self.bind(midi, midi_item)
