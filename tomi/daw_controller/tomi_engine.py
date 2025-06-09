from tomi import (Project, ARRANGEMENT_FRONT_PADDING_BARS, MIDINode, AudioNode, BarStepTick, TrackType,
                   ProjectVisualizer, printer, TrackNode, TimeSignature, LinkType)
from . import REAPERController
import pretty_midi
import numpy as np
import os


class TOMIEngine:
    def __init__(self, project: Project):
        self.project = project
        self.track_arrangement = {}
        self.render_engine = None

    @property
    def time_signature(self) -> TimeSignature:
        return self.project.time_signature

    def run_reaper(self, stream_output: bool = False, section_pattern: list = None):
        printer.print(printer.header(f"Rendering <{self.project.song_name}> @Reaper", background_func=printer.Bgreen))
        self.init_run()
        self.render_engine = REAPERController(self.project, stream_output)
        self.render_engine.clear_project(self.render_engine.project)
        if section_pattern is not None:
            self.render_engine.add_curated_section_regions(section_pattern)
        else:
            self.render_engine.add_section_regions()
        self.render_engine.render_arrangement_view(self.track_arrangement)

    def init_run(self):
        printer.print(printer.header("Rendering Arrangement Graph"))
        self.render_arrangement()
        viz = ProjectVisualizer(self)
        viz.print_playlist()

    def render_arrangement(self):
        graph = self.project.node_graph.arrangement_links
        self.track_arrangement = {}
        track_content_alloc = {}
        for track in self.project.track_nodes:
            self.track_arrangement[track] = {}
            track_content_alloc[track] = {section: {} for section in self.project.section_nodes}
        self.project.run()
        for section_node in printer.progress(self.project.section_nodes, desc="Running Section Nodes", unit="sections"):
            for transformation_node in graph[section_node]:
                for clip_node in graph[section_node][transformation_node]:
                    if isinstance(clip_node, MIDINode):
                        ranges = transformation_node.outputs[section_node.length][clip_node.length]["ranges"]
                        midi_nl = transformation_node.get_midi_notelist(section_node, clip_node)
                        if clip_node.fit_key:
                            midi_nl.move_key(clip_node.transpose_steps)
                        data = (midi_nl, ranges[0][0][0], ranges[-1][-1][-1])
                    else:
                        data = transformation_node.get_audio_positions(section_node, clip_node)
                    for track_node in graph[section_node][transformation_node][clip_node]:
                        if clip_node not in track_content_alloc[track_node][section_node]:
                            track_content_alloc[track_node][section_node][clip_node] = []
                        track_content_alloc[track_node][section_node][clip_node].append(data)
        current_bar = BarStepTick(ARRANGEMENT_FRONT_PADDING_BARS, time_signature=self.time_signature)
        for section_node in printer.progress(self.project.arrangement, desc="Finalize Arrangement", unit="sections"):
            for track_node in track_content_alloc:
                content: dict = track_content_alloc[track_node][section_node]
                if track_node.track_type == TrackType.Audio:
                    for clip, positions in content.items():
                        final_positions = np.vstack(positions)
                        final_positions[:, 0:2] += current_bar.to_beats()
                        if clip not in self.track_arrangement[track_node]:
                            self.track_arrangement[track_node][clip] = []
                        self.track_arrangement[track_node][clip].extend(final_positions)
                else:
                    for clip, midi_data in content.items():
                        for m_data in midi_data:
                            midi_nl, st, ed = m_data
                            offset = current_bar.to_seconds(bpm=self.project.bpm)
                            if clip not in self.track_arrangement[track_node]:
                                self.track_arrangement[track_node][clip] = []
                            self.track_arrangement[track_node][clip].append([midi_nl, st+offset, ed+offset])
            current_bar += section_node.length
        for node in self.project.nodes:
            node.need_sync = False

    def sync_audio_nodes(self):
        for node in self.project.audio_nodes:
            if node.need_sync:
                node.need_sync = False
                link = self.project.node_graph[node][LinkType.ArrangementLink].reorder(['T', 'F', 'S'])
                track_alloc = {track: {section: {node: transformation.get_audio_positions(section, node)} for transformation in link[track] for section in link[track][transformation]} for track in link}
                current_bar = BarStepTick(ARRANGEMENT_FRONT_PADDING_BARS, time_signature=self.time_signature)
                track_arrangement = {track: {node: []} for track in track_alloc}
                for section_node in self.project.arrangement:
                    for track_node in track_alloc:
                        if section_node in track_alloc[track_node]:
                            for clip, positions in track_alloc[track_node][section_node].items():
                                final_positions = np.vstack(positions)
                                final_positions[:, 0:2] += current_bar.to_beats()
                                track_arrangement[track_node][clip].extend(final_positions)
                    current_bar += section_node.length
                self.track_arrangement.update(track_arrangement)
                self.render_engine.update_audio_node(track_arrangement)

    def sync_midi_nodes(self):
        for node in self.project.midi_nodes:
            if node.need_sync:
                node.need_sync = False
                link = self.project.node_graph[node][LinkType.ArrangementLink].reorder(['T', 'F', 'S'])
                track_alloc = {}
                for track in link:
                    track_alloc[track] = {}
                    for transformation in link[track]:
                        for section in link[track][transformation]:
                            track_alloc[track][section] = {}
                            ranges = transformation.outputs[section.length][node.length]["ranges"]
                            midi_nl = transformation.get_midi_notelist(section, node)
                            if node.fit_key:
                                midi_nl.move_key(node.transpose_steps)
                            if node not in track_alloc[track][section]:
                                track_alloc[track][section][node] = []
                            track_alloc[track][section][node].append((midi_nl, ranges[0][0][0], ranges[-1][-1][-1]))
                current_bar = BarStepTick(ARRANGEMENT_FRONT_PADDING_BARS, time_signature=self.time_signature)
                track_arrangement = {track: {node: []} for track in track_alloc}
                for section_node in self.project.arrangement:
                    for track_node in track_alloc:
                        if section_node in track_alloc[track_node]:
                            for clip, midi_data in track_alloc[track_node][section_node].items():
                                for m_data in midi_data:
                                    midi_nl, st, ed = m_data
                                    offset = current_bar.to_seconds(bpm=self.project.bpm)
                                    if clip not in track_arrangement[track_node]:
                                        track_arrangement[track_node][clip] = []
                                    track_arrangement[track_node][clip].append([midi_nl, st+offset, ed+offset])
                    current_bar += section_node.length
                self.track_arrangement.update(track_arrangement)
                self.render_engine.update_midi_node(track_arrangement)

    def sync_project(self):
        for node in self.project.transformation_nodes:
            node.clear()
            node.run()
        self.sync_audio_nodes()
        self.sync_midi_nodes()

    def save_song_midi(self, save_path: str, split_tracks: bool = False):
        def midi_data_to_pretty_midi_inst(t: TrackNode, d: dict):
            pi = pretty_midi.Instrument(2, is_drum=False, name=f"{t.id}{t.name}")
            for midi, midi_data in d.items():
                for m_data in midi_data:
                    midi_nl, st, _ = m_data
                    midi_nl.time_offset(st)
                    pi.notes.extend([pretty_midi.Note(note.velocity, note.pitch, note.start, note.end) for note in midi_nl.notes])
            return pi
        dir_path = os.path.join(save_path, self.project.song_name + "_midis")
        os.makedirs(dir_path, exist_ok=True)
        if split_tracks:
            for track, data in self.track_arrangement.items():
                if track.track_type == TrackType.Midi:
                    pm = pretty_midi.PrettyMIDI(initial_tempo=self.project.bpm)
                    pm.instruments.append(midi_data_to_pretty_midi_inst(track, data))
                    pm.write(os.path.join(dir_path, f"{self.project.song_name}_{track.id}{track.name}.mid"))
        else:
            pm = pretty_midi.PrettyMIDI(initial_tempo=self.project.bpm)
            for track, data in self.track_arrangement.items():
                if track.track_type == TrackType.Midi:
                    pm.instruments.append(midi_data_to_pretty_midi_inst(track, data))
            pm.write(os.path.join(dir_path, self.project.song_name + ".mid"))
