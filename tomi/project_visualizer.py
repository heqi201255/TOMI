import random
import numpy as np
from tomi import (get_random_rgb, OrderedDotdict, ARRANGEMENT_FRONT_PADDING_BARS, TrackType,
                   BarStepTick, TrackNode, NodeType, RollBlock, NodeGraph, NodeLinks, ConsoleRollViz,
                   RollSectionInfo, RollBlockGroupInfo, RollBlockInfo, RollPrintingMode, LinkType, printer)

random.seed(2024)
np.random.seed(2024)


class ProjectVisualizer:
    def __init__(self, engine, show_sap_shadow: bool = True, show_color_blocks: bool = True, show_bars: bool = True):
        self.engine = engine
        self.time_signature = self.engine.time_signature
        self.project = self.engine.project
        self.show_sap_shadow = show_sap_shadow
        self.show_color_blocks = show_color_blocks
        self.show_bars = show_bars
        # Get color maps
        self.sap_color_map = self._get_nodes_color_pattern(self.project.section_nodes, (0, 125))
        if self.show_color_blocks:
            self.clip_color_map = self._get_nodes_color_pattern(self.project.clip_nodes, (125, 255))
        self.channel_names = []
        self.sap_sections = self.get_section_ranges()
        self.roll = []
        self.block_groups = self.get_track_action_sequences()
        vbg_colors = {tuple(self.sap_sections[sec]): self.sap_color_map[sec.name] for sec in self.sap_sections}
        self.tr = ConsoleRollViz(roll=self.block_groups,
                                 channel_names=self.channel_names,
                                 row_length=int(self.project.song_beat_length * 4 + 16),
                                 roll_info=self.get_song_info(),
                                 arrangement_info=[RollSectionInfo(length=section.length, text=section.name, color=self.sap_color_map[section.name]) for section in self.project.arrangement],
                                 start_loc=BarStepTick(ARRANGEMENT_FRONT_PADDING_BARS, time_signature=self.time_signature),
                                 vbg_colors=vbg_colors,
                                 mode=RollPrintingMode.Compact,
                                 show_bars=self.show_bars,
                                 time_signature=self.time_signature)

    def print_playlist(self):
        self.tr.draw()

    def _get_track_name(self, track_node, clip_node=None):
        if clip_node:
            return f"{printer.remove_ansi_codes(track_node.__str__())} ({printer.remove_ansi_codes(clip_node.__str__())})"
        else:
            return printer.remove_ansi_codes(track_node.__str__())

    def _get_nodes_color_pattern(self, nodes: list, color_range: tuple, taken_colors: list = None) -> dict:
        colors = get_random_rgb(r_range=color_range, g_range=color_range, b_range=color_range,
                                taken_colors=taken_colors, num_output=len(nodes))
        return {node.name: colors[color_ind] for color_ind, node in enumerate(nodes)}

    def get_song_info(self):
        # Process song info line
        song_seconds = BarStepTick.beat2sec(self.project.song_beat_length - 2, self.project.bpm, time_signature=self.time_signature)
        song_bst = BarStepTick.sec2bst(song_seconds, self.project.bpm, time_signature=self.time_signature)
        info = (f"Project: {self.project.song_name}  Genre: {self.project.genre.name}  BPM: {self.project.bpm}  TimeSignature: {self.time_signature.numerator}/{self.time_signature.denominator}"
                f"Key: {self.project.key_mode.__str__()}       SongLength: {song_seconds} seconds ({song_bst.bar} "
                f"Bars {song_bst.step} Steps)")
        return info

    def get_section_ranges(self):
        # Process arrangement line
        sap_sections = OrderedDotdict()
        current_bar = BarStepTick(ARRANGEMENT_FRONT_PADDING_BARS, time_signature=self.time_signature)
        for sec in self.project.arrangement:
            start = current_bar.to_steps()
            current_bar += sec.length
            end = current_bar.to_steps()
            if sec not in sap_sections:
                sap_sections[sec] = []
            sap_sections[sec].append((start, end))
        return sap_sections

    def get_converter(self, transformation):
        return {
            NodeType.FxTransform: {
                RollBlock.Start: '█',
                RollBlock.Duration: '◀' if transformation.node_type == NodeType.FxTransform and transformation.is_riser else "▶",
                RollBlock.Empty: ' ',
            },
            NodeType.DrumTransform: {
                RollBlock.Start: '█',
                RollBlock.Duration: '▆',
                RollBlock.Empty: ' ',
            },
            NodeType.GeneralTransform: {
                RollBlock.Start: '█',
                RollBlock.Duration: '▆',
                RollBlock.Empty: ' ',
            },
            NodeType.FillTransform: {
                RollBlock.Start: '█',
                RollBlock.Duration: '▟',
                RollBlock.Empty: ' ',
            },
        }[transformation.node_type]

    def get_track_action_sequences(self):
        def _get_track_type_str(tn: TrackNode):
            return "I" if tn.track_type == TrackType.Midi else "A"
        tracks = {}
        graph: NodeGraph = self.project.node_graph
        sampler_clips = []
        for tid, track_node in enumerate(self.project.track_nodes):
            track_node: TrackNode
            links: NodeLinks = graph[track_node, LinkType.ArrangementLink].reorder(['S', 'F', 'C'])
            blocks = {}
            for sec, v in links.items():
                if sec.node_type != NodeType.Section:
                    sampler_clips.append(sec)
                    continue
                for transformation, v2 in v.items():
                    for clip in v2:
                        if clip not in blocks:
                            blocks[clip] = {}
                        for start, end in self.sap_sections[sec]:
                            blocks[clip][(start, end)] = (transformation.outputs[sec.length][clip.length]['action_sequence'], self.get_converter(transformation))
            match track_node.track_type:
                case TrackType.Audio:
                    first_clip = True
                    for clip in blocks:
                        if first_clip:
                            first_clip = False
                            front_line_part = f"{tid+1}. {self._get_track_name(track_node, clip)} ({_get_track_type_str(track_node)})"
                        else:
                            pad = f"   └{"-" * (len(f"{self._get_track_name(track_node)}") - 1)}"
                            front_line_part = f"{pad} ({printer.remove_ansi_codes(clip.__str__())}) ({_get_track_type_str(track_node)})"
                        tracks[front_line_part] = {clip: blocks[clip]}
                        self.channel_names.append(front_line_part)
                case TrackType.Midi:
                    if blocks:
                        front_line_part = f"{tid+1}. {self._get_track_name(track_node)} ({_get_track_type_str(track_node)})"
                        tracks[front_line_part] = blocks
                        self.channel_names.append(front_line_part)
        block_groups = []
        for clip in self.project.clip_nodes:
            if clip in sampler_clips:
                continue
            locs = []
            for tid, t in enumerate(self.channel_names):
                if clip in tracks[t]:
                    for inds, content in tracks[t][clip].items():
                        locs.append(RollBlockInfo(row_index=tid, start_index=inds[0], arr=content[0], converter=content[1]))
            block_groups.append(RollBlockGroupInfo(info=clip.name,
                                                   blocks=locs,
                                                   color=self.clip_color_map[clip.name]))
        return block_groups
