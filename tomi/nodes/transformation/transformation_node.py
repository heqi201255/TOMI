from .. import NodeBase
from tomi import NodeType, BarStepTick, MIDINoteList, RollBlock
import numpy as np


class TransformationNode(NodeBase):
    DEFAULT_ACTION_SEQUENCE = np.ones(16 * 128, dtype=np.int8)
    DEFAULT_ACTION_SEQUENCE[0] = RollBlock.Start

    def __init__(self, node_type: NodeType, project, name: str,
                 action_sequence: list | np.ndarray[np.int8] = DEFAULT_ACTION_SEQUENCE,
                 loop_action_sequence: bool = True):
        super(TransformationNode, self).__init__(node_type, project, name,
                                                   parent_accept=[NodeType.Section],
                                                   child_accept=[NodeType.MidiClip, NodeType.AudioClip])
        if isinstance(action_sequence, list):
            action_sequence = np.array(action_sequence, dtype=np.int8)
        self.action_sequence = action_sequence
        self.loop = loop_action_sequence
        self.loop_clip = True

    def set_loop_action_sequence(self, loop: bool):
        self.loop = loop

    def set_action_sequence(self, action_sequence: list | np.ndarray[np.int8]):
        if isinstance(action_sequence, list):
            action_sequence = np.array(action_sequence, dtype=np.int8)
        self.action_sequence = action_sequence

    def run(self):
        for section in self._parents:
            sec_len = section.length
            if sec_len not in self.outputs:
                self.outputs[sec_len] = {}
            for clip in self._childs:
                self.update_clip_place(section, clip)
        super(TransformationNode, self).run()

    def update_clip_place(self, section, clip):
        if clip.length not in self.outputs[section.length]:
            sec_action_sequence = self.get_section_action_sequences(section, clip)
            self.outputs[section.length][clip.length] = {"action_sequence": sec_action_sequence, "ranges": self.get_ranges(sec_action_sequence)}

    def get_section_action_sequences(self, section, clip=None):
        assert section in self._parents, f"{section.name} is not in {self.name}'s parent node list"
        section_step_len = section.length.to_steps()
        if self.action_sequence.size >= section_step_len:
            return self.action_sequence[:section_step_len]
        else:
            if self.loop:
                return np.hstack([self.action_sequence] * int(np.ceil(section_step_len / self.action_sequence.size)))[
                       :section_step_len]
            else:
                new_action_sequence = np.zeros(int(self.action_sequence.size * np.ceil(section_step_len / self.action_sequence.size)))[
                                      :section_step_len]
                new_action_sequence[:self.action_sequence.size] = self.action_sequence
                return new_action_sequence

    def get_ranges(self, action_sequence: np.ndarray):
        ranges = []
        current_range = []
        current_range_short = []
        stopped = False
        for i, block in enumerate(action_sequence):
            if block == RollBlock.Start:
                if current_range_short:
                    stopped = False
                    current_range_short.append(BarStepTick.step2bst(i).to_seconds(bpm=self.bpm))
                    current_range.append(current_range_short)
                    current_range_short = []
                if current_range:
                    ranges.append(current_range)
                    current_range = []
                current_range_short.append(BarStepTick.step2bst(i).to_seconds(bpm=self.bpm))
            elif block == RollBlock.Duration:
                if stopped:
                    stopped = False
                    current_range_short.append(BarStepTick.step2bst(i).to_seconds(bpm=self.bpm))
            elif block == RollBlock.Empty:
                if current_range_short:
                    stopped = True
                    current_range_short.append(BarStepTick.step2bst(i).to_seconds(bpm=self.bpm))
                    current_range.append(current_range_short)
                    current_range_short = []
            if i == len(action_sequence) - 1:
                if current_range_short:
                    current_range_short.append(BarStepTick.step2bst(i + 1).to_seconds(bpm=self.bpm))
                    current_range.append(current_range_short)
                if current_range:
                    ranges.append(current_range)
        return ranges

    def get_audio_positions(self, section_node, audio_node) -> np.ndarray:
        if section_node.length not in self.outputs or audio_node.length not in self.outputs[section_node.length]:
            self.update_clip_place(section_node, audio_node)
        ranges = self.outputs[section_node.length][audio_node.length]['ranges']
        relative_audio_positions = []
        for turn in ranges:
            start = 0.
            for i in range(len(turn)):
                if i == 0:
                    start = turn[i][0]
                    relative_audio_positions.append(np.array([BarStepTick.sec2beats(start, self.project.bpm), BarStepTick.sec2beats(turn[i][1], self.project.bpm), 0]))
                else:
                    offset = BarStepTick.sec2beats(turn[i][0] - start, self.project.bpm)
                    relative_audio_positions.append(np.array([BarStepTick.sec2beats(turn[i][0], self.project.bpm), BarStepTick.sec2beats(turn[i][1], self.project.bpm), offset]))
        if self.loop_clip:
            try:
                clip_len_beats = max(round(audio_node.length.to_beats()), 1)
            except Exception as e:
                print(audio_node)
                raise e
            looped_positions = []
            for position in relative_audio_positions:
                offsetedbeats = clip_len_beats - position[2]
                poslen = position[1] - position[0]
                remainbeats = poslen
                start = position[0]
                if remainbeats - offsetedbeats <= 0:
                    looped_positions.append(position)
                else:
                    while remainbeats > 0:
                        new_end = start + offsetedbeats
                        if new_end > position[1]:
                            new_end = position[1]
                        looped_positions.append(np.array([start, new_end, position[2]]))
                        start += offsetedbeats
                        remainbeats -= offsetedbeats
            return np.array(looped_positions)
        else:
            return np.array(relative_audio_positions)

    def get_midi_notelist(self, section_node, midi_node) -> MIDINoteList:
        if section_node.length not in self.outputs or midi_node.length not in self.outputs[section_node.length]:
            self.update_clip_place(section_node, midi_node)
        ranges = self.outputs[section_node.length][midi_node.length]['ranges']
        fullnl = []
        for turn in ranges:
            offset = turn[0][0]
            nl: MIDINoteList = midi_node.get_notelist()
            if self.loop_clip:
                while 0 < nl.end_time < turn[-1][1]:
                    nl = nl.duplicate()
            for sec in turn:
                nn = [note.crop([sec[0] - offset, sec[1] - offset]) for note in nl.get_notes()]
                fullnl.extend([note.time_offset(offset) for note in nn if note is not None])
        return MIDINoteList(fullnl, bpm=self.bpm)
