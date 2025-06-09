from .. import TransformationNode
from tomi import NodeType, RollBlock
import numpy as np


class FillTransformNode(TransformationNode):
    def __init__(self, project, name: str,
                 action_sequence: list | np.ndarray[np.int8] = None,
                 loop_bars: float = None, offset: int = None):
        super(FillTransformNode, self).__init__(NodeType.FillTransform, project, name, action_sequence, True)
        self.loop_bars = loop_bars
        self.offset = offset
        if self.offset is None:
            self.offset = 0
        if self.action_sequence is not None:
            self.loop_bars, self.offset = self._get_info_from_set_action_sequence()
        if self.loop_bars is None:
            self.loop = False
        self.loop_clip = False

    def equals(self, other):
        return isinstance(other, FillTransformNode) and np.all(self.action_sequence == other.action_sequence) and self.loop_bars == other.loop_bars and self.offset == other.offset and self.loop == other.loop

    def get_section_action_sequences(self, section, clip=None):
        if self.action_sequence is not None:
            return super(FillTransformNode, self).get_section_action_sequences(section)
        assert clip is not None, "'clip' is required if placement is not set"
        assert self.loop_bars is not None, "'loop_bars' is required if 'clip' is set"
        assert section in self._parents, f"{section.name} is not in {self.name}'s parent node list"
        full_steps = section.length.to_steps()
        action_sequence = np.zeros(full_steps, dtype=np.int8)
        loop_steps = int(self.loop_bars * 16)
        temp_pos = loop_steps
        try:
            clip_step_len = clip.length.to_steps()
        except Exception as e:
            print(clip)
            raise e
        while temp_pos <= full_steps:
            end = temp_pos + self.offset
            start = max(0, end - clip_step_len)
            action_sequence[start: min(end, len(action_sequence))] = RollBlock.Duration
            action_sequence[start] = RollBlock.Start
            temp_pos += loop_steps
        return action_sequence

    def _get_info_from_set_action_sequence(self):
        start_loc = None
        end_loc = None
        loop_steps = None
        for block_id, block in enumerate(self.action_sequence):
            if block == RollBlock.Start:
                if start_loc is None:
                    start_loc = block_id
                else:
                    if end_loc is None:
                        end_loc = block_id
                    loop_steps = block_id - start_loc
                    break
            elif block == RollBlock.Empty:
                if end_loc is None and start_loc is not None:
                    end_loc = block_id
            elif block_id == self.action_sequence.size - 1:
                if end_loc is None and start_loc is not None:
                    end_loc = block_id + 1
        if loop_steps is not None:
            loop_bars = loop_steps // 16
        else:
            if end_loc is not None:
                loop_bars = end_loc // 16
            else:
                loop_bars = None
        if loop_bars:
            offset = 9999
            while end_loc >= 0:
                end_loc -= loop_bars * 16
                if abs(end_loc) < abs(offset):
                    offset = end_loc
                else:
                    break
        else:
            offset = 0
        return loop_bars, offset

    def set_action_sequence(self, placement: list | np.ndarray[np.int8]):
        super().set_action_sequence(placement)
        self.loop_bars, self.offset = self._get_info_from_set_action_sequence()


