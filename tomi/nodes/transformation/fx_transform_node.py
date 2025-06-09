from .. import TransformationNode
from tomi import NodeType, RollBlock, BarStepTick
import numpy as np


class FxTransformNode(TransformationNode):
    def __init__(self, project, name: str,
                 action_sequence: list | np.ndarray[np.int8] = None,
                 offset: int = None,
                 is_faller: bool = False):
        super(FxTransformNode, self).__init__(NodeType.FxTransform, project, name, action_sequence, False)
        if offset is None:
            offset = 0
        self.is_faller = is_faller
        self.offset = offset
        self.loop_clip = False

    def equals(self, other):
        return isinstance(other, FxTransformNode) and self.offset == other.offset and self.is_faller == other.is_faller and np.all(self.action_sequence == other.action_sequence) and self.loop == other.loop

    @property
    def is_riser(self) -> bool:
        return not self.is_faller

    def get_section_action_sequences(self, section, clip=None):
        if self.action_sequence is not None:
            return super(FxTransformNode, self).get_section_action_sequences(section)
        assert clip is not None, "'clip' is required if action_sequence is not set"
        full_steps = section.length.to_steps()
        action_sequence = np.zeros(full_steps, dtype=np.int8)
        try:
            clip_step_len = clip.length.to_steps()
        except Exception as e:
            print(clip)
            raise e
        if self.is_faller:
            start = self.offset
            end = min(start + clip_step_len, full_steps)
        else:
            end = full_steps + self.offset
            start = max(0, end - clip_step_len)
        action_sequence[start: min(end, len(action_sequence))] = RollBlock.Duration
        action_sequence[start] = RollBlock.Start
        return action_sequence

    def set_offset(self, offset: int):
        self.offset = offset

    def set_front(self, front: bool):
        self.is_faller = front

    def get_audio_positions(self, section_node, audio_node) -> np.ndarray:
        positions = super().get_audio_positions(section_node, audio_node)
        ranges = self.outputs[section_node.length][audio_node.length]['ranges']
        segment: list[float] = ranges[0][0]
        clip_len = audio_node.length
        rm = clip_len.to_seconds(self.bpm) - (segment[1] - segment[0])
        if rm > 0:
            positions[:, 2] += BarStepTick.sec2beats(rm, self.bpm)
        return positions
