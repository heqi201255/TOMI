from .. import TransformationNode
from tomi import NodeType, RollBlock, BarStepTick
import numpy as np


class DrumTransformNode(TransformationNode):
    def __init__(self, project, name: str,
                 action_sequence: list | np.ndarray[np.int8] = TransformationNode.DEFAULT_ACTION_SEQUENCE,
                 loop_action_sequence: bool = True):
        super(DrumTransformNode, self).__init__(NodeType.DrumTransform, project, name, action_sequence, loop_action_sequence)
        if action_sequence is None:
            self.action_sequence = self.DEFAULT_ACTION_SEQUENCE
        self.loop_clip = False

    def get_ranges(self, action_sequence: np.ndarray):
        ranges = []
        current_range = []
        current_range_short = []
        for i, block in enumerate(action_sequence):
            if block == RollBlock.Start:
                if current_range_short:
                    current_range_short.append(BarStepTick.step2bst(i).to_seconds(bpm=self.bpm))
                    current_range.append(current_range_short)
                    current_range_short = []
                if current_range:
                    ranges.append(current_range)
                    current_range = []
                current_range_short.append(BarStepTick.step2bst(i).to_seconds(bpm=self.bpm))
            if i == len(action_sequence) - 1:
                if current_range_short:
                    current_range_short.append(BarStepTick.step2bst(i + 1).to_seconds(bpm=self.bpm))
                    current_range.append(current_range_short)
                if current_range:
                    ranges.append(current_range)
        return ranges

    def equals(self, other):
        return isinstance(other, DrumTransformNode) and np.all(self.action_sequence == other.action_sequence) and self.loop == other.loop


