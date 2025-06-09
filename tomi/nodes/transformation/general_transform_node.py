from .. import TransformationNode
from tomi import NodeType
import numpy as np


class GeneralTransformNode(TransformationNode):
    def __init__(self, project, name: str,
                 action_sequence: list | np.ndarray[np.int8] = TransformationNode.DEFAULT_ACTION_SEQUENCE,
                 loop_action_sequence: bool = True):
        super(GeneralTransformNode, self).__init__(NodeType.GeneralTransform, project, name, action_sequence, loop_action_sequence)
        if action_sequence is None:
            self.action_sequence = self.DEFAULT_ACTION_SEQUENCE

    def equals(self, other):
        return isinstance(other, GeneralTransformNode) and len(self.action_sequence) == len(other.action_sequence) and np.all(self.action_sequence == other.action_sequence) and self.loop == other.loop and self.loop_clip == other.loop_clip


