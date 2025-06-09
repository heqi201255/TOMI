from .. import NodeBase
from tomi import NodeType, MIDIType, AudioType


class ClipNode(NodeBase):
    def __init__(self, node_type: NodeType, project, name: str, dependent_node: 'ClipNode' = None, dependent_type: MIDIType | AudioType = None):
        super(ClipNode, self).__init__(node_type,
                                       project,
                                       name,
                                       parent_accept=[NodeType.GeneralTransform, NodeType.DrumTransform, NodeType.FxTransform, NodeType.FillTransform],
                                       child_accept=[NodeType.Track]
                                       )
        self.length = None
        self.db_cur = None
        self.dependent_node: ClipNode = dependent_node
        self.dependent_type: MIDIType | AudioType = dependent_type

    def run(self):
        if self.node_type == NodeType.MidiClip:
            self.db_cur = self.project.databases['midiCur']
        elif self.node_type == NodeType.AudioClip:
            self.db_cur = self.project.databases['audioCur']
        super(ClipNode, self).run()

    def clear(self):
        super(ClipNode, self).clear()
        self.db_cur = None

    def search_db(self, query: str, parameters: list = None):
        if parameters is not None:
            self.db_cur.execute(query, parameters)
        else:
            self.db_cur.execute(query)
        return self.db_cur.fetchall()
