from . import NodeChainProcessor
from tomi import ExcessiveParentError, GUIContentConfigs, NodeType, BarStepTick, TimeSignature, printer


class NodeBase:
    _color_map = {
        NodeType.Section: printer.blue,
        NodeType.GeneralTransform: printer.yellow,
        NodeType.DrumTransform: printer.yellow,
        NodeType.FillTransform: printer.yellow,
        NodeType.FxTransform: printer.yellow,
        NodeType.AudioClip: printer.green,
        NodeType.MidiClip: printer.green,
        NodeType.Track: printer.red
    }
    def __init__(self,
                 node_type: NodeType,
                 project: 'Project',
                 name: str,
                 parent_accept: list[NodeType] = None,
                 child_accept: list[NodeType] = None,
                 plug_accept: list[NodeType] = None,
                 host_accept: list[NodeType] = None,
                 single_parent_restriction: bool = False):
        if parent_accept is None:
            parent_accept = []
        if child_accept is None:
            child_accept = []
        if plug_accept is None:
            plug_accept = []
        if host_accept is None:
            host_accept = []
        self.node_type = node_type
        self.project = project
        self.id = project.get_new_node_id()
        self.console_color = self._color_map[self.node_type]
        self.project.nodes.add(self)
        self.name = name
        self.parent_accept = parent_accept
        self.child_accept = child_accept
        self.plug_accept = plug_accept
        self.host_accept = host_accept
        self.single_parent_restriction = single_parent_restriction
        self._parents = set()
        self._childs = set()
        self.outputs = {}
        self.gui_configs = GUIContentConfigs(self)
        self._initialized = False
        self.need_sync = False

    def __getstate__(self):
        state = self.__dict__.copy()
        rl = ['project', 'console_color', 'parent_accept', 'child_accept', 'plug_accept', 'host_accept',
              'single_parent_restriction', '_parents', '_childs', 'gui_configs', 'outputs']
        for attr in rl: state.pop(attr, None)
        return state

    def is_clip_node(self) -> bool:
        return self.node_type in (NodeType.AudioClip, NodeType.MidiClip)

    def is_transformation_node(self) -> bool:
        return self.node_type in (NodeType.GeneralTransform, NodeType.DrumTransform, NodeType.FxTransform, NodeType.FillTransform)

    @property
    def time_signature(self) -> TimeSignature:
        return self.project.time_signature

    @property
    def bpm(self):
        return self.project.bpm

    @property
    def key(self):
        return self.project.key

    @property
    def mode(self):
        return self.project.mode

    @property
    def key_mode(self):
        return self.project.key_mode

    def verify_parent(self, node):
        assert isinstance(node, NodeBase), "'node' must be an instance of NodeBase"
        assert node.node_type in self.parent_accept, f"{self.name}: Wrong Node type '{node.node_type}', '{self.node_type}' only supports {self.parent_accept} as parent nodes."
        assert self.node_type in node.child_accept, f"{self.name}: Wrong Node type '{node.node_type}', '{node.node_type}' only supports {node.child_accept} as child nodes."

    def verify_child(self, node):
        assert isinstance(node, NodeBase), "'node' must be an instance of NodeBase"
        assert node.node_type in self.child_accept, f"{self.name}: Wrong Node type '{node.node_type}', '{self.node_type}' only supports {self.child_accept} as child nodes."
        assert self.node_type in node.parent_accept, f"{self.name}: Wrong Node type '{node.node_type}', '{node.node_type}' only supports {node.parent_accept} as parent nodes."

    def add_parent(self, node: 'NodeBase'):
        self.verify_parent(node)
        if node not in self._parents:
            if not (self.single_parent_restriction and self._parents):
                self._parents.add(node)
                node.add_child(self)
            else:
                raise ExcessiveParentError(f"{self.name}: Already has a parent, cannot add more than one parent.")

    def add_child(self, node: 'NodeBase'):
        self.verify_child(node)
        self._childs.add(node)
        node.add_parent(self)

    @property
    def usage(self) -> int:
        usage_links = self.project.node_graph[self]
        return sum(len(links.get_links()) for links in usage_links.values())

    @property
    def parents(self):
        return self._parents

    @property
    def childs(self):
        return self._childs

    def _remove_all_relations(self, high: str, low: str):
        for node in getattr(self, high):
            if self in getattr(node, low):
                getattr(node, low).remove(self)
        for node in self.__getattribute__(low):
            if self in node.__getattribute__(high):
                node.__getattribute__(high).remove(self)
        setattr(self, high, set())
        setattr(self, low, set())

    def remove_all_arrangement_relations(self):
        self._remove_all_relations('_parents', '_childs')

    def remove_parent(self, node: 'NodeBase'):
        assert node in self._parents, f"Cannot remove node {node} because it is not in the parent list."
        self._parents.remove(node)
        if self in node._childs:
            node.remove_child(self)
        self.project.node_graph.break_connection(self, node)

    def remove_child(self, node: 'NodeBase'):
        assert node in self._childs, f"Cannot remove node {node} because it is not in the child list."
        self._childs.remove(node)
        if self in node._parents:
            node.remove_parent(self)
        self.project.node_graph.break_connection(self, node)

    def run(self):
        self._initialized = True

    def clear(self):
        self.outputs = {}
        self._initialized = False

    def log(self, msg):
        printer.print(f"{self.__repr__()} - {msg}")

    @staticmethod
    def _get_param_name_and_loc(gui_param_name: str):
        if gui_param_name.__contains__('$'):
            name, loc = gui_param_name.split('$')
            return name, int(loc)
        return gui_param_name, None

    @staticmethod
    def _get_updated_bst(old_bst: BarStepTick, gui_param_loc: int, gui_param_value: int):
        assert 0<= gui_param_loc <= 2
        return BarStepTick(gui_param_value if gui_param_loc == 0 else old_bst.bar,
                          gui_param_value if gui_param_loc == 1 else old_bst.step,
                          gui_param_value if gui_param_loc == 2 else old_bst.tick)

    def update_gui_configs(self): ...

    def config_update(self, param_name: str, param_value): ...

    def __str__(self):
        return f"{self.console_color(f'<{self.node_type.name}>')}{self.name}"

    @property
    def gui_name(self):
        return f"<{self.node_type.name}>{self.name}"

    def __repr__(self):
        return f"{self.console_color(f'<{self.node_type.name}>')}{self.name}"

    def __or__(self, other):
        return NodeChainProcessor(self).__or__(other) if isinstance(other, NodeChainProcessor) else NodeChainProcessor(self).__or__(NodeChainProcessor(other))

    def __add__(self, other):
        return NodeChainProcessor(self).__add__(other) if isinstance(other, NodeChainProcessor) else NodeChainProcessor(self).__add__(NodeChainProcessor(other))

    def __and__(self, other):
        return NodeChainProcessor(self).__and__(other) if isinstance(other, NodeChainProcessor) else NodeChainProcessor(self).__and__(NodeChainProcessor(other))

    def __rshift__(self, other):
        return NodeChainProcessor(self).__rshift__(other) if isinstance(other, NodeChainProcessor) else NodeChainProcessor(self).__rshift__(NodeChainProcessor(other))

    def __floordiv__(self, other):
        return NodeChainProcessor(self).__floordiv__(other) if isinstance(other, NodeChainProcessor) else NodeChainProcessor(self).__floordiv__(NodeChainProcessor(other))



