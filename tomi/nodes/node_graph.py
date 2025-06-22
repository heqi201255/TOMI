from . import NodeBase, NodeChainProcessor
from tomi import NodeType, LinkType, better_property
from collections import OrderedDict
from functools import reduce
from copy import copy


class CompositionLinks(OrderedDict):
    def __init__(self):
        super().__init__()

    def print_link(self, title: str = "", indent_steps: int = 4):
        def _print_link(_link: OrderedDict, _indent: int = 0):
            for k, v in _link.items():
                print(f"{_indent * " "}{k}")
                if isinstance(v, OrderedDict):
                    _print_link(v, _indent + indent_steps)
        if title:
            print(title)
        _print_link(self)
        print(f"depth: {self.depth}")

    @property
    def depth(self):
        def _get_depth(_link: OrderedDict):
            if not _link:
                return 0
            return max(1 + _get_depth(v) for k, v in _link.items() if isinstance(v, OrderedDict))
        return _get_depth(self)

    def reorder(self, order_pattern: list):
        if not order_pattern:
            return self
        assert all(hp in ['S', 'F', 'C', 'T'] for hp in order_pattern), "Invalid hierarchy pattern, element must be one of 'S':Section or 'F':Transformation or 'C':Clip or 'T':Track."
        links = self.get_links()
        result = CompositionLinks()
        transformation_types = [NodeType.GeneralTransform, NodeType.DrumTransform, NodeType.FxTransform, NodeType.FillTransform]
        clip_types = [NodeType.AudioClip, NodeType.MidiClip]
        node_type_mapping = {'S': NodeType.Section, 'T': NodeType.Track}
        for link in links:
            new_l = [
                node for hp in order_pattern for node in link
                if (
                        (hp == 'S' and node.node_type == NodeType.Section) or
                        (hp == 'F' and node.node_type in transformation_types) or
                        (hp == 'C' and node.node_type in clip_types) or
                        (hp in node_type_mapping and node.node_type == node_type_mapping[hp])
                )
            ]
            new_l.extend([node for node in link if node not in new_l])
            result.add_link(new_l)
        return result

    def add_link(self, keys: tuple | list[NodeBase], verify_hierarchy: bool = False):
        def _add(_link: OrderedDict, keys: tuple | list[NodeBase]):
            if keys:
                if keys[0] not in _link:
                    _link[keys[0]] = CompositionLinks()
                if len(keys) > 1:
                    if verify_hierarchy: keys[0].verify_child(keys[1])
                    _add(_link[keys[0]], keys[1:])
        _add(self, keys)

    def get_links(self):
        def _get_links(_link: OrderedDict = None, prev_keys: list = None):
            if prev_keys is None:
                prev_keys = []
            for k, v in _link.items():
                if v:
                    _get_links(_link[k], prev_keys + [k])
                else:
                    all_links.append(prev_keys + [k])
        all_links = []
        _get_links(self)
        return all_links

    def remove_node(self, node: NodeBase):
        def _remove(_link: OrderedDict):
            if node in _link:
                _link.pop(node)
            for v in _link.values():
                _remove(v)
        _remove(self)

    def sync_nodes(self):
        def sync(_link: OrderedDict, key: NodeBase = None):
            for k in _link.keys():
                if key is not None:
                    k.add_parent(key)
                sync(_link[k], k)
        sync(self)

    def break_connection(self, node1: NodeBase, node2: NodeBase):
        def _break(_link: OrderedDict):
            for k in _link.keys():
                if k == node1:
                    temp = None
                    for k2 in _link[k]:
                        if k2 == node2:
                            temp = k2
                            break
                    if temp: del _link[k][temp]
                elif k == node2:
                    temp = None
                    for k2 in _link[k]:
                        if k2 == node1:
                            temp = k2
                            break
                    if temp: del _link[k][temp]
                _break(_link[k])
        _break(self)
    
    def get_item_links(self, item):
        all_links = self.get_links()
        result = CompositionLinks()
        get_multiple = isinstance(item, tuple)
        item = (item,) if not isinstance(item, tuple) else item
        for link in all_links:
            for it in item:
                if it in link:
                    iid = link.index(it)
                    prev = link[:iid]
                    prev.reverse()
                    link = [it] + prev + link[iid + 1:] if get_multiple else prev + link[iid + 1:]
                    result.add_link(link)
        return result


class NodeGraph:
    arrangement_links: CompositionLinks
    def __init__(self, node_graph: 'NodeGraph' = None):
        self.registered_nodes = set()
        self.links = {
            LinkType.ArrangementLink: CompositionLinks()
        }
        if node_graph:
            self.registered_nodes = copy(node_graph.registered_nodes)
            self.links = copy(node_graph.links)

    def print_link(self, title: str = "", indent_steps: int = 4):
        self.links[LinkType.ArrangementLink].print_link(title, indent_steps)

    @better_property(value_type=CompositionLinks)
    def arrangement_links(self):
        def fget(_self):
            return _self.links[LinkType.ArrangementLink]
        def fset(_self, links: CompositionLinks):
            assert all(NodeChainProcessor.get_link_type(l) == LinkType.ArrangementLink for l in links.get_links())
            _self.links[LinkType.ArrangementLink] = links
        return fget, fset

    def register_node(self, node: NodeBase | list[NodeBase]):
        assert isinstance(node, NodeBase) or all(isinstance(n, NodeBase) for n in node)
        if isinstance(node, NodeBase): node = [node]
        self.registered_nodes.update(node)

    def shrink(self):
        node_direct = {node.node_type: {} for node in self.registered_nodes}
        prune_count = 0
        pruned_nodes = []
        for node in self.registered_nodes:
            for comp_node in node_direct[node.node_type]:
                if comp_node.equals(node):
                    node_direct[node.node_type][node] = comp_node
                    pruned_nodes.append(node)
                    print(f"{node} is the same as {comp_node}")
                    prune_count += 1
                    break
            if node not in node_direct[node.node_type]:
                node_direct[node.node_type][node] = node
        links = self.arrangement_links.get_links()
        new_links = [[node_direct[node.node_type][node] for node in link] for link in links]
        result = CompositionLinks()
        for link in new_links:
            result.add_link(link)
        self.arrangement_links = result
        self.sync_nodes()
        for node in pruned_nodes:
            self.registered_nodes.remove(node)
        print(f"Pruned {prune_count} nodes: {pruned_nodes}")

    def __setitem__(self, key, value):
        if isinstance(value, (list, tuple)):
            for v in value:
                self.__setitem__(key, v)
        elif value is None:
            self.add(*key if isinstance(key, tuple) else [key])
        else:
            self.add((*key, value) if isinstance(key, tuple) else (key, value))

    def add(self, chain: tuple | list[NodeBase] | NodeChainProcessor):
        if not isinstance(chain, NodeChainProcessor):
            assert all(isinstance(n, NodeBase) for n in chain), 'All elements must be Node.'
            assert len(chain) > 1, 'Chain must contain more than one node.'
            chain = reduce(lambda x, y: x >> y, chain)
        chain = chain.get_links()
        for link_type, links in chain.items():
            for link in links:
                self.links[link_type].add_link(link, verify_hierarchy=True)
                self.register_node(link)

    def remove_node(self, node: NodeBase, inplace: bool = False):
        assert node in self.registered_nodes, f"Cannot remove node {node} because it is not registered."
        ng = self if inplace else NodeGraph(self)
        ng.registered_nodes.remove(node)
        ng.arrangement_links.remove_node(node)
        return ng

    def break_connection(self, node1: NodeBase, node2: NodeBase):
        assert node1 in self.registered_nodes, f"Cannot find node {node1} because it is not registered."
        assert node2 in self.registered_nodes, f"Cannot find node {node2} because it is not registered."
        self.arrangement_links.break_connection(node1, node2)

    def sync_nodes(self):
        for nodes in self.registered_nodes:
            nodes.remove_all_arrangement_relations()
        self.arrangement_links.sync_nodes()

    def __getitem__(self, item):
        if isinstance(item, tuple):
            if isinstance(item[-1], LinkType):
                lt = item[-1]
                item = item[:-1]
            else:
                lt = None
            for node in item:
                assert node in self.registered_nodes, f"Node {node} not registered"
        else:
            lt = None
            assert item in self.registered_nodes, f"Node {item} not registered"
        return {lt: ls.get_item_links(item) for lt, ls in self.links.items()} if lt is None else self.links[lt].get_item_links(item)

    def __repr__(self):
        return f"NodeGraph: {len(self.registered_nodes)} registered_nodes"

