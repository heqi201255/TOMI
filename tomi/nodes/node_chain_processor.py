from tomi import NodeType, LinkType


class NodeChainProcessor:
    class NodeChainLayer:
        """
        This class is used to represent and process a layer of a chain.
        """
        def __init__(self, layer: list):
            """
            Process the layer to get layer_types and layer_dim.
            :param layer: An unprocessed 1D layer list, including multiple nodes separated by '&' or '+'.
            """
            self.layer = self.parse_layer(layer)
            self.layer_types = set(i.node_type for sub in self.layer for item in sub for i in (item if isinstance(item, list) else [item]))
            # All unique node types found in current layer.
            self.layer_dim = [len(l) for l in self.layer]
            # A list containing lengths of each 'layer block'.

        def __len__(self) -> int:
            """
            :return: The number of layer blocks of current layer.
            """
            return len(self.layer)

        def __getitem__(self, item: int) -> list:
            """
            Access a layer block in current layer.
            :param item: index
            :return: layer block
            """
            return self.layer[item]

        @staticmethod
        def parse_layer(layer: list) -> list:
            """
            Separate the unprocessed 1D layer list first by '|' to get layer blocks. Then separate each layer block by '&' to get each individual layer node, if layer node contains '+', then
            separate it as a sub-list for further processing as multiple links.
            :param layer: unprocessed 1D layer list.
            :return: a 2D list, for some layer blocks maybe 3D if it includes '+'.
            """
            segments = NodeChainProcessor._split_list(layer, '|')
            return [[x[0] if len(x)==1 else [a[0] if len(a)==1 else a for a in NodeChainProcessor._split_list(x, '+')] for x in NodeChainProcessor._split_list(segment, '&')] for segment in segments]

        def check_type(self, node_type: NodeType | tuple | list[NodeType]) -> bool:
            """
            Iterate through a layer block to check if nodes meet the node_type.
            :param node_type: check if all nodes in the layer meet this node type.
            :return: True if all nodes meet the node_type. False otherwise.
            """
            return all(x == node_type for x in self.layer_types) if isinstance(node_type, NodeType) else all(x in node_type for x in self.layer_types)

        def get_by_index(self, index: tuple | list) -> 'NodeBase':
            """
            Find the corresponding layer node according to the index of current layer.
            :param index: a tuple or list of 2 integers.
            :return: the corresponding Node.
            """
            assert len(index) == 2
            try:
                # First try to fetch the index directly.
                return self.layer[index[0]][index[1]]
            except:
                # If fails, that means the layer may contain just a single element or the layer block contain a single element.
                # Similar to numpy's broadcasting feature.
                if len(self.layer) == 1 and len(self.layer[0]) == 1:
                    return self.layer[0][0]
                row = self.layer[index[0]]
                if len(row) == 1:
                    return row[0]
                else:
                    return row[index[1]]

        def __repr__(self):
            s = f"ChainLayer{self.layer_dim}:"
            for l in self.layer:
                s += f"\n\t{l}"
            return s

    def __init__(self, node: 'NodeBase'):
        """
        Just put the node inside a list as the chain.
        :param node: any Node instance.
        """
        self.chains = [node]

    def __floordiv__(self, other):
        """
        Used to separate different chains. Each chain will be processed separately.
        Add the operator sign to the chain list.
        :param other: a Node or a NodeChainProcessor.
        :return: self.
        """
        ot = NodeChainProcessor(other) if not isinstance(other, NodeChainProcessor) else other
        self.chains.append('//')
        self.chains += ot.chains
        return self

    def __rshift__(self, other):
        """
        Used for transition to next element in the chain.
        :param other:
        :return:
        """
        ot = NodeChainProcessor(other) if not isinstance(other, NodeChainProcessor) else other
        self.chains.append('>>')
        self.chains += ot.chains
        return self

    def __or__(self, other):
        """
        Used for parallel nodes.
        :param other:
        :return:
        """
        ot = NodeChainProcessor(other) if not isinstance(other, NodeChainProcessor) else other
        self.chains.append('|')
        self.chains += ot.chains
        return self

    def __and__(self, other):
        ot = NodeChainProcessor(other) if not isinstance(other, NodeChainProcessor) else other
        self.chains.append('&')
        self.chains += ot.chains
        return self

    def __add__(self, other):
        ot = NodeChainProcessor(other) if not isinstance(other, NodeChainProcessor) else other
        self.chains.append('+')
        self.chains += ot.chains
        return self

    @staticmethod
    def get_link_type(link: list['NodeBase']):
        if link[0].node_type == NodeType.Section:
            if len(link) == 4:
                if (link[1].node_type in (NodeType.GeneralTransform, NodeType.DrumTransform, NodeType.FxTransform, NodeType.FillTransform)
                        and link[2].node_type in (NodeType.MidiClip, NodeType.AudioClip) and link[3].node_type == NodeType.Track):
                    return LinkType.ArrangementLink
                else:
                    raise SyntaxError('Arrangement Link must follow "Section >> Transformation >> Clip >> Track"')
            else:
                raise SyntaxError(f'Cannot parse link: {link}')
        else:
            raise SyntaxError(f'Cannot parse link: {link}')

    @staticmethod
    def get_chain_type(chain_layers):
        if chain_layers[0].check_type(NodeType.Section):
            if len(chain_layers) == 4:
                if (chain_layers[1].check_type((NodeType.GeneralTransform, NodeType.DrumTransform, NodeType.FxTransform, NodeType.FillTransform))
                        and chain_layers[2].check_type((NodeType.MidiClip, NodeType.AudioClip)) and chain_layers[3].check_type(NodeType.Track)):
                    return LinkType.ArrangementLink
                else:
                    raise SyntaxError('Arrangement Link must follow "Section >> Transformation >> Clip(exclude Automation) >> Track"')
            else:
                raise SyntaxError(f'Cannot parse link: {chain_layers}')
        else:
            raise SyntaxError(f'Cannot parse link: {chain_layers}')

    def _check_chain_is_valid(self, chain_layers: list):
        def compare(a, b):
            if len(a) == 1 or len(b) == 1:
                return True
            if len(a) == len(b):
                f = True
                for ia, ib in zip(a, b):
                    if ia != ib and ia != 1 and ib != 1:
                        f = False
                return f
            return False
        max_dim = None
        last_dim = None
        for ele in chain_layers:
            dim = ele.layer_dim
            if last_dim is None:
                last_dim = dim
                max_dim = dim
                continue
            if not compare(dim, last_dim):
                raise ValueError(f"Dimension does not match: {dim} != {last_dim}")
            if len(dim) > len(max_dim):
                max_dim = dim
            else:
                max_dim = [max(max_dim[i], dim[i]) for i in range(len(dim))]
            last_dim = dim
        chain_indexes = [(i, j) for i in range(len(max_dim)) for j in range(max_dim[i])]
        return chain_indexes

    def _get_chain_links(self, chain_layers):
        chain_indexes = self._check_chain_is_valid(chain_layers)
        all_chains = []
        for index in chain_indexes:
            cs = [[]]
            for ele in chain_layers:
                e = ele.get_by_index(index)
                cs = [n[:] + [en] for en in e for n in cs] if isinstance(e, list) else [n + [e] for n in cs]
            all_chains.extend(cs)
        return all_chains

    def _parse_chains(self, chain: list):
        chain_layers_unprocessed = self._split_list(chain, ('>>',))
        chain_layers = [NodeChainProcessor.NodeChainLayer(layer) for layer in chain_layers_unprocessed]
        return self.get_chain_type(chain_layers), self._get_chain_links(chain_layers)

    def get_links(self, separate_link_types: bool = True):
        chain_groups = self._split_list(self.chains, "//")
        if separate_link_types:
            all_links = {LinkType.ArrangementLink: []}
        else:
            all_links = []
        for chain in chain_groups:
            link_type, links = self._parse_chains(chain)
            if separate_link_types:
                all_links[link_type].extend(links)
            else:
                all_links.extend(links)
        return all_links

    def __repr__(self):
        s = "NodeChain:"
        chains = self.get_links()
        for link_type, links in chains.items():
            s += f"\n\t{link_type.name}:"
            for link in links:
                s += f'\n\t\t{"-->".join([str(n) for n in link])}'
        return s

    @staticmethod
    def _split_list(l: list, delimiter):
        delimiters = set(delimiter) if isinstance(delimiter, (list, tuple, set)) else {delimiter}
        split_indexes = [-1] + [i for i, item in enumerate(l) if item in delimiters] + [len(l)]
        return [l[split_indexes[i]+1:split_indexes[i+1]] for i in range(len(split_indexes)-1)]