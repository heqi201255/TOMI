from .. import NodeBase
from tomi import NodeType, BarStepTick, SectionType, printer, config
import random
import math


class SectionNode(NodeBase):
    def __init__(self,
                 project,
                 name: str,
                 section_type: SectionType,
                 bst_length: BarStepTick = None):
        super(SectionNode, self).__init__(NodeType.Section,
                                          project,
                                          name,
                                          parent_accept=[],
                                          child_accept=[NodeType.GeneralTransform, NodeType.DrumTransform, NodeType.FxTransform, NodeType.FillTransform])
        if bst_length is None:
            bst_length = BarStepTick(8, time_signature=project.time_signature)
        self.section_type = section_type
        self.length = bst_length

    def set_length(self, bst_length: BarStepTick):
        self.length = bst_length

    def equals(self, other):
        return isinstance(other, SectionNode) and self.length == other.length and self.section_type == other.section_type

    def update_gui_configs(self):
        self.gui_configs.update_config(param_name='id', gui_name='Node ID', param_type=int, value=self.id, mutable=False)
        self.gui_configs.update_config(param_name='name', gui_name='Node Name', param_type=str, value=self.gui_name, mutable=True)
        self.gui_configs.update_config(param_name='_childs', gui_name='Child Nodes', param_type=set, value={n.gui_name for n in self._childs}, mutable=False, special_entry='string_container')
        self.gui_configs.update_config(param_name='section_type', gui_name='Section Type', param_type=SectionType, value=self.section_type, mutable=True)
        self.gui_configs.update_config(param_name='length', gui_name='Section Length', param_type=BarStepTick, value=self.length, mutable=True)

    def config_update(self, param_name: str, param_value):
        name, loc = self._get_param_name_and_loc(param_name)
        match name:
            case 'name': self.name = param_value
            case 'section_type': self.section_type = param_value
            case 'length': self.length = self._get_updated_bst(self.length, loc, param_value)

    @staticmethod
    def generate_arrangement(project, song_length: int = 96, richness: int = 2, regular: bool = True,
                             normal: bool = True, template: list = None, non_stop: bool = False):
        '''
        @retval: list of (section name, bar length)
        '''
        if config.FIX_SEED is not None: random.seed(config.FIX_SEED)

        def _double_length(template: list) -> list:
            '''
            For each section in the template, double its length.
            '''
            return [(i[0], 2 * i[1]) for i in template]

        def _scale_length(template: list, scaler) -> list:
            '''
            For each section in the template, scale its length based on a random selected number from scaler. Only
            the verse and chorus sections will be extended.
            '''
            return [(i[0], i[1] * random.choice(scaler[i[0]])) for i in template]

        def _get_num_of_bars(template: list) -> int:
            '''
            Calculate the total number of bars of the template.
            '''
            return sum([x[1] for x in template])

        def _is_tuple(template: list) -> bool:
            '''
            Check whether the element in the list is tuple.
            '''
            for i in template:
                if not isinstance(i, tuple):
                    return False
            return True

        def _get_abnormal(template: list, length: int) -> list:
            '''
            Fill the sections of the template in an abnormal (creative) way by randomly select a section and extend it 4 more
            bars, until the total number of bars reached the specified length.
            '''
            bars = _get_num_of_bars(template)
            remaining_bars = math.ceil((length - bars) / 4) * 4
            while remaining_bars >= 0:
                index = random.randint(0, len(template) - 1)
                template[index] = (template[index][0], template[index][1] + 4)
                remaining_bars -= 4
            return template

        def _get_normal(template: list, length: int) -> list:
            '''
            Fill the sections of the template in a normal way, by firstly double the length of all sections, then scale the
            section lengths twice, then double again, until the total number of bars reached the specified length.
            '''
            scaler = {'intro': [1], 'verse': [2, 4],
                      'pre_chorus': [1], 'chorus': [2, 4],
                      'bridge': [1], 'outro': [1]}
            flag = 0
            while True:
                backup = template
                if flag % 3 == 1:
                    template = _scale_length(template, scaler)
                else:
                    template = _double_length(template)
                    if _get_num_of_bars(template) > length:
                        template = _scale_length(backup, scaler)
                        if template == backup:
                            template = _double_length(template)
                flag += 1
                bars = _get_num_of_bars(template)
                if bars > length:
                    break
            return template

        # get the rough bar length of the song, which is a multiple of 4.
        length = math.ceil(song_length / 4) * 4
        while True:
            if not template is None:
                # if the 'SAP_template' parameter is not None, it must be a list where each element should be a section
                # name or a tuple such as ('intro',8), if the element is str names, we just need to assign a bar length
                # to each of those sections; if the element is a tuple, the second value of the tuple is the bar length,
                # so we just need to return the template in such case.
                if not _is_tuple(template):
                    if len(template) == 1:
                        # This if statement is applied for special case that the template only contains a single section
                        template = [(template[0], length)]
                    else:
                        template = [(i, 4) for i in template]
                        if normal:
                            template = _get_normal(template, length)
                        else:
                            template = _get_abnormal(template, length)
            else:
                # if the 'SAP_template' parameter is a None value, that means the user did not specify a template,
                # thus the function need to create the template first
                if regular:
                    template = [('intro', 4)]
                    hasBridge = random.choice([0, 1])
                    i = richness
                    while i != 0:
                        i -= 1
                        template.extend([('verse', 4), ('pre_chorus', 4), ('chorus', 4)])
                        if _get_num_of_bars(template) >= length:
                            break
                        if hasBridge and i != 0:
                            template.append(('bridge', 4))
                    template.append(('outro', 4))
                    if normal:
                        template = _get_normal(template, length)
                    else:
                        template = _get_abnormal(template, length)
                else:
                    template = []
                    template.extend([('intro', 4)] if random.choice([0, 1]) == 1 else [])
                    i = richness
                    prevSection = None
                    while i != 0:
                        i -= 1
                        for i2 in range(4):
                            chosen_sec = random.choice(['verse', 'pre_chorus', 'chorus', 'bridge'])
                            while chosen_sec == prevSection:
                                chosen_sec = random.choice(['verse', 'pre_chorus', 'chorus', 'bridge'])
                            template.append((chosen_sec, 4))
                            prevSection = chosen_sec
                    template.extend([('outro', 4)] if random.choice([0, 1]) == 1 else [])

                    if normal:
                        template = _get_normal(template, length)
                    else:
                        template = _get_abnormal(template, length)
                arrangement = template
                printer.print(arrangement)
                # prompt the user if the generated SAP is satisified
                if non_stop:
                    break
                msg = input("Do you like this song arrangement? (y/n)")
                if msg == 'y' or msg == 'yes':
                    printer.print('SAP Done!')
                    break
        return SectionNode.get_arrangement_nodes(project, arrangement)

    @staticmethod
    def get_arrangement_nodes(project, arrangement: list):
        def add_wo_duplicate(sec_name, sec_type, sec_length):
            for compare_sec in new_arrangement:
                if compare_sec.name == sec_name and compare_sec.section_type == sec_type and compare_sec.length == sec_length:
                    new_arrangement.append(compare_sec)
                    return
            new_arrangement.append(SectionNode(project, sec_name, sec_type, sec_length))
        sapTypeConverter = {'intro': SectionType.Intro,
                            'verse': SectionType.Verse,
                            'pre_chorus': SectionType.PreChorus,
                            'chorus': SectionType.Chorus,
                            'bridge': SectionType.Bridge,
                            'outro': SectionType.Outro,
                            'Intro': SectionType.Intro,
                            'Verse': SectionType.Verse,
                            'Pre_chorus': SectionType.PreChorus,
                            'Chorus': SectionType.Chorus,
                            'Bridge': SectionType.Bridge,
                            'Outro': SectionType.Outro,
                            'PreChorus': SectionType.PreChorus,
                            'prechorus': SectionType.PreChorus,
                            }
        nodes = {}
        name_count = {}
        new_arrangement = []
        if len(arrangement[0]) == 2:
            for sec in arrangement:
                if sec[0] in nodes:
                    new_name = sec[0]
                    if BarStepTick(sec[1], time_signature=project.time_signature) != nodes[sec[0]].length:
                        new_name = f"{sec[0]}_{name_count[sec[0]]}"
                        name_count[sec[0]] += 1
                        nodes[new_name] = SectionNode(project, new_name, sapTypeConverter[sec[0]],
                                                      BarStepTick(sec[1], time_signature=project.time_signature))
                    new_arrangement.append(nodes[new_name])
                else:
                    name_count[sec[0]] = 1
                    nodes[sec[0]] = SectionNode(project, sec[0], sapTypeConverter[sec[0]],
                                                BarStepTick(sec[1], time_signature=project.time_signature))
                    new_arrangement.append(nodes[sec[0]])
        else:
            # That means the node names are provided, no need to generate new names.
            for sec in arrangement:
                add_wo_duplicate(sec[0], sapTypeConverter[sec[1]], BarStepTick(sec[2], time_signature=project.time_signature))
        for sec in new_arrangement:
            project.node_graph.register_node(sec)
        return new_arrangement


