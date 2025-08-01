import json
from .. import ClipNode
from tomi import (NodeType, BarStepTick, MIDIProcessor, MIDIType, Groove, GrooveSpeed,
                   MIDINoteList, ItemSelector, Mode, SongGenre)
import os


class MIDINode(ClipNode):
    def __init__(self,
                 project,
                 name: str,
                 midi_type: MIDIType,
                 length: BarStepTick = None,
                 dynamic=None,
                 complexity=None,
                 groove_speed: GrooveSpeed = GrooveSpeed.Normal,
                 groove: Groove = None,
                 midi: MIDIProcessor | str = None,
                 dependent_node: 'ClipNode' = None,
                 dependent_type: MIDIType = None,
                 root_progression: list = None,
                 min_progression_count: int = 4,
                 fit_key: bool = True):
        if complexity is None:
            complexity = [0, 100]
        if dynamic is None:
            dynamic = [0, 100]
        super(MIDINode, self).__init__(NodeType.MidiClip, project, name, dependent_node, dependent_type)
        self.genre = self.project.genre
        self.genre_set = True
        self.midi_type = midi_type
        self.dynamic = dynamic if dynamic is not None and len(dynamic) == 2 else (None, None)
        self.complexity = complexity if complexity is not None and len(complexity) == 2 else (None, None)
        self.groove = None
        self.input_groove = groove
        self.use_input_groove_only = True if self.input_groove is not None else False
        self.groove_set = False
        self.groove_speed = groove_speed
        self.groove_speed_set = True
        self.midi_candidates = ItemSelector(self, 'midi_candidates', MIDIProcessor | int)
        self.groove_candidates = ItemSelector(self, 'groove_candidates', Groove | int)
        self.midi = None
        self.input_midi = midi
        self.use_dependent_midi_only = True if self.dependent_node is not None else False
        self.use_input_midi_only = True if self.input_midi is not None and not self.use_dependent_midi_only else False
        self.length = BarStepTick() if length is None else length
        self.length_set = True if not self.length.is_empty() else False
        self.root_progression = root_progression if root_progression is None else tuple(root_progression)
        self.min_progression_count = min_progression_count
        self.transpose_steps = 0
        self.transpose_steps_set = False
        self.fit_key = fit_key

    def run(self):
        super(MIDINode, self).run()
        self.update()
        self._add_to_project_lib()

    def clear(self):
        super(MIDINode, self).clear()
        self._remove_from_project_lib()
        self.midi = None
        self.groove = None
        self.midi_candidates.clear()
        self.groove_candidates.clear()
        self.length_set = True if self.length else False
        self.transpose_steps = 0
        self.transpose_steps_set = False
        self.fit_key = True
        self.use_dependent_midi_only = True if self.dependent_node is not None else False
        self.use_input_midi_only = True if self.input_midi is not None and not self.use_dependent_midi_only else False
        self.groove_speed_set = True
        self.use_input_groove_only = True if self.input_groove is not None else False
        self.groove_set = False
        self.genre_set = True

    def update_gui_configs(self):
        self.gui_configs.update_config(param_name='id', gui_name='Node ID', param_type=int, value=self.id, mutable=False)
        self.gui_configs.update_config(param_name='name', gui_name='Node Name', param_type=str, value=self.gui_name, mutable=True)
        self.gui_configs.update_config(param_name='_parents', gui_name='Parent Nodes', param_type=set, value={n.gui_name for n in self._parents}, mutable=False, special_entry='string_container')
        self.gui_configs.update_config(param_name='_childs', gui_name='Child Nodes', param_type=set, value={n.gui_name for n in self._childs}, mutable=False, special_entry='string_container')
        self.gui_configs.update_config(param_name='genre', gui_name='MIDI Genre', param_type=SongGenre, value=self.genre, mutable=True)
        self.gui_configs.update_config(param_name='genre_set', gui_name='MIDI Genre Constraint', param_type=bool, value=self.genre_set, mutable=True)
        self.gui_configs.update_config(param_name='midi_type', gui_name='MIDI Type', param_type=MIDIType, value=self.midi_type, mutable=True)
        self.gui_configs.update_config(param_name='length', gui_name='Duration', param_type=BarStepTick, value=self.length, mutable=True)
        self.gui_configs.update_config(param_name='length_set', gui_name='Fixed Length', param_type=bool, value=self.length_set, mutable=self.length is not None)
        self.gui_configs.update_config(param_name='dynamic', gui_name='Dynamic', param_type=tuple, value=self.dynamic, mutable=True, special_entry='min_max_range')
        self.gui_configs.update_config(param_name='complexity', gui_name='Complexity', param_type=tuple, value=self.complexity, mutable=True, special_entry='min_max_range')
        self.gui_configs.update_config(param_name='current_groove', gui_name='Current Groove', param_type=Groove, value=self.groove, mutable=False)
        self.gui_configs.update_config(param_name='input_groove_values', gui_name='Input Groove Values', param_type=list, value=self.input_groove.groove.tolist() if self.input_groove is not None else [], mutable=True)
        self.gui_configs.update_config(param_name='input_groove', gui_name='Input Groove', param_type=Groove, value=self.input_groove, mutable=False)
        self.gui_configs.update_config(param_name='groove_set', gui_name='Fixed Groove', param_type=bool, value=self.groove_set or self.use_input_groove_only, mutable=self.groove is not None)
        self.gui_configs.update_config(param_name='groove_speed', gui_name='Groove Speed', param_type=GrooveSpeed, value=self.groove_speed, mutable=True)
        self.gui_configs.update_config(param_name='groove_speed_set', gui_name='Fixed Groove Speed', param_type=bool, value=self.groove_speed_set, mutable=True)
        self.gui_configs.update_config(param_name='midi_candidates', gui_name='MIDI Candidates', param_type=ItemSelector, value=self.midi_candidates, mutable=True)
        self.gui_configs.update_config(param_name='midi_candidates_count', gui_name='MIDI Candidates Count', param_type=int, value=len(self.midi_candidates), mutable=False)
        self.gui_configs.update_config(param_name='groove_candidates', gui_name='Groove Candidates', param_type=ItemSelector, value=self.groove_candidates, mutable=True)
        self.gui_configs.update_config(param_name='groove_candidates_count', gui_name='Groove Candidates Count', param_type=int, value=len(self.groove_candidates), mutable=False)
        self.gui_configs.update_config(param_name='midi', gui_name='Current MIDI', param_type=MIDIProcessor, value=self.midi, mutable=False)
        self.gui_configs.update_config(param_name='input_midi', gui_name='Input MIDI', param_type=str, value=self.input_midi.name if isinstance(self.input_midi, MIDIProcessor) else self.input_midi, mutable=True)
        self.gui_configs.update_config(param_name='use_input_midi_only', gui_name='Use Input MIDI Only', param_type=bool, value=self.use_input_midi_only, mutable=self.input_midi is not None)
        self.gui_configs.update_config(param_name='use_dependent_midi_only', gui_name='Use Dependent MIDI Only', param_type=bool, value=self.use_dependent_midi_only, mutable=self.dependent_node is not None)
        self.gui_configs.update_config(param_name='root_progression', gui_name='Root Progression', param_type=tuple, value=self.root_progression, mutable=True)
        self.gui_configs.update_config(param_name='min_progression_count', gui_name='Min Progression Count', param_type=int, value=self.min_progression_count, mutable=True)
        self.gui_configs.update_config(param_name='current_progression_count', gui_name='Current Progression Count', param_type=int, value=self.midi.progression_count, mutable=False)
        self.gui_configs.update_config(param_name='transpose_steps', gui_name='Transpose Steps', param_type=int, value=self.transpose_steps, mutable=True)
        self.gui_configs.update_config(param_name='transpose_steps_set', gui_name='Fixed Transpose Steps', param_type=bool, value=self.transpose_steps_set, mutable=True)
        self.gui_configs.update_config(param_name='fit_key', gui_name='Fit Key', param_type=bool, value=self.fit_key, mutable=True)

    def config_update(self, param_name: str, param_value):
        name, loc = self._get_param_name_and_loc(param_name)
        match name:
            case 'name': self.name = param_value
            case 'genre': self.set_genre(param_value)
            case 'genre_set': self.set_genre_set(param_value)
            case 'midi_type': self.set_midi_type(param_value)
            case 'length': self.set_length(self._get_updated_bst(self.length, loc, param_value))
            case 'length_set': self.set_length_set(param_value)
            case 'dynamic':
                match loc:
                    case 0: self.set_dynamic((param_value, self.dynamic[1]))
                    case 1: self.set_dynamic((self.dynamic[0], param_value))
                    case _: raise ValueError('Dynamic Range Loc must be 0 or 1.')
            case 'complexity':
                match loc:
                    case 0: self.set_complexity((param_value, self.complexity[1]))
                    case 1: self.set_complexity((self.complexity[0], param_value))
                    case _: raise ValueError('Complexity Range Loc must be 0 or 1.')
            case 'input_groove_values': self.set_input_groove(param_value)
            case 'groove_set': self.set_groove_set(param_value)
            case 'midi_candidates': self.midi_candidates.gui_select(param_value)
            case 'groove_candidates': self.groove_candidates.gui_select(param_value)
            case 'midi_candidates!r': self.midi_candidates.gui_random_select()
            case 'groove_candidates!r': self.groove_candidates.gui_random_select()
            case 'groove_speed': self.set_groove_speed(param_value)
            case 'groove_speed_set': self.set_groove_speed_set(param_value)
            case 'input_midi': self.set_input_midi(param_value)
            case 'use_input_midi_only': self.set_use_input_midi_only(param_value)
            case 'use_dependent_midi_only': self.set_use_dependent_midi_only(param_value)
            case 'root_progression': self.set_root_progression(param_value)
            case 'min_progression_count': self.set_min_progression_count(param_value)
            case 'transpose_steps': self.set_transpose_steps(param_value)
            case 'transpose_steps_set': self.set_transpose_steps_set(param_value)
            case 'fit_key': self.set_fit_key(param_value)

    def set_transpose_steps(self, transpose_steps: int):
        assert isinstance(transpose_steps, int)
        if self.transpose_steps != transpose_steps:
            self.transpose_steps = transpose_steps
            self.transpose_steps_set = True

    def set_transpose_steps_set(self, transpose_steps_set: bool):
        assert isinstance(transpose_steps_set, bool)
        if self.transpose_steps_set != transpose_steps_set:
            self.transpose_steps_set = transpose_steps_set
            if not transpose_steps_set:
                if self.midi:
                    self.transpose_steps = self.midi.key - self.key_mode
                else:
                    self.transpose_steps = 0

    def set_fit_key(self, fit_key: bool):
        assert isinstance(fit_key, bool)
        self.fit_key = fit_key

    def set_root_progression(self, root_progression: list | tuple):
        assert isinstance(root_progression, list) and all(isinstance(x, int) and 1<=x<=7 for x in root_progression)
        if self.root_progression != root_progression:
            self.root_progression = (root_progression[0],) if len(root_progression) == 1 else tuple(root_progression)
            self.update()

    def set_min_progression_count(self, min_progression_count: int):
        assert isinstance(min_progression_count, int) and min_progression_count >= 0
        if self.min_progression_count != min_progression_count:
            self.min_progression_count = min_progression_count
            self.update()

    def set_use_dependent_midi_only(self, use_dependent_midi_only: bool):
        assert isinstance(use_dependent_midi_only, bool)
        if self.use_dependent_midi_only != use_dependent_midi_only:
            if self.dependent_node is not None:
                self.use_dependent_midi_only = use_dependent_midi_only
                if self.use_dependent_midi_only and self.use_input_midi_only:
                    self.use_input_midi_only = False
                self.update()

    def set_use_input_midi_only(self, use_input_midi_only: bool):
        assert isinstance(use_input_midi_only, bool)
        if self.use_input_midi_only != use_input_midi_only:
            if self.input_midi:
                self.use_input_midi_only = use_input_midi_only
                if self.use_dependent_midi_only and self.use_input_midi_only:
                    self.use_dependent_midi_only = False
                self.update()

    def _check_audio_file_exists(self, file_path: str):
        assert isinstance(file_path, str)
        return os.path.isfile(file_path) and file_path.endswith('.mid')

    def set_input_midi(self, input_midi: str | MIDIProcessor):
        assert isinstance(input_midi, (str, MIDIProcessor))
        if isinstance(input_midi, str) and not self._check_audio_file_exists(input_midi):
            return
        if self.input_midi != input_midi:
            self.input_midi = input_midi
            self.use_input_midi_only = True
            self.update()

    def set_groove_speed(self, groove_speed: GrooveSpeed):
        assert isinstance(groove_speed, GrooveSpeed)
        if groove_speed != self.groove_speed:
            self.groove_speed = groove_speed
            self.groove_speed_set = True
            self.update_groove()

    def set_groove_speed_set(self, groove_speed_set: bool):
        assert isinstance(self.groove_speed_set, bool)
        if self.groove_speed_set != groove_speed_set:
            self.groove_speed_set = groove_speed_set
            self.update_groove()

    def set_groove_set(self, groove_set: bool):
        assert isinstance(groove_set, bool)
        if self.groove_set != groove_set:
            self.groove_set = groove_set
            self.update_groove()

    def set_input_groove(self, groove: list[int]):
        assert isinstance(groove, list)
        if self.input_groove is None and not groove:
            self.use_input_groove_only = False
            return
        if self.input_groove is None or groove != self.input_groove.groove.tolist():
            if not groove:
                self.input_groove = None
                self.use_input_groove_only = False
            else:
                g = Groove(groove, self.input_groove.midi_type if self.input_groove is not None else self.midi_type)
                self.input_groove = g
                self.use_input_groove_only = True
            self.update_groove()

    def set_genre(self, genre: SongGenre):
        assert isinstance(genre, SongGenre)
        if genre != self.genre:
            self.genre = genre
            self.genre_set = True
            self.update()

    def set_genre_set(self, genre_set: bool):
        assert isinstance(genre_set, bool)
        if genre_set != self.genre_set:
            self.genre_set = genre_set
            self.update()

    def set_midi_type(self, midi_type: MIDIType):
        assert isinstance(midi_type, MIDIType)
        if midi_type != self.midi_type:
            self.midi_type = midi_type
            self.update()

    def set_length(self, length: BarStepTick):
        assert isinstance(length, BarStepTick)
        if length != self.length and not length.is_empty():
            self.length = length
            self.length_set = True
            self.update()

    def set_length_set(self, length_set: bool):
        assert isinstance(length_set, bool)
        if length_set != self.length_set:
            self.length_set = length_set
            self.update()

    def set_dynamic(self, dynamic_range: tuple[float, float]):
        assert dynamic_range[0] <= dynamic_range[1]
        self.dynamic = dynamic_range
        self.update()

    def set_complexity(self, complexity_range: tuple[float, float]):
        assert complexity_range[0] <= complexity_range[1]
        self.complexity = complexity_range
        self.update()

    def on_selector_update(self, selector_name: str):
        if selector_name == 'midi_candidates':
            self.load_midi_data(self.midi_candidates.get_current_value())
            self.update_groove()
        elif selector_name == 'groove_candidates':
            self.load_groove_data(self.groove_candidates.get_current_value())
            self.need_sync = True

    def equals(self, other):
        return (isinstance(other, MIDINode) and
                self.midi_type == other.midi_type and
                self.length == other.length and
                self.dynamic == other.dynamic and
                self.complexity == other.complexity and
                self.groove == other.groove and
                self.groove_speed == other.groove_speed and
                self.midi == other.midi and
                self.root_progression == other.root_progression and
                self.min_progression_count == other.min_progression_count)

    def _remove_from_project_lib(self):
        if self.midi_type.name in self.project.midi_lib and self in self.project.midi_lib[self.midi_type.name]:
            self.project.midi_lib[self.midi_type.name].remove(self)

    def _add_to_project_lib(self):
        if self.midi_type.name not in self.project.midi_lib:
            self.project.midi_lib[self.midi_type.name] = []
        if self not in self.project.midi_lib[self.midi_type.name]:
            self.project.midi_lib[self.midi_type.name].append(self)

    def update(self):
        self.update_midi()
        self.update_groove()

    def update_midi(self):
        self.midi_candidates.clear()
        self.init_midi()
        self.search_midi()
        if self.use_dependent_midi_only:
            self.midi_candidates.select('--DEPEND--')
            self.load_midi_data(self.midi_candidates.get_current_value())
        elif self.use_input_midi_only:
            self.midi_candidates.select('--INPUT--')
            self.load_midi_data(self.midi_candidates.get_current_value())
        if self.midi is None:
            self.midi_candidates.random_select()
            self.load_midi_data(self.midi_candidates.get_current_value())
        else:
            if len(self.midi_candidates) == 0:
                self.midi = None
            elif self.midi_candidates.current_key in ('--DEPEND--', '--INPUT--'):
                pass
            elif self.midi.name in self.midi_candidates.keys():
                self.midi_candidates.select(self.midi.name)
            else:
                self.midi_candidates.random_select()
                self.load_midi_data(self.midi_candidates.get_current_value())
        if self.midi is None:
            self.log(f"{self.name} is empty.")
        self.need_sync = True

    def update_groove(self):
        self.groove_candidates.clear()
        self.init_groove()
        if self.midi is not None:
            self.search_groove()
            if self.use_input_groove_only:
                self.groove_candidates.select('--INPUT--')
                self.load_groove_data(self.groove_candidates.get_current_value())
            elif self.groove_set:
                self.change_groove()
            else:
                self.groove_candidates.select('--ORIGINAL--')
                self.groove = self.midi.groove
        self.need_sync = True

    def search_groove(self):
        query = (f"SELECT id, midi_type "
                 f"FROM TOMI_MIDI_Grooves "
                 f"WHERE midi_type = '{self.midi_type.name}' "
                 f"AND note_group_count!=0 "
                 f"AND {self.midi.progression_count} % note_group_count == 0")
        if self.groove_speed_set:
            query += f" AND speed='{self.groove_speed.name}'"
        output = self.search_db(query)
        for g in output:
            self.groove_candidates[f"{g[0]}-{g[1]}"] = g[0]

    def search_midi(self):
        select_clause = f"SELECT m.id, m.file_name FROM TOMI_MIDI AS m "
        if self.genre_set:
            select_clause += f"JOIN TOMI_MIDI_File_Genres AS fg ON fg.file_id=m.id JOIN TOMI_Genres as g ON fg.genre_id=g.id "
        if self.midi_type == MIDIType.Melody:
            select_clause += f"WHERE m.midi_type = '{MIDIType.Melody.name}'"
        elif self.midi_type == MIDIType.Chord:
            select_clause += f"WHERE m.midi_type in ('{MIDIType.Composite.name}', '{MIDIType.Chord.name}')"
        else:
            select_clause += f"WHERE m.midi_type='{self.midi_type.name}'"
        condition_clause = {'min_progression_count': f"m.progression_count>={self.min_progression_count}"}
        if self.length_set and self.length is not None:
            condition_clause['length'] = f"m.bar={self.length.bar}"
        if self.genre_set:
            condition_clause['genre'] = f"g.genre='{self.genre.name.lower()}'"
        if self.dynamic[0] is not None:
            condition_clause['dynamic_min'] = f"m.midi_zd >= {self.dynamic[0]}"
        if self.dynamic[1] is not None:
            condition_clause['dynamic_max'] = f"m.midi_zd <= {self.dynamic[1]}"
        if self.complexity[0] is not None:
            condition_clause['complexity_min'] = f"m.midi_zs >= {self.complexity[0]}"
        if self.complexity[1] is not None:
            condition_clause['complexity_max'] = f"m.midi_zs <= {self.complexity[1]}"
        if self.root_progression is not None and len(self.root_progression) > 0:
            condition_clause['root_progression'] = f"{'m.maj_root_prog' if self.mode == Mode.Major else 'm.min_root_prog'} ='{self.root_progression}'"
        first_try_clause = " AND ".join(condition_clause.values())
        query = f"{select_clause} AND {first_try_clause}" if first_try_clause else select_clause
        output = self.search_db(query)
        if len(output) == 0:
            second_try_clause = " AND ".join([v for k, v in condition_clause.items() if k != 'root_progression'])
            query = f"{select_clause} AND {second_try_clause}" if second_try_clause else select_clause
            output = self.search_db(query)
            if len(output) == 0:
                third_try_clause = " AND ".join([condition_clause['length']] if 'length' in condition_clause else [])
                query = f"{select_clause} AND {third_try_clause}" if third_try_clause else select_clause
                output = self.search_db(query)
        for midi in output:
            self.midi_candidates[midi[1]] = midi[0]

    def get_notelist(self):
        assert self.midi is not None, f'{self.name} is not initialized.'
        notelist = self.midi.get_notelist()
        for note in notelist:
            note.velocity = max(70, note.velocity)
        return notelist

    def init_midi(self):
        if self.dependent_node is not None:
            if not self.dependent_type:
                self.dependent_type = MIDIType.Composite
            assert isinstance(self.dependent_node, MIDINode)
            if self.dependent_node.midi is not None:
                self.midi_candidates['--DEPEND--'] = self.dependent_node.midi.get_processor_by_type(self.dependent_type)
        else:
            self.use_dependent_midi_only = False
        if self.input_midi is not None:
            self.midi_candidates['--INPUT--'] = MIDIProcessor(self.input_midi, target_bpm=self.bpm, midi_type=self.midi_type) if isinstance(self.input_midi, str) else self.input_midi
        else:
            self.use_input_midi_only = False

    def init_groove(self):
        if self.input_groove is not None:
            self.groove_candidates['--INPUT--'] = self.input_groove
        else:
            self.use_input_groove_only = False
        if self.groove is None:
            self.groove_set = False
        if self.midi is not None:
            self.groove_candidates['--ORIGINAL--'] = self.midi.groove

    def load_midi_data(self, midi_info: int | MIDIProcessor):
        if midi_info is None:
            return
        if isinstance(midi_info, int):
            midi_info = self.search_db(f"SELECT file_name, note_list, maj_key FROM TOMI_MIDI WHERE id={midi_info}")[0]
            self.midi = MIDIProcessor(MIDINoteList(json.loads(midi_info[1])), self.bpm, self.midi_type, fit_beat=False, name=midi_info[0])
        else:
            self.midi = midi_info
        if not self.transpose_steps_set:
            self.transpose_steps = self.midi.key - self.key_mode
        self.length = self.midi.ceil_bst
        self.midi.piano_roll.print_roll()

    def change_groove(self):
        try:
            midi = MIDIProcessor(self.midi.piano_roll.apply_groove(self.groove), self.bpm, self.midi_type,
                                 fit_beat=False, name=self.midi.name)
            self.midi = midi
        except:
            pass

    def load_groove_data(self, groove_info: int | Groove):
        if isinstance(groove_info, int):
            groove_info = self.search_db(f"SELECT midi_type, groove_list FROM TOMI_GROOVE_TABLE WHERE id={groove_info}")[0]
            self.groove = Groove(json.loads(groove_info[1]), MIDIType(groove_info[0]))
        else:
            self.groove = groove_info
        self.change_groove()

