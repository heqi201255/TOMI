from .. import ClipNode
from tomi import (NodeType, ADSRSampleKeyMapping, KeyMode, BarStepTick, AudioType,
                   OrderedDotdict, ItemSelector)
import librosa
import os
import re


class AudioNode(ClipNode):
    def __init__(self,
                 project,
                 name: str,
                 audio_type: AudioType,
                 query: list[str] = None,
                 loop: bool = True,
                 bpm_range: tuple[int | float, int | float] = (0, 999),
                 audio_file_path: str = None,
                 minlen: BarStepTick = None,
                 maxlen: BarStepTick = None,
                 reverse: bool = False,
                 fit_key: bool = False,
                 transpose_steps: int = 0,
                 fit_tempo: bool = False):
        super(AudioNode, self).__init__(NodeType.AudioClip, project, name)
        if query is None:
            query = []
        self.audio_type = audio_type
        self.loop = loop
        self.bpm_range = bpm_range
        self.query = set(query)
        self.minlen = minlen if minlen is not None else BarStepTick()
        self.maxlen = maxlen if maxlen is not None else BarStepTick(16)
        self.reverse = reverse
        self.transpose_steps = transpose_steps
        self.transpose_steps_set = False if transpose_steps == 0 else True
        self.fit_key = fit_key
        self.fit_tempo = fit_tempo
        self.file_path = audio_file_path
        self.use_input_file_only = True if self.file_path is not None else False
        self.sample_candidates = ItemSelector(self, 'sample_candidates', tuple)
        self.current_sample = OrderedDotdict()
        self.length = None

    def update_gui_configs(self):
        self.gui_configs.update_config(param_name='id', gui_name='Node ID', param_type=int, value=self.id, mutable=False)
        self.gui_configs.update_config(param_name='name', gui_name='Node Name', param_type=str, value=self.gui_name, mutable=True)
        self.gui_configs.update_config(param_name='_parents', gui_name='Parent Nodes', param_type=set, value={n.gui_name for n in self._parents}, mutable=False, special_entry='string_container')
        self.gui_configs.update_config(param_name='_childs', gui_name='Child Nodes', param_type=set, value={n.gui_name for n in self._childs}, mutable=False, special_entry='string_container')
        self.gui_configs.update_config(param_name='audio_type', gui_name='Audio Type', param_type=AudioType, value=self.audio_type, mutable=True)
        self.gui_configs.update_config(param_name='loop', gui_name='Loop', param_type=bool, value=self.loop, mutable=True)
        self.gui_configs.update_config(param_name='bpm_range', gui_name='BPM Range', param_type=tuple, value=self.bpm_range, mutable=True, special_entry='min_max_range')
        self.gui_configs.update_config(param_name='query', gui_name='Search Query', param_type=set, value=self.query, mutable=True, special_entry='string_container')
        self.gui_configs.update_config(param_name='minlen', gui_name='Min Length', param_type=BarStepTick, value=self.minlen, mutable=True)
        self.gui_configs.update_config(param_name='maxlen', gui_name='Max Length', param_type=BarStepTick, value=self.maxlen, mutable=True)
        self.gui_configs.update_config(param_name='reverse', gui_name='Reverse', param_type=bool, value=self.reverse, mutable=True)
        self.gui_configs.update_config(param_name='transpose_steps', gui_name='Transpose Steps', param_type=int, value=self.transpose_steps, mutable=True)
        self.gui_configs.update_config(param_name='transpose_steps_set', gui_name='Fixed Transpose Steps', param_type=bool, value=self.transpose_steps_set, mutable=True)
        self.gui_configs.update_config(param_name='fit_key', gui_name='Fit Key', param_type=bool, value=self.fit_key, mutable=True)
        self.gui_configs.update_config(param_name='fit_tempo', gui_name='Fit Tempo', param_type=bool, value=self.fit_tempo, mutable=True)
        self.gui_configs.update_config(param_name='file_path', gui_name='Dedicated File', param_type=str, value=self.file_path, mutable=True)
        self.gui_configs.update_config(param_name='use_input_file_only', gui_name='Use Dedicated File Only', param_type=bool, value=self.use_input_file_only, mutable=self.file_path is not None)
        self.gui_configs.update_config(param_name='sample_candidates', gui_name='Sample Candidates', param_type=ItemSelector, value=self.sample_candidates, mutable=True)
        self.gui_configs.update_config(param_name='sample_candidates_count', gui_name='Sample Candidates Count', param_type=int, value=len(self.sample_candidates), mutable=False)
        self.gui_configs.update_config(param_name='length', gui_name='Duration', param_type=BarStepTick, value=self.length, mutable=False)
        self.gui_configs.update_config(param_name='selected_sample_name', gui_name='Selected Sample Name', param_type=str, value=self.current_sample.name, mutable=False)
        self.gui_configs.update_config(param_name='selected_sample_path', gui_name='Selected Sample Path', param_type=str, value=self.current_sample.path, mutable=False)
        self.gui_configs.update_config(param_name='selected_sample_tempo', gui_name='Selected Sample Tempo', param_type=float, value=self.current_sample.tempo, mutable=False)
        self.gui_configs.update_config(param_name='selected_sample_key', gui_name='Selected Sample Key', param_type=KeyMode, value=self.current_sample.key, mutable=False)
        self.gui_configs.update_config(param_name='selected_sample_length', gui_name='Selected Sample Length', param_type=BarStepTick, value=self.current_sample.bst_length, mutable=False)
        self.gui_configs.update_config(param_name='selected_sample_loop', gui_name='Selected Sample is Loop', param_type=bool, value=self.current_sample.loop, mutable=False)

    def config_update(self, param_name: str, param_value):
        name, loc = self._get_param_name_and_loc(param_name)
        match name:
            case 'name': self.name = param_value
            case 'audio_type': self.set_audio_type(param_value)
            case 'loop': self.set_loop(param_value)
            case 'bpm_range':
                match loc:
                    case 0:  self.set_bpm_range((param_value, self.bpm_range[1]))
                    case 1:  self.set_bpm_range((self.bpm_range[0], param_value))
                    case _:  raise ValueError('BPM Range Loc must be 0 or 1.')
            case 'query': self.set_query(param_value)
            case 'minlen': self.set_minlen(self._get_updated_bst(self.minlen, loc, param_value))
            case 'maxlen': self.set_maxlen(self._get_updated_bst(self.maxlen, loc, param_value))
            case 'reverse': self.set_reverse(param_value)
            case 'transpose_steps': self.set_transpose_steps(param_value)
            case 'transpose_steps_set': self.set_transpose_steps(param_value)
            case 'fit_key': self.set_fit_key(param_value)
            case 'fit_tempo': self.set_fit_tempo(param_value)
            case 'file_path': self.set_audio_file_path(param_value)
            case 'use_input_file_only': self.set_use_audio_file_only(param_value)
            case 'sample_candidates': self.sample_candidates.gui_select(param_value)
            case 'sample_candidates!r': self.sample_candidates.gui_random_select()

    def on_selector_update(self, selector_name: str):
        if selector_name == 'sample_candidates':
            self.load_sample_data(self.sample_candidates.get_current_value())
            self.need_sync = True

    def __getstate__(self):
        state = super().__getstate__()
        state.pop('sample_candidates')
        return state

    def set_audio_type(self, audio_type: AudioType):
        assert isinstance(audio_type, AudioType)
        if self.audio_type != audio_type:
            self._remove_from_project_lib()
            self.audio_type = audio_type
            self._add_to_project_lib()
            self.update()

    def set_loop(self, loop: bool):
        assert isinstance(loop, bool)
        if self.loop != loop:
            self.loop = loop
            self.update()

    def set_bpm_range(self, bpm_range: tuple[int, int]):
        assert bpm_range[0] <= bpm_range[1]
        if self.bpm_range != bpm_range:
            self.bpm_range = bpm_range
            self.update()

    def set_query(self, query: list[str]):
        assert isinstance(query, list)
        if self.query != query:
            self.query = query
            self.update()

    def set_minlen(self, minlen: BarStepTick):
        assert isinstance(minlen, BarStepTick) and minlen < self.maxlen
        if self.minlen != minlen:
            self.minlen = minlen
            self.update()

    def set_maxlen(self, maxlen: BarStepTick):
        assert isinstance(maxlen, BarStepTick) and maxlen > self.minlen
        if self.maxlen != maxlen:
            self.maxlen = maxlen
            self.update()

    def set_reverse(self, reverse: bool):
        assert isinstance(reverse, bool)
        self.reverse = reverse

    def set_transpose_steps(self, transpose_steps: int):
        assert isinstance(transpose_steps, int)
        if self.transpose_steps != transpose_steps:
            self.transpose_steps = transpose_steps
            self.transpose_steps_set = True

    def set_transpose_steps_set(self, steps_set: bool):
        assert isinstance(steps_set, bool)
        if self.transpose_steps_set != steps_set:
            self.transpose_steps_set = steps_set
            if not steps_set:
                if self.current_sample:
                    self.transpose_steps = self.current_sample.key - self.key_mode if self.current_sample else 0
                else:
                    self.transpose_steps = 0

    def set_fit_key(self, fit_key: bool):
        assert isinstance(fit_key, bool)
        self.fit_key = fit_key

    def set_fit_tempo(self, fit_tempo: bool):
        assert isinstance(fit_tempo, bool)
        self.fit_tempo = fit_tempo

    def set_audio_file_path(self, audio_file_path: str):
        assert isinstance(audio_file_path, str)
        if self._check_audio_file_exists(audio_file_path):
            self.file_path = audio_file_path
            self.use_input_file_only = True
            self.update()

    def set_use_audio_file_only(self, use_audio_file_only: bool):
        assert isinstance(use_audio_file_only, bool)
        if use_audio_file_only != self.use_input_file_only:
            if use_audio_file_only and self.file_path is not None:
                self.set_audio_file_path(self.file_path)
            else:
                self.use_input_file_only = False
                self.update()

    def _check_audio_file_exists(self, file_path: str):
        assert isinstance(file_path, str)
        return os.path.isfile(file_path) and any((file_path.endswith(ext) for ext in ('.wav', '.mp3', '.aac', '.flac')))

    def equals(self, other):
        return (isinstance(other, AudioNode) and
                self.audio_type == other.audio_type and
                self.loop == other.loop and
                self.bpm_range == other.bpm_range and
                self.query == other.query and
                self.minlen == other.minlen and
                self.maxlen == other.maxlen and
                self.reverse == other.reverse and
                self.transpose_steps == other.transpose_steps and
                self.fit_key == other.fit_key and
                self.file_path == other.file_path and
                self.fit_tempo == other.fit_tempo and
                self.current_sample == other.current_sample and
                self.length == other.length)

    def run(self):
        super(AudioNode, self).run()
        self.update()
        self._add_to_project_lib()

    def _add_to_project_lib(self):
        if self.audio_type.name not in self.project.sample_lib:
            self.project.sample_lib[self.audio_type.name] = []
        if self not in self.project.sample_lib[self.audio_type.name]:
            self.project.sample_lib[self.audio_type.name].append(self)

    def _remove_from_project_lib(self):
        if self.audio_type.name in self.project.sample_lib and self in self.project.sample_lib[self.audio_type.name]:
            self.project.sample_lib[self.audio_type.name].remove(self)

    def clear(self):
        super(AudioNode, self).clear()
        self._remove_from_project_lib()
        self.sample_candidates.clear()
        self.current_sample = OrderedDotdict()
        self.use_input_file_only = True if self.file_path is not None else False
        self.length = None

    def update(self):
        self.sample_candidates.clear()
        if self.file_path:
            self.log(f"Loading audio sample {self.file_path}")
            self.load_audio_file()
        else:
            self.use_input_file_only = False
        if not self.use_input_file_only:
            if self.minlen is not None and self.maxlen is not None:
                results = self.search_sample()
                if len(results) == 0:
                    results = self.search_sample_backup()
                    if len(results) == 0:
                        results = self.search_sample_backup2()
                        if len(results) == 0:
                            self.log(f"No matched sample found!")
                        else:
                            self._add_to_candidates(results)
                            self.log(f"{len(self.sample_candidates)} matched samples was found")
                    else:
                        self._add_to_candidates(results)
                        self.log(f"{len(self.sample_candidates)} matched samples was found")
                else:
                    self._add_to_candidates(results)
                    self.log(f"{len(self.sample_candidates)} matched samples was found")
        if not self.current_sample:
            if len(self.sample_candidates) != 0:
                self.sample_candidates.random_select()
                self.load_sample_data(self.sample_candidates.get_current_value())
            else:
                self.current_sample = OrderedDotdict()
                # raise NoAvailableSamplesError
        else:
            if len(self.sample_candidates) == 0:
                self.current_sample = OrderedDotdict()
                # raise NoAvailableSamplesError
            elif self.current_sample.path not in self.sample_candidates.keys():
                self.sample_candidates.random_select()
                self.load_sample_data(self.sample_candidates.get_current_value())
            else:
                self.sample_candidates.select(self.current_sample.path)
        self.need_sync = True

    def split_by_upper_case_letters(self, s):
        return [x for x in re.split(r'(?=[A-Z])', s) if x]

    def search_sample(self):
        founded_tag_ids = {}
        queries = [*self.query, *self.split_by_upper_case_letters(self.audio_type.name), self.audio_type.name]
        augmented_query = {t: tuple({t, t.lower(), t.upper(), t.lower().capitalize(), " ".join([x.lower().capitalize() for x in t.split()]), "-".join([x.lower().capitalize() for x in t.split("-")])}) for t in queries}
        for tag, aug_tags in augmented_query.items():
            if len(aug_tags) == 1:
                tag_query = f"SELECT id, name FROM tags WHERE name={aug_tags[0]};"
            else:
                tag_query = f"SELECT id, name FROM tags WHERE name in {aug_tags};"
            output = self.search_db(tag_query)
            if len(output) != 0:
                founded_tag_ids.update({str(t[0]): t[1] for t in output})
            else:
                self.log(f"Tag '{tag}' was not found!")
        maxlen = self.maxlen.to_seconds(self.project.bpm) * 1000 if self.maxlen else 9999999999
        minlen = self.minlen.to_seconds(self.project.bpm) * 1000 if self.minlen else 0
        query = (
            f"SELECT ft.file_id, ft.file_path, ft.file_name, fs.path "
            f"FROM ("
            f"  SELECT source_id, file_id, file_path, file_name "
            f"  FROM file_tags "
            f"  WHERE tag_id IN ({",".join(founded_tag_ids.keys())}) "
            f"  GROUP BY file_path, file_name "
            f"  HAVING COUNT(DISTINCT tag_id) = {len(founded_tag_ids)}) "
            f"AS ft "
            f"JOIN sample_meta AS sm ON ft.file_id = sm.id "
            f"JOIN filesources AS fs ON ft.source_id = fs.id "
            f"WHERE sm.loop = {int(self.loop)} "
            f"AND sm.length BETWEEN {minlen} AND {maxlen} "
            f"AND sm.tempo BETWEEN {self.bpm_range[0]} AND {self.bpm_range[1]}"
        )
        output = self.search_db(query)
        return output

    def search_sample_backup(self):
        queries = [*self.query, *self.split_by_upper_case_letters(self.audio_type.name), self.audio_type.name]
        maxlen = self.maxlen.to_seconds(self.project.bpm) * 1000 if self.maxlen else 9999999999
        minlen = self.minlen.to_seconds(self.project.bpm) * 1000 if self.minlen else 0
        keyword_conditions = " + ".join([f"CASE WHEN name LIKE '%{kw}%' THEN 1 ELSE 0 END" for kw in queries])
        where_conditions = " OR ".join([f"name LIKE '%{kw}%'" for kw in queries])

        query = f"""
                SELECT ft.id, ft.path, ft.name, fs.path 
                FROM (
                    SELECT id, path, name, source_id,
                           ({keyword_conditions}) AS relevance 
                    FROM files 
                    WHERE {where_conditions} 
                    ORDER BY relevance DESC) 
                AS ft 
                JOIN sample_meta AS sm ON ft.id=sm.id 
                JOIN filesources AS fs ON ft.source_id=fs.id 
                WHERE sm.loop={int(self.loop)} 
                AND sm.length BETWEEN {minlen} AND {maxlen} 
                AND sm.tempo BETWEEN {self.bpm_range[0]} AND {self.bpm_range[1]};
                """
        output = self.search_db(query)
        return output

    def search_sample_backup2(self):
        queries = [*self.query, *self.split_by_upper_case_letters(self.audio_type.name), self.audio_type.name]
        maxlen = self.maxlen.to_seconds(self.project.bpm) * 1000 if self.maxlen else 9999999999
        minlen = self.minlen.to_seconds(self.project.bpm) * 1000 if self.minlen else 0
        keyword_conditions = " + ".join([f"CASE WHEN name LIKE '%{kw}%' THEN 1 ELSE 0 END" for kw in queries])
        where_conditions = " OR ".join([f"name LIKE '%{kw}%'" for kw in queries])
        query = f"""
                SELECT ft.id, ft.path, ft.name, fs.path 
                FROM (
                    SELECT id, path, name, source_id,
                           ({keyword_conditions}) AS relevance 
                    FROM files 
                    WHERE {where_conditions} 
                    ORDER BY relevance DESC) 
                AS ft 
                JOIN sample_meta AS sm ON ft.id=sm.id 
                JOIN filesources AS fs ON ft.source_id=fs.id 
                WHERE sm.loop={int(self.loop)} 
                AND sm.length BETWEEN {minlen} AND {maxlen};
                """
        output = self.search_db(query)
        return output

    def _add_to_candidates(self, candidates):
        for sample in candidates:
            sample_path = f"{sample[3]}/{sample[2]}"f"{sample[3]}/{sample[2]}" if sample[1] == '.' else f"{sample[3]}/{sample[1]}/{sample[2]}"
            self.sample_candidates[sample_path] = (sample[0], sample_path)

    def load_audio_file(self):
        assert self.file_path is not None
        pathseg = self.file_path.split("/")
        folder, filename = "/".join(pathseg[:-1]), pathseg[-1]
        query = ("SELECT f.id "
                 "FROM files AS f "
                 "JOIN folders AS fd "
                 "ON f.source_id=fd.source_id "
                 "AND f.folder_id=fd.id "
                 "WHERE fd.path=? "
                 "AND f.name=?;")
        parameters = [folder, filename]
        output = self.search_db(query, parameters)
        if len(output) == 0:
            audio_len = librosa.get_duration(filename=self.file_path) * 1000
            query = "SELECT f.id FROM files AS f WHERE f.name=? AND f.length BETWEEN ? AND ?;"
            parameters = [filename, audio_len-0.001, audio_len+0.001]
            output = self.search_db(query, parameters)
            if len(output) == 0:
                raise ValueError(f"{self.__repr__()} - {self.file_path} was not found in the analyzed audio library, make sure it is processed by ADSR sample manager before")
        self.sample_candidates[self.file_path] = (output[0][0], self.file_path)

    def load_sample_data(self, sample: tuple):
        query = f"SELECT tempo,key,length,loop FROM sample_meta WHERE id={sample[0]};"
        output = self.search_db(query)
        if len(output) == 0:
            self.length = BarStepTick()
            self.log(f"Something went wrong when getting sample info.")
        else:
            output = output[0]
            self.current_sample = OrderedDotdict({"id": sample[0],
                                                  'name': sample[1].split('/')[-1],
                                                  "path": sample[1],
                                                  "tempo": round(output[0]),
                                                  "key": KeyMode(ADSRSampleKeyMapping[output[1]]),
                                                  "length": output[2] / 1000,
                                                  "bst_length": BarStepTick.sec2bst(output[2] / 1000, round(output[0])),
                                                  "loop": bool(output[3])})
            # print(self.current_sample)
            if not self.transpose_steps_set:
                self.transpose_steps = self.current_sample.key - self.key_mode
            if self.fit_tempo:
                self.length = self.current_sample.bst_length
            else:
                self.length = BarStepTick.sec2bst(self.current_sample.length, self.project.bpm)
