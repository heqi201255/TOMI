from tomi import (Project, TOMIEngine, SongGenre, TrackType, MIDIType,
                   AudioType, NodeFactory, Key, BarStepTick, SectionNode, SectionType,
                  DEFAULT_INSTRUMENT_PLUGIN, DEFAULT_INSTRUMENT_PRESET)
import numpy as np
from .tomi_llm_request import TOMILLMRequest
from .llm_prompts import *
from .experiment_section_patterns import pattern_total_bars, structure_and_sections_dict
import os.path
import json


class StandaloneLLMRequest(TOMILLMRequest):
    """ This class is used in 'standalone_llm' ablation only. """
    def __init__(self, model: str = None, section_pattern: str = None):
        super(StandaloneLLMRequest, self).__init__(model, section_pattern)
        assert self.section_pattern is not None, "'section_pattern' is required."

    def _update_defined_section_info(self, reply: dict):
        reply.update(structure_and_sections_dict(self.section_pattern))
        reply['TotalBars'] = pattern_total_bars(self.section_pattern)
        return reply

    def _get_system_prompt(self):
        component_types = dict(section_type=SectionType.option_list(),
                               midi_type=[x for x in MIDIType.option_list() if x not in ('Kick', 'ClapSnare', 'Hihat', 'Drummer')],
                               audio_type=AudioType.option_list())
        return STANDALONE_LLM_SYSTEM_PROMPT_GIVEN_STRUCTURE.format(**component_types)

    def _get_user_prompt(self, song_genre: SongGenre, user_prompt: str = None):
        return STANDALONE_LLM_USER_PROMPT_FULL_SONG_GENERATION.format(genre=song_genre.name) if user_prompt is None else user_prompt

    @staticmethod
    def validate_output(response: dict):
        midi_type = MIDIType.option_list()
        audio_type = AudioType.option_list()
        wrong_elements = {'Tracks': [], 'Clips': []}
        track_names = []
        clip_names = []
        prompt = []
        track_name_and_type = {}
        track_available = clip_available = True
        if 'Tracks' not in response:
            wrong_elements['Tracks'].append([-1, 'NoTracksKey', 'NoTracksKey'])
            prompt.append(f"There is no 'Tracks' key in your response, which is a requirement")
            track_available = False
        if 'Clips' not in response:
            wrong_elements['Clips'].append([-1, 'NoClipsKey', 'NoClipsKey'])
            prompt.append(f"There is no 'Clips' key in your response, which is a requirement")
            clip_available = False
        if track_available:
            for i, tr in enumerate(response['Tracks']):
                track_names.append(tr[0])
                track_name_and_type[tr[0]] = tr[1]
        if clip_available:
            for i, clip in enumerate(response['Clips']):
                clip_names.append(clip[0])
                if not isinstance(clip[2], list):
                    wrong_elements['Clips'].append([i, clip[2], 'InvalidPlaybackTimes'])
                    prompt.append(f"In Clips, '{clip[0]}' has an invalid playback_times: '{clip[2]}', this should be a list of tuples.")
                    continue
                else:
                    error = False
                    for time in clip[2]:
                        if len(time) != 2:
                            wrong_elements['Clips'].append([i, clip[2], 'InvalidPlaybackTimes'])
                            prompt.append(f"In Clips, '{clip[0]}'s playback_times got an invalid time tuple: {time}, it should be 2 unsigned integers [bar, step].")
                            error = True
                        elif not isinstance(time[0], int) or not isinstance(time[1], int):
                            wrong_elements['Clips'].append([i, clip[2], 'InvalidPlaybackTimes'])
                            prompt.append(f"In Clips, '{clip[0]}'s playback_times got an invalid time tuple: {time}, it should be 2 unsigned integers [bar, step].")
                            error = True
                        elif time[0] < 0 or time[1] < 0:
                            wrong_elements['Clips'].append([i, clip[2], 'InvalidPlaybackTimes'])
                            prompt.append(f"In Clips, '{clip[0]}'s playback_times got an invalid time tuple: {time}, it should be 2 unsigned integers [bar, step].")
                            error = True
                        elif time[1] > 15:
                            wrong_elements['Clips'].append([i, clip[2], 'InvalidPlaybackTimes'])
                            prompt.append(f"In Clips, '{clip[0]}'s playback_times got an invalid time tuple: {time} with step '{time[1]}', remember a bar contains 16 steps, this parameter is zero-based so it should be range from 0 to 15 inclusively.")
                            error = True
                    if error:
                        continue
                if clip[3] not in track_names:
                    wrong_elements['Clips'].append([i, clip[3], 'InvalidTrackLocation'])
                    prompt.append(f"In Clips, '{clip[0]}' has an invalid track_location: '{clip[3]}', this track name does not exist.")
                elif clip[1] == "MIDI":
                    if track_name_and_type[clip[3]] != "MIDI":
                        wrong_elements['Clips'].append([i, clip[3], 'UnMatchedClipTrack'])
                        prompt.append(f"In Clips, MIDI clip '{clip[0]}' cannot be added to track '{clip[3]}' with track_type '{track_name_and_type[clip[3]]}', consider add this clip to other tracks of 'MIDI' track_type or creating a new track.")
                    if clip[4] not in midi_type:
                        wrong_elements['Clips'].append([i, clip[4], 'InvalidMIDIType'])
                        prompt.append(f"In Clips, '{clip[0]}' has an invalid midi_type: '{clip[4]}'")
                elif clip[1] == 'Audio':
                    if track_name_and_type[clip[3]] != "Audio":
                        wrong_elements['Clips'].append([i, clip[3], 'UnMatchedClipTrack'])
                        prompt.append(f"In Clips, Audio clip '{clip[0]}' cannot be added to track '{clip[3]}' with track_type '{track_name_and_type[clip[3]]}', consider add this clip to other tracks of 'Audio' track_type or creating a new track.")
                    if clip[4] not in audio_type:
                        wrong_elements['Clips'].append([i, clip[4], 'InvalidAudioType'])
                        prompt.append(f"In Clips, '{clip[0]}' has an invalid audio_type: '{clip[4]}'")
        prompt = ("Thank you for your creation, however, there are some errors in your reply:\n" + ";\n".join(prompt) +
                  f"\n\nNote:\n"
                  f"ALL element names MUST be UNIQUE.\n"
                  f"midi_type (string): must be one of {midi_type}.\n"
                  f"audio_type (string): must be one of {audio_type}.\n"
                  "Please fix these errors and send me the updated version in JSON format.")
        return wrong_elements, prompt


class StandaloneLLMSong:
    def __init__(self,
                 song_name: str = "standalone_llm_song",
                 song_genre: SongGenre = SongGenre.Pop,
                 song_blocks: dict | str = None,
                 key: Key = Key.C):
        self.song_genre = song_genre
        self.project = Project(song_name, key=key, genre=self.song_genre)
        self.engine = TOMIEngine(self.project)
        if song_blocks is None:
            self.song_blocks = StandaloneLLMRequest(section_pattern='pattern1').generate_song_blocks(self.song_genre)
        elif isinstance(song_blocks, str):
            if not os.path.exists(song_blocks):
                raise FileNotFoundError(song_blocks)
            self.song_blocks = json.load(open(song_blocks))
        elif isinstance(song_blocks, dict):
            self.song_blocks = song_blocks
        else:
            raise TypeError("song_blocks must be a str or dict")
        self.section_pattern = self.song_blocks['Sections']
        self.total_bars = self.song_blocks['TotalBars']
        self.sections = [["Intro", "Intro", self.total_bars]]
        self.tracks = self.song_blocks['Tracks']
        self.clips = self.song_blocks['Clips']
        self.factory = NodeFactory(self.project)
        self.gui = None
        self.track_nodes = {}
        self.section_nodes = {}
        self.clip_nodes = {}

    def cast_sections(self):
        sections = SectionNode.get_arrangement_nodes(self.project, self.sections)
        for i in range(len(self.sections)):
            self.section_nodes[self.sections[i][0]] = sections[i]
        self.sections = sections

    def cast_track(self, track_attrs: tuple):
        """
        @param track_attrs:
        @return:
        """
        track_name, track_type = track_attrs
        track_type = TrackType.Audio if track_type == "Audio" else TrackType.Midi
        if track_type == TrackType.Midi:
            self.track_nodes[track_name] = self.factory.track(track_type, plugin_name=DEFAULT_INSTRUMENT_PLUGIN, plugin_preset=DEFAULT_INSTRUMENT_PRESET, node_name=track_name)
        else:
            self.track_nodes[track_name] = self.factory.track(track_type, node_name=track_name)

    def cast_clips(self, clip_attrs: tuple):
        """
        @param clip_attrs:
        @return:
        """
        if clip_attrs[1] == 'MIDI':
            if len(clip_attrs) == 8:
                clip_name, _, section_location, specific_time, track_location, midi_type, midi_length, root_progression = clip_attrs
            else:
                clip_name, _, playback_times, track_location, midi_type, midi_length, root_progression = clip_attrs
            midi_type = MIDIType.get_object_by_name(midi_type)
            self.clip_nodes[clip_name] = self.factory.midi(midi_type, BarStepTick(midi_length), root_progression=root_progression, node_name=clip_name)
        else:
            if len(clip_attrs) == 8:
                clip_name, _, section_location, specific_time, track_location, audio_type, query, loop = clip_attrs
            else:
                clip_name, _, playback_times, track_location, audio_type, query, loop = clip_attrs
            audio_type = AudioType.get_object_by_name(audio_type)
            fit_tempo = fit_key = loop
            self.clip_nodes[clip_name] = self.factory.audio(audio_type, query=query, loop=loop, fit_key=fit_key, fit_tempo=fit_tempo, node_name=clip_name)
        if len(clip_attrs) == 8:
            action_sequence = np.zeros(self.section_nodes[section_location].length.bar * 16, dtype=np.int8)
            specific_time = 1 if specific_time is None else specific_time
            action_sequence[int((specific_time-1) * 16):] = 1
            action_sequence[int((specific_time-1) * 16)] = 2
            section = self.section_nodes[section_location]
        else:
            action_sequence = np.ones(self.total_bars * 16, dtype=np.int8)
            for pt in playback_times:
                steps = BarStepTick(pt[0],pt[1]).to_steps()
                try:
                    action_sequence[steps] = 2
                except IndexError:
                    continue
            mask = np.nonzero(action_sequence==2)[0]
            if mask.size==0:
                action_sequence[:] = 0
            else:
                action_sequence[:mask[0]] = 0
            section = list(self.section_nodes.values())[0]
        transformation = self.factory.general_transform(action_sequence=action_sequence, loop=False, node_name=f'{clip_name}_transformation')
        return  section >> transformation >> self.clip_nodes[clip_name] >> self.track_nodes[track_location]

    def gen(self, stream_output: bool = False, open_editor: bool = True):
        self.cast_sections()
        self.project.set_arrangement(self.sections)
        for t in self.tracks:
            self.cast_track(t)
        links = [self.cast_clips(c) for c in self.clips]
        for nl in links:
            self.project.node_graph.add(nl)
        self.project.remove_unused_nodes()
        self.project.node_graph.print_link("Arrangement Structure")
        self.engine.run_reaper(stream_output=stream_output, section_pattern=self.section_pattern)
        if open_editor:
            from tomi.editor import TOMIEditor
            self.gui = TOMIEditor(self.engine)
            self.gui.run()


if __name__ == '__main__':
    llm_gen = StandaloneLLMSong(song_blocks=None)
    llm_gen.gen(False, open_editor=True)
