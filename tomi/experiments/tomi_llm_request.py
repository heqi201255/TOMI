import requests
import ollama
from tomi.config import LLM_MODEL, API_KEY, API_BASE, MODEL_PROVIDER
from tomi import printer, GrooveSpeed, TrackType, SectionType, MIDIType, AudioType, SongGenre, PROGRAM_DIR
from .experiment_section_patterns import structure_and_sections_dict
import json
import openai
from datetime import datetime
from .llm_prompts import *
from json_repair import repair_json
from collections import defaultdict
import os


class TOMILLMRequest:
    def __init__(self, model_name: str = None, section_pattern: str = None):
        if model_name is None:
            model_name = LLM_MODEL
        self.model = model_name
        if MODEL_PROVIDER == 'OpenAI':
            openai.base_url = API_BASE
            openai.api_key = API_KEY
        self.chat_history = []
        self.section_pattern = section_pattern

    def append_chat(self, chat: dict):
        printer.print(printer.Bblue(printer.bright_white(chat['role'].capitalize())))
        printer.print(printer.bright_white(chat['content']))
        self.chat_history.append(chat)

    def _update_defined_section_info(self, reply: dict):
        if self.section_pattern is not None:
            reply.update(structure_and_sections_dict(self.section_pattern))
        return reply

    def _get_system_prompt(self):
        component_types = dict(section_type=SectionType.option_list(),
                               track_type=[x for x in TrackType.option_list()],
                               midi_type=[x for x in MIDIType.option_list() if x not in ('Kick', 'ClapSnare', 'Hihat', 'Drummer')],
                               groove_speed=GrooveSpeed.option_list(),
                               audio_type=AudioType.option_list())
        return TOMI_SYSTEM_PROMPT_FSTRING.format(**component_types)

    def _get_user_prompt(self, song_genre: SongGenre, user_prompt: str = None):
        return TOMI_USER_PROMPT_FULL_SONG_GENERATION_FSTRING.format(genre=song_genre.name) if user_prompt is None else user_prompt

    def generate_song_blocks(self, song_genre: SongGenre, user_prompt: str = None, output_save_path: str = None, save_file_name: str = None, stream: bool = True):
        if output_save_path is None:
            output_save_path = os.path.join(PROGRAM_DIR, f"model_outputs/{self.__class__.__name__}")
        os.makedirs(output_save_path, exist_ok=True)
        system_prompt = self._get_system_prompt()
        user_prompt = self._get_user_prompt(song_genre, user_prompt)
        if self.model.startswith(('o1', 'o3')):
            self.append_chat({'role': 'user', 'content': "\n".join([system_prompt, user_prompt])})
        else:
            self.append_chat({"role": "system", "content": system_prompt})
            self.append_chat({"role": "user", "content": user_prompt})
        reply = self.get_reply(stream=stream)
        reply = self.get_reply_json_format(reply)
        reply = self._update_defined_section_info(reply)
        incorrect_items, fix_prompt = self.validate_output(reply)
        while any(len(v) > 0 for v in incorrect_items.values()):
            self.append_chat({"role": "user", "content": fix_prompt})
            reply = self.get_reply(stream=stream)
            reply = self.get_reply_json_format(reply)
            reply = self._update_defined_section_info(reply)
            incorrect_items, fix_prompt = self.validate_output(reply)
        try:
            if save_file_name is None:
                save_file_name = f"{self.model}_{str(datetime.now())}.json".replace(":", "-").replace(" ", "_")
            with open(os.path.join(output_save_path, save_file_name), 'w') as f:
                json.dump(reply, f, indent=4)
        except:
            pass
        return reply

    def delete_last_chat(self):
        self.chat_history = self.chat_history[:-1]

    def get_reply_json_format(self, reply):
        try:
            last_bracket = reply.rfind('}')
            reply = reply[reply.index('{'):last_bracket + 1]
            reply = repair_json(reply)
            if not reply:
                raise ValueError()
            reply = json.loads(reply)
            return reply
        except:
            prompt = "It looks like your response is not in JSON format or your JSON format got some syntax error or you just said something else, please fix the issue and send me the updated version in JSON format ONLY, no extra texts."
            self.append_chat({"role": "user", "content": prompt})
            reply = self.get_reply()
            return self.get_reply_json_format(reply)

    @staticmethod
    def validate_output(response: dict):
        section_type=SectionType.option_list()
        track_type=[x for x in TrackType.option_list()]
        midi_type=[x for x in MIDIType.option_list() if x not in ('Kick', 'ClapSnare', 'Hihat', 'Drummer')]
        groove_speed=GrooveSpeed.option_list()
        audio_type=AudioType.option_list()
        wrong_elements = defaultdict(list)
        sec_names = []
        track_names = []
        clip_names = []
        clip_types = {}
        track_types = {}
        transformation_names = []
        prompt = []
        structure_available = section_available = track_available = clip_available = transformation_available = link_available = True
        if 'Structure' not in response:
            wrong_elements['Structure'].append([-1, 'NoStructureKey', 'NoStructureKey'])
            prompt.append(f"There is no 'Structure' key in your response, which is a requirement")
            structure_available = False
        if 'Sections' not in response:
            wrong_elements['Sections'].append([-1, 'NoSectionsKey', 'NoSectionsKey'])
            prompt.append(f"There is no 'Sections' key in your response, which is a requirement")
            section_available = False
        if 'Tracks' not in response:
            wrong_elements['Tracks'].append([-1, 'NoTracksKey', 'NoTracksKey'])
            prompt.append(f"There is no 'Tracks' key in your response, which is a requirement")
            track_available = False
        if 'Clips' not in response:
            wrong_elements['Clips'].append([-1, 'NoClipsKey', 'NoClipsKey'])
            prompt.append(f"There is no 'Clips' key in your response, which is a requirement")
            clip_available = False
        if 'Transformations' not in response:
            wrong_elements['Transformations'].append([-1, 'NoTransformationsKey', 'NoTransformationsKey'])
            prompt.append(f"There is no 'Transformations' key in your response, which is a requirement")
            transformation_available = False
        if 'Links' not in response:
            wrong_elements['Links'].append([-1, 'NoLinksKey', 'NoLinksKey'])
            prompt.append(f"There is no 'Links' key in your response, which is a requirement")
            link_available = False
        if section_available:
            for i, sec in enumerate(response['Sections']):
                sec_names.append(sec[0])
                if len(sec) == 3 and sec[1] not in section_type:
                    wrong_elements['Sections'].append([i, sec[1], 'InvalidSectionType'])
                    prompt.append(f"In Sections, '{sec[0]}' has an invalid section_type: '{sec[1]}'")
        if structure_available:
            if section_available:
                for i, section in enumerate(response['Structure']):
                    if section not in sec_names:
                        wrong_elements['Structure'].append([i, section, 'InvalidSectionName'])
                        prompt.append(f"In Structure, '{section}' is not converted to a Section node in Sections part.")
        if track_available:
            for i, tr in enumerate(response['Tracks']):
                track_names.append(tr[0])
                if tr[1] not in track_type:
                    wrong_elements['Tracks'].append([i, tr[1], 'InvalidTrackType'])
                    prompt.append(f"In Tracks, '{tr[0]}' has an invalid track_type: '{tr[1]}'")
                else:
                    track_types[tr[0]] = tr[1]
        if clip_available:
            for i, clip in enumerate(response['Clips']):
                clip_names.append(clip[0])
                if clip[1] == "Midi":
                    if clip[2] not in midi_type:
                        wrong_elements['Clips'].append([i, clip[2], 'InvalidMIDIType'])
                        prompt.append(f"In Clips, '{clip[0]}' has an invalid midi_type: '{clip[2]}'")
                    if clip[4] not in groove_speed:
                        wrong_elements['Clips'].append([i, clip[4], 'InvalidGrooveSpeed'])
                        prompt.append(f"In Clips, '{clip[0]}' has an invalid groove_speed: '{clip[4]}'")
                    clip_types[clip[0]] = clip[1]
                elif clip[1] == "Audio":
                    if clip[2] not in audio_type:
                        wrong_elements['Clips'].append([i, clip[2], 'InvalidAudioType'])
                        prompt.append(f"In Clips, '{clip[0]}' has an invalid audio_type: '{clip[2]}'")
                    clip_types[clip[0]] = clip[1]
                else:
                    wrong_elements['Clips'].append([i, clip[1], 'InvalidClipType'])
                    prompt.append(f"In Clips, '{clip[0]}' has an invalid clip_type: '{clip[1]}'")
        if transformation_available:
            for i, transformation in enumerate(response['Transformations']):
                attrs = []
                for x in transformation:
                    if isinstance(x, tuple):
                        attrs.extend(x)
                    else:
                        attrs.append(x)
                transformation_names.append(transformation[0])
                if attrs[1] not in ['general_transform', 'drum_transform', 'fx_transform', 'fill_transform']:
                    wrong_elements['Transformations'].append([i, attrs[1], 'InvalidTransformation'])
                    prompt.append(f"In Transformations, '{attrs[0]}' has an invalid transformation_type: '{attrs[1]}'")
        if link_available:
            check_sec_names = list(set(sec_names))
            for i, link in enumerate(response['Links']):
                nodes = link.split('->')
                if len(nodes) != 4:
                    wrong_elements['Links'].append([i, nodes, 'InvalidLink'])
                    prompt.append(f"In Links, the link '{link}' is invalid because it's not having 4 elements (section->transformation->clip->track).")
                    continue
                s, p, c, t = nodes
                if s not in sec_names:
                    wrong_elements['Links'].append([i, s, 'SectionNotExist'])
                    prompt.append(f"In Links, node '{s}' is not created or it's not a section node")
                else:
                    try:
                        check_sec_names.remove(s)
                    except ValueError:
                        pass
                if p not in transformation_names:
                    wrong_elements['Links'].append([i, p, 'TransformationNotExist'])
                    prompt.append(f"In Links, node '{p}' is not created or it's not a transformation node")
                if c not in clip_names:
                    wrong_elements['Links'].append([i, c, 'ClipNotExist'])
                    prompt.append(f"In Links, node '{c}' is not created or it's not a clip node")
                if t not in track_names:
                    wrong_elements['Links'].append([i, t, 'TrackNotExist'])
                    prompt.append(f"In Links, node '{t}' is not created or it's not a track node")
                if c in clip_names and t in track_names:
                    if clip_types[c] != track_types[t]:
                        wrong_elements['Links'].append([i, t, 'UnMatchedClipTrack'])
                        prompt.append(f"In Links, clip node '{c}' with clip_type '{clip_types[c]}' cannot be linked to track '{t}' with track_type '{track_types[t]}', consider linking this clip to other tracks of '{clip_types[c]}' track_type or creating a new track.")
            if check_sec_names:
                for i, sec in enumerate(check_sec_names):
                    wrong_elements['Sections'].append([sec_names.index(sec), sec, 'SectionNotUsed'])
                    prompt.append(f"In Sections, node '{sec}' is not used in any of the links, this section will be empty in the composition, which is not allowed.")
        if clip_available:
            for i, clip in enumerate(response['Clips']):
                if clip[1] == 'Midi' and clip[5] is not None:
                    if clip[5] not in clip_names:
                        wrong_elements['Clips'].append([i, clip[5], 'InvalidDependentNode'])
                        prompt.append(f"In Clips, '{clip[0]}' has an dependent_node '{clip[5]}' that is not created in your generation")
                    else:
                        if clip_types[clip[5]] != "Midi":
                            wrong_elements['Clips'].append([i, clip[5], 'InvalidDependentNode'])
                            prompt.append(f"In Clips, '{clip[0]}' has an dependent_node '{clip[5]}' that is not a MIDI Clip")
        prompt = ("Thank you for your creation, however, there are some errors in your reply:\n" + ";\n".join(prompt) +
                  f"\n\nNote:\n"
                  f"ALL node names MUST be UNIQUE.\n"
                  f"All section nodes must be appeared in the links for at least once.\n"
                  f"section_type (string): must be one of {section_type}.\n"
                  f"track_type (string): must be one of {track_type}.\n"
                  f"midi_type (string): must be one of {midi_type}.\n"
                  f"midi_groove_speed (string): must be one of {groove_speed}.\n"
                  f"dependent_midi (string/null): can be null or another midi node's name that you have created.\n"
                  f"audio_type (string): must be one of {audio_type}.\n"
                  f"transformation_type (string): must be one of ['general_transform', 'drum_transform', 'fx_transform', 'fill_transform'].\n"
                  f"Links should only include names of the nodes that's being created in your generation.\n"
                  "Please fix these errors and send me the updated version in JSON format.")
        return wrong_elements, prompt

    def get_reply(self, stream: bool = True):
        if MODEL_PROVIDER == 'OpenAI':
            response = None
            while response is None:
                try:
                    response = openai.chat.completions.create(model=self.model, stream=stream, messages=self.chat_history)
                    if not stream:
                        response = response.choices[0].message['content']
                    else:
                        printer.print(printer.Bblue(printer.bright_white('assistant'.capitalize())))
                        full_response = ''
                        for chunk in response:
                            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                full_response += content
                                printer.print(content, end='')
                        printer.print()
                        response = full_response
                except Exception as e:
                    printer.print(f"Got error: '{e}', trying again ...")
        elif MODEL_PROVIDER == 'Claude':
            response = None
            while response is None:
                try:
                    url = API_BASE
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": API_KEY
                    }
                    data = {
                        "model": LLM_MODEL,
                        "messages": self.chat_history
                    }
                    response = requests.post(url=url, headers=headers, json=data).json()['choices'][0]['message']['content']
                except Exception as e:
                    printer.print(f"Got error: '{e}', trying again ...")
        else:
            # Ollama is not tested.
            response = ollama.chat(model=self.model, messages=self.chat_history)
            response = response['message']['content']
        if not stream:
            self.append_chat({'role': 'assistant', 'content': response})
        else:
            self.chat_history.append({'role': 'assistant', 'content': response})
        return response


if __name__ == '__main__':
    s = TOMILLMRequest().generate_song_blocks(SongGenre.Pop)
    print(s)