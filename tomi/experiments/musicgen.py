# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Main model for using MusicGen. This will combine all the required components
and provide easy access to the generation API.
"""

import typing as tp
import torch
from audiocraft.models import MusicGen
from audiocraft.modules.conditioners import ConditioningAttributes, WavCondition
import math
from typing import Union


class BarStepTick:
    """
    This clss is used as the DAW time unit in many other modules. From its name you can tell it means the bar, step, and
    tick typically used in modern DAWs. In MIDI standards, it only uses Ticks for timing, you'll need to convert it
    to other timing units with stuff like PPQ. Here we use BarStepTick as the only time unit across Sprin to simplify
    the understanding of music timing.
    """
    def __init__(self, bar: Union[int, list] = 0, step: int = 0, tick: int = 0):
        """
        The creation of a BarStepTick instance, it will check all the params whether they are in valid ranges.
        :param bar: should be a non-negative integer. Each bar has 16 steps.
        :param step: should be between 0 and 15 inclusively. Each step has 24 ticks.
        :param tick: should be between 0 and 23 inclusively.
        """
        if type(bar) is list:
            if len(bar) == 1:
                bbar = int(bar[0])
                bstep = 0
                btick = 0
            elif len(bar) == 2:
                bbar = int(bar[0])
                bstep = int(bar[1])
                btick = 0
            elif len(bar) == 3:
                bbar = int(bar[0])
                bstep = int(bar[1])
                btick = int(bar[2])
            else:
                raise ValueError("Wrong format")
        else:
            bbar = int(bar)
            bstep = int(step)
            btick = int(tick)
        if bbar < 0:
            raise ValueError("'Bar' must be a non-negative integer")
        elif bstep < 0 or bstep > 15:
            raise ValueError("'Step' must be between 0 and 15 inclusively")
        elif btick < 0 or btick > 23:
            raise ValueError("'Tick' must be between 0 and 23 inclusively")
        self.bar = bbar
        self.step = bstep
        self.tick = btick

    def __hash__(self):
        return hash(self.to_steps())

    def get_bar_step_tick(self):
        return self.bar, self.step, self.tick

    def append(self, bst: Union['BarStepTick', list[int, int], list[int, int, int]], inplace=False):
        if type(bst) is BarStepTick:
            bar = self.bar + bst.bar
            step = self.step + bst.step
            tick = self.tick + bst.tick
            bar += int((step + int(tick / 24)) / 16)
            step = (step + int(tick / 24)) % 16
            tick = tick % 24
        else:
            if len(bst) == 2:
                bar = self.bar + bst[0]
                step = self.step + bst[1]
                bar += int(step / 16)
                step = step % 16
                tick = self.tick
            elif len(bst) == 3:
                bar = self.bar + bst[0]
                step = self.step + bst[1]
                tick = self.tick + bst[2]
                bar += int((step + int(tick / 24)) / 16)
                step = (step + int(tick / 24)) % 16
                tick = tick % 24
            else:
                raise ValueError("'bst' should be a 'BarStepTick' instance or a list of two integers or three integers.")
        if inplace:
            self.bar = bar
            self.step = step
            self.tick = tick
        return bar, step, tick

    def __lt__(self, other):
        if self.to_ticks() < other.to_ticks():
            return True
        return False

    def __eq__(self, other):
        if self.to_ticks() == other.to_ticks():
            return True
        return False

    def __le__(self, other):
        if self.to_ticks() <= other.to_ticks():
            return True
        return False

    def __add__(self, other):
        return BarStepTick(*self.append(other))

    def __sub__(self, other):
        if other.bar < self.bar:
            if other.step <= self.step:
                if other.tick <= self.tick:
                    bar = self.bar - other.bar
                    step = self.step - other.step
                    tick = self.tick - other.tick
                else:  # [1,1,15] [0, 1, 20]
                    tick = 24 - (other.tick - self.tick)
                    step = self.step - other.step - 1
                    if step < 0:
                        step += 16
                        bar = self.bar - other.bar - 1
                    else:
                        bar = self.bar - other.bar
            else:
                if other.tick <= self.tick:
                    bar = self.bar - other.bar - 1
                    step = 16 - (other.step - self.step)
                    tick = self.tick - other.tick
                else:  # [2,1,15] [0,15,20]
                    tick = 24 - (other.tick - self.tick)
                    step = 16 - (other.step - self.step) - 1
                    bar = self.bar - other.bar - 1
        elif other.bar == self.bar:
            bar = 0
            if other.step <= self.step:
                if other.tick <= self.tick:
                    step = self.step - other.step
                    tick = self.tick - other.tick
                else:
                    tick = 24 - (other.tick - self.tick)
                    step = self.step - other.step - 1
                    if step < 0:
                        raise ValueError("BarStepTick cannot subtract another BarStepTick instance that is longer.")
            else:
                raise ValueError("BarStepTick cannot subtract another BarStepTick instance that is longer.")
        else:
            raise ValueError("BarStepTick cannot subtract another BarStepTick instance that has more length.")
        return BarStepTick(bar, step, tick)

    def is_empty(self):
        return self.bar == 0 and self.step == 0 and self.tick == 0

    def to_beats(self):
        beat = self.bar * 4
        beat += self.step / 4
        beat += self.tick / 96
        return round(beat, 2)

    def to_bars(self):
        return round(self.bar + (self.step / 16) + (self.tick / 384), 2)

    def to_seconds(self, bpm):
        return BarStepTick.bst2sec(self, bpm=bpm)

    def to_steps(self):
        return self.bar * 16 + self.step + round(self.tick / 24)

    def to_ticks(self):
        return self.bar * 384 + self.step * 24 + self.tick

    @staticmethod
    def str2bst(s: str) -> 'BarStepTick':
        try:
            bind = s.index("b")
        except:
            bind = 0
        try:
            sind = s.index("s")
        except:
            sind = 0
        try:
            tind = s.index("t")
        except:
            tind = 0
        bar = step = tick = 0
        if bind:
            bar = int(s[:bind])
        if sind:
            if bind:
                step = int(s[bind + 1:sind])
            else:
                step = int(s[:sind])
        if tind:
            if sind:
                tick = int(s[sind + 1:tind])
            elif bind:
                tick = int(s[bind + 1:tind])
            else:
                tick = int(s[:tind])
        return BarStepTick(bar, step, tick)

    def __str__(self):
        s = []
        if self.bar != 0:
            s.append(str(self.bar) + "b")
        if self.step != 0:
            s.append(str(self.step) + "s")
        if self.tick != 0:
            s.append(str(self.tick) + "t")
        if not s:
            s = ["0b"]
        return "".join(s)

    def __repr__(self):
        return self.__str__()

    def get_bar(self):
        return self.bar

    def get_step(self):
        return self.step

    def get_tick(self):
        return self.tick

    def set_bar(self, bar: int):
        bar = int(bar)
        if bar < 0:
            raise ValueError("'Bar' must be a non-negative integer")
        self.bar = bar

    def set_step(self, step: int):
        step = int(step)
        if step < 0 or step > 15:
            raise ValueError("'Step' must be between 0 and 15 inclusively")
        self.step = step

    def set_tick(self, tick: int):
        tick = int(tick)
        if tick < 0 or tick > 23:
            raise ValueError("'Tick' must be between 0 and 23 inclusively")
        self.tick = tick

    @staticmethod
    def sec2bst(sec: float, bpm: float = 120) -> 'BarStepTick':
        '''
        BST is the Bar-Step-Tick time signature, but in pretty_midi we need the exact start time and end time in seconds to
        draw a note, so we need this function to convert the BST to seconds given the bpm, bar, and step. Tick is not used
        here because it is too small and we really don't need it.
        '''
        total_ticks = math.ceil(sec / (60 / (bpm * 96)))
        ticks = total_ticks % 24
        steps = int(total_ticks / 24)
        bar = int(steps / 16)
        steps = steps % 16
        # bar += 1
        # steps += 1
        return BarStepTick(bar, steps, ticks)

    @staticmethod
    def bst2sec(bst: 'BarStepTick', bpm: float = 120) -> float:
        '''
        BST is the Bar-Step-Tick time signature, but in pretty_midi we need the exact start time and end time in seconds
        to draw a note, so we need this function to convert the BST to seconds given the bpm, bar, and step.
        '''
        barsec = bst.bar * (60 / (bpm / 4))
        stepsec = bst.step * (60 / (bpm / 4)) / 16
        ticksec = 1 / 192 * bst.tick
        return barsec + stepsec + ticksec

    @staticmethod
    def beat2sec(beat: float, bpm) -> float:
        # step = int(beat * 4)
        # return BarStepTick.bst2sec(BarStepTick.step2bst(step), bpm=bpm)
        return beat * (60 / (bpm / 4)) / 4

    @staticmethod
    def step2bst(step: int) -> 'BarStepTick':
        '''
        Calculate the bar and step location based on steps, this function is used when implement grooves.
        '''
        bar = math.floor(step / 16)
        step = (step - (bar * 16)) % 16
        #     print(bar,step)
        return BarStepTick(bar, step)

    @staticmethod
    def beat2steps(beat: float) -> int:
        steps = round(beat * 4)
        return steps

    @staticmethod
    def bst2beats(bst: 'BarStepTick'):
        return bst.bar * 4 + bst.step / 4 + bst.tick / 24

    @staticmethod
    def sec2beats(sec: float, bpm: int = 120) -> float:
        bst = BarStepTick.sec2bst(sec, bpm)
        return bst.bar * 4 + bst.step / 4

    @staticmethod
    def sec2bars(sec: float, bpm: float = 120) -> float:
        return math.ceil(sec / (60 / (bpm * 96))) / 384


# backward compatible names mapping
_HF_MODEL_CHECKPOINTS_MAP = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large": "facebook/musicgen-large",
    "melody": "facebook/musicgen-melody",
    "style": "facebook/musicgen-style",
}


section_patterns = {
    'pattern1': [
        ("Intro", 8),
        ("Verse1", 16),
        ("PreChorus", 8),
        ("Chorus1", 16),
        ("Verse2", 16),
        ("PreChorus", 8),
        ("Chorus2", 16),
        ("Bridge", 8),
        ("Chorus3", 16),
        ("Outro", 8)
    ],
    'pattern2': [
        ("Intro", 8),
        ("Verse1", 16),
        ("Chorus1", 8),
        ("Verse2", 16),
        ("Chorus2", 8),
        ("Bridge", 8),
        ("Chorus3", 8),
        ("Outro", 8)
    ],
    'pattern3': [
        ("Intro", 8),
        ("Verse1", 16),
        ("PreChorus1", 8),
        ("Chorus1", 16),
        ("Verse2", 16),
        ("PreChorus2", 8),
        ("Chorus2", 16),
        ("Bridge", 8),
        ("Chorus3", 16),
        ("Outro", 8)
    ],
    'pattern4': [
        ("Intro", 8),
        ("Chorus", 8),
        ("Verse", 16),
        ("PreChorus", 4),
        ("Chorus", 8),
        ("Verse", 16),
        ("PreChorus", 4),
        ("Chorus", 8),
        ("Outro", 8)
    ]
}

def form_prompt_structure(pattern: str):
    structure = section_patterns[pattern]
    return "->".join([f"{x[0]}({x[1]})" for x in structure])

def total_bars(pattern: str):
    structure = section_patterns[pattern]
    return sum(x[1] for x in structure)

def form_pattern_string_list(pattern: str):
    structure = section_patterns[pattern]
    r = []
    for x in structure:
        r.extend([x[0] for _ in range(x[1])])
    return r

def form_stat_prompt(pattern_str_list: str, from_bar: int, to_bar: int):
    belong_sections = pattern_str_list[from_bar: to_bar]
    sections_stat = []
    for s in belong_sections:
        if not sections_stat:
            sections_stat.append([s, 1])
            continue
        if sections_stat[-1][0] == s:
            sections_stat[-1][1] = sections_stat[-1][1]+1
        else:
            sections_stat.append([s, 1])
    sections_stat = [f"{count} bars of {label}" for label, count in sections_stat]
    if len(sections_stat) > 1:
        sections_stat[-1] = "and " + sections_stat[-1]
    return ", ".join(sections_stat)


class MusicGenBaseline(MusicGen):
    def _generate_tokens(self, attributes: tp.List[ConditioningAttributes],
                         prompt_tokens: tp.Optional[torch.Tensor], progress: bool = False, sec_pattern: str = None) -> torch.Tensor:
        """Generate discrete audio tokens given audio prompt and/or conditions.

        Args:
            attributes (list of ConditioningAttributes): Conditions used for generation (text/melody).
            prompt_tokens (torch.Tensor, optional): Audio prompt used for continuation.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        Returns:
            torch.Tensor: Generated audio, of shape [B, C, T], T is defined by the generation params.
        """
        total_gen_len = int(self.duration * self.frame_rate)
        max_prompt_len = int(min(self.duration, self.max_duration) * self.frame_rate)
        current_gen_offset: int = 0

        def _progress_callback(generated_tokens: int, tokens_to_generate: int):
            generated_tokens += current_gen_offset
            if self._progress_callback is not None:
                # Note that total_gen_len might be quite wrong depending on the
                # codebook pattern used, but with delay it is almost accurate.
                self._progress_callback(generated_tokens, tokens_to_generate)
            else:
                print(f'{generated_tokens: 6d} / {tokens_to_generate: 6d}', end='\r')

        if prompt_tokens is not None:
            assert max_prompt_len >= prompt_tokens.shape[-1], \
                "Prompt is longer than audio to generate"

        callback = None
        if progress:
            callback = _progress_callback
        if self.duration <= self.max_duration:
            # generate by sampling from LM, simple case.
            with self.autocast:
                gen_tokens = self.lm.generate(
                    prompt_tokens, attributes,
                    callback=callback, max_gen_len=total_gen_len, **self.generation_params)

        else:
            # now this gets a bit messier, we need to handle prompts,
            # melody conditioning etc.
            ref_wavs = [attr.wav['self_wav'] for attr in attributes]
            all_tokens = []
            if prompt_tokens is None:
                prompt_length = 0
            else:
                all_tokens.append(prompt_tokens)
                prompt_length = prompt_tokens.shape[-1]

            assert self.extend_stride is not None, "Stride should be defined to generate beyond max_duration"
            assert self.extend_stride < self.max_duration, "Cannot stride by more than max generation duration."
            stride_tokens = int(self.frame_rate * self.extend_stride)
            x = 0
            pattern_string_list = form_pattern_string_list(sec_pattern) if sec_pattern is not None else []
            initial_text_prompt = attributes[0].text['description']
            while current_gen_offset + prompt_length < total_gen_len:
                print(f"Iteration {x}")
                x += 1
                time_offset = current_gen_offset / self.frame_rate
                chunk_duration = min(self.duration - time_offset, self.max_duration)
                # print(time_offset, chunk_duration)
                max_gen_len = int(chunk_duration * self.frame_rate)
                from_sec = time_offset
                to_sec = from_sec + chunk_duration
                from_bar = int(BarStepTick.sec2bars(from_sec, 120))
                to_bar = int(BarStepTick.sec2bars(to_sec, 120))
                sections_stat = form_stat_prompt(pattern_string_list, from_bar, to_bar)
                for ii, att in enumerate(attributes):
                    structure_context = (f"\nFor now, you are generating the segment "
                                               f"between the {from_sec}th second and the {to_sec}th second, which corresponds to "
                                               f"the the {from_bar}th bar and the {to_bar}th bar regarding to the whole song, "
                                               f"your output should include {sections_stat} of the song structure.")
                    att.text['description'] = initial_text_prompt + structure_context
                    print(f"Iteration prompt: {att.text['description']}")
                    # print(f"Attribute text: {att.text}\nAttribute wav: {att.wav}\nAttribute attributes: {att.attributes}\nText attributes: {att.text_attributes}\nWav attributes: {att.wav_attributes}\nJoint embed: {att.joint_embed}\nJoint embed attributes: {att.joint_embed_attributes}")
                for attr, ref_wav in zip(attributes, ref_wavs):
                    wav_length = ref_wav.length.item()
                    if wav_length == 0:
                        continue
                    # We will extend the wav periodically if it not long enough.
                    # we have to do it here rather than in conditioners.py as otherwise
                    # we wouldn't have the full wav.
                    initial_position = int(time_offset * self.sample_rate)
                    wav_target_length = int(self.max_duration * self.sample_rate)
                    positions = torch.arange(initial_position,
                                             initial_position + wav_target_length, device=self.device)
                    attr.wav['self_wav'] = WavCondition(
                        ref_wav[0][..., positions % wav_length],
                        torch.full_like(ref_wav[1], wav_target_length),
                        [self.sample_rate] * ref_wav[0].size(0),
                        [None], [0.])
                with self.autocast:
                    gen_tokens = self.lm.generate(
                        prompt_tokens, attributes,
                        callback=callback, max_gen_len=max_gen_len, **self.generation_params)
                if prompt_tokens is None:
                    all_tokens.append(gen_tokens)
                else:
                    all_tokens.append(gen_tokens[:, :, prompt_tokens.shape[-1]:])
                prompt_tokens = gen_tokens[:, :, stride_tokens:]
                prompt_length = prompt_tokens.shape[-1]
                current_gen_offset += stride_tokens

            gen_tokens = torch.cat(all_tokens, dim=-1)
        return gen_tokens


if __name__ == '__main__':
    import torchaudio
    import os
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # Uncomment this if you cannot access to huggingface
    model = MusicGenBaseline.get_pretrained('facebook/musicgen-large')

    def bars2sec(bars, bpm = 120, ppq: int = 96) -> float:
        total_ticks = bars * 16 * 24
        return total_ticks * (60 / (bpm * ppq))

    def generate_musicgen_song(sec_pattern: str, key: str, demo_id: int):
        prompt = f"Generate an electronic pop track at 120BPM in {key} major. Follow this structure: {form_prompt_structure(sec_pattern)}."
        model.set_generation_params(
            use_sampling=True,
            top_k=250,
            duration=bars2sec(total_bars(sec_pattern), 120),
            extend_stride=10
        )
        output = model.generate(descriptions=[prompt], progress=True, return_tokens=False, sec_pattern=sec_pattern)
        save_path = f"./demos/MusicGen/{sec_pattern}"
        os.makedirs(save_path, exist_ok=True)
        torchaudio.save(os.path.join(save_path, f"musicgen_{sec_pattern}_{key}_{demo_id}.wav"), output[0].detach().cpu(), 32000)

    for sp in ('pattern1', 'pattern2', 'pattern3', 'pattern4'):
        for key in ('C', 'F', 'G', 'A#'):
            for i in range(2):
                generate_musicgen_song(sp, key, i+1)
