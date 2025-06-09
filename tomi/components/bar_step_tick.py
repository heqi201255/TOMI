"""
Version: 1.0 add time_signature handle.
"""

import math
from typing import Union
from . import TimeSignature
from tomi import make_property, better_property


class BarStepTick:
    """
    This clss is used as the DAW time unit in many other modules. From its name you can tell it means the bar, step, and
    tick typically used in modern Digital Audio Workstation (DAW) programs. In MIDI standards, it only uses Ticks for timing,
    you'll need to convert it to other timing units with stuff like PPQ. Here we use BarStepTick as the only time unit across
    TOMI to simplify the understanding of music timing.
    """
    bar = make_property(src_attr_name='_bar', value_type=int, value_min=0)
    step = make_property(src_attr_name='_step', value_type=int, value_min=0, value_max=lambda self: self.time_signature.steps_per_bar)
    tick = make_property(src_attr_name='_tick', value_type=int, value_min=0, value_max=23)
    time_signature: TimeSignature

    def __init__(self, bar: int | list = 0, step: int = 0, tick: int = 0, time_signature: TimeSignature = None):
        """
        The creation of a BarStepTick instance, it will check all the params whether they are in valid ranges.
        :param bar: should be a non-negative integer. Each bar has 16 steps.
        :param step: should be between 0 and 15 inclusively. Each step has 24 ticks.
        :param tick: should be between 0 and 23 inclusively.
        """
        if isinstance(bar, list):
            if len(bar) == 1:
                b, s, t = int(bar[0]), 0, 0
            elif len(bar) == 2:
                b, s, t = int(bar[0]), int(bar[1]), 0
            elif len(bar) == 3:
                b, s, t = int(bar[0]), int(bar[1]), int(bar[2])
            else:
                raise ValueError("Wrong format")
        else:
            b, s, t = int(bar), int(step), int(tick)
        self._bar = 0
        self._step = 0
        self._tick = 0
        self._time_signature = TimeSignature.default() if time_signature is None else time_signature
        self.bar = b
        self.step = s
        self.tick = t

    @better_property(src_attr_name='_time_signature', value_type=TimeSignature)
    def time_signature(self):
        def fset(_self, time_signature):
            seconds = _self.to_seconds(120)
            _self._time_signature = time_signature
            bst = BarStepTick.sec2bst(seconds, 120, time_signature=time_signature)
            _self.bar = bst.bar
            _self.step = bst.step
            _self.tick = bst.tick
        return fset

    def __hash__(self):
        return hash(self.to_steps())

    def __lt__(self, other):
        return self.to_ticks() < other.to_ticks()

    def __eq__(self, other):
        return self.to_ticks() == other.to_ticks() and self.time_signature == other.time_signature

    def __le__(self, other):
        return self.to_ticks() <= other.to_ticks()

    def __iter__(self):
        return iter((self.bar, self.step, self.tick))

    def __add__(self, other: Union['BarStepTick', int, str]):
        if isinstance(other, str):
            other = BarStepTick.str2bst(other.lower())
        elif isinstance(other, int):
            other = BarStepTick(other, time_signature=self.time_signature)
        other.time_signature = self.time_signature
        total_ticks = self.to_ticks() + other.to_ticks()
        return self.tick2bst(total_ticks, self.time_signature)

    def __sub__(self, other: Union['BarStepTick', int, str]):
        if isinstance(other, str):
            other = BarStepTick.str2bst(other.lower())
        elif isinstance(other, int):
            other = BarStepTick(other, time_signature=self.time_signature)
        other.time_signature = self.time_signature
        sub_ticks = self.to_ticks() - other.to_ticks()
        assert sub_ticks >= 0, "BarStepTick cannot subtract another BarStepTick instance that is longer."
        return self.tick2bst(sub_ticks, self.time_signature)

    def __mul__(self, other):
        ticks = self.to_ticks() * other
        return self.tick2bst(ticks, self.time_signature)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        ticks = int(self.to_ticks() / other)
        return self.tick2bst(ticks, self.time_signature)

    def __rtruediv__(self, other):
        ticks = int(other / self.to_ticks())
        return self.tick2bst(ticks, self.time_signature)

    def is_empty(self) -> bool:
        """
        Whether the bar, step, and tick values are all 0.
        :return: bool.
        """
        return self.bar == 0 and self.step == 0 and self.tick == 0

    def to_beats(self) -> float:
        """
        Convert current BST to unit of beats, could be a float.
        :return: float.
        """
        return self.bar * self.time_signature.beats_per_bar + self.step / self.time_signature.steps_per_beat + self.tick / self.time_signature.ticks_per_beat

    def to_bars(self) -> float:
        """
        Convert current BST to unit of bars, could be a float.
        :return: float.
        """
        return self.bar + (self.step / self.time_signature.steps_per_bar) + (self.tick / self.time_signature.ticks_per_bar)

    def to_steps(self) -> int:
        """
        Convert current BST to unit of steps.
        :return: int.
        """
        return self.bar * self.time_signature.steps_per_bar + self.step + round(self.tick / 24)

    def to_ticks(self) -> int:
        """
        Convert current BST to unit of ticks.
        :return: int.
        """
        return self.bar * self.time_signature.ticks_per_bar + self.step * 24 + self.tick

    def to_seconds(self, bpm: int | float = 120, ppq: int = 96) -> float:
        """
        Convert current BST to unit of seconds.
        :param bpm: BPM.
        :param ppq: PPQ, default is 96.
        :return: float
        """
        return self.to_ticks() * (60 / (bpm * ppq))

    @staticmethod
    def str2bst(s: str) -> 'BarStepTick':
        try:
            bar_index = s.index("b")
        except ValueError:
            bar_index = 0
        try:
            step_index = s.index("s")
        except ValueError:
            step_index = 0
        try:
            tick_index = s.index("t")
        except ValueError:
            tick_index = 0
        try:
            ts_index_left = s.index("(")
            ts_index_mid = s.index("/")
            ts_index_right = s.index(")")
            ts_numerator = int(s[ts_index_left + 1: ts_index_mid])
            ts_denominator = int(s[ts_index_mid + 1: ts_index_right])
            time_signature = TimeSignature(ts_numerator, ts_denominator)
        except ValueError:
            time_signature = TimeSignature.default()
        bar = 0 if not bar_index else int(s[:bar_index])
        step = 0 if not step_index else int(s[bar_index + 1:step_index]) if bar_index else int(s[:step_index])
        tick = 0 if not tick_index else int(s[step_index + 1:tick_index]) if step_index else int(s[bar_index + 1:tick_index]) if bar_index else int(s[:tick_index])
        return BarStepTick(bar, step, tick, time_signature=time_signature)

    def __str__(self):
        s = []
        if self.bar != 0:
            s.append(str(self.bar) + "b")
        if self.step != 0:
            s.append(str(self.step) + "s")
        if self.tick != 0:
            s.append(str(self.tick) + "t")
        ts = f"({self.time_signature.numerator}/{self.time_signature.denominator})"
        return f"{"".join(s)}{ts}" if s else f"0b{ts}"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def sec2bst(sec: int | float, bpm: int | float = 120, time_signature: TimeSignature = None, ppq: int = 96) -> 'BarStepTick':
        total_ticks = math.ceil(sec / (60 / (bpm * ppq)))
        return BarStepTick.tick2bst(total_ticks, time_signature=TimeSignature.default() if time_signature is None else time_signature)

    @staticmethod
    def tick2bst(tick: int, time_signature: TimeSignature = None) -> 'BarStepTick':
        time_signature = TimeSignature.default() if time_signature is None else time_signature
        ticks = tick % 24
        steps = int(tick / 24)
        steps_for_bar = time_signature.steps_per_bar
        bar = int(steps / steps_for_bar)
        steps = steps % steps_for_bar
        return BarStepTick(bar, steps, ticks, time_signature=time_signature)

    @staticmethod
    def beat2sec(beat: int | float, bpm: int | float = 120, time_signature: TimeSignature = None) -> float:
        time_signature = TimeSignature.default() if time_signature is None else time_signature
        return 240 * beat / time_signature.denominator / bpm

    @staticmethod
    def step2sec(step: int, bpm: int | float = 120, ppq: int = 96) -> float:
        return step * 24 * (60 / (bpm * ppq))

    @staticmethod
    def step2bst(step: int, time_signature: TimeSignature = None) -> 'BarStepTick':
        time_signature = TimeSignature.default() if time_signature is None else time_signature
        bar = math.floor(step / time_signature.steps_per_bar)
        step = step - (bar * time_signature.steps_per_bar)
        return BarStepTick(bar, step, time_signature=time_signature)

    @staticmethod
    def beat2steps(beat: int | float, time_signature: TimeSignature = None) -> int:
        time_signature = TimeSignature.default() if time_signature is None else time_signature
        return round(time_signature.steps_per_beat * beat)

    @staticmethod
    def sec2beats(sec: int | float, bpm: int | float = 120, time_signature: TimeSignature = None) -> float:
        time_signature = TimeSignature.default() if time_signature is None else time_signature
        return sec * bpm * time_signature.denominator / 240

    @staticmethod
    def sec2bars(sec: int | float, bpm: int | float = 120, time_signature: TimeSignature = None, ppq: int = 96) -> float:
        time_signature = TimeSignature.default() if time_signature is None else time_signature
        return math.ceil(sec / (60 / (bpm * ppq))) / time_signature.ticks_per_bar

    @staticmethod
    def bars2sec(bars: int | float, bpm: int | float = 120, time_signature: TimeSignature = None, ppq: int = 96) -> float:
        time_signature = TimeSignature.default() if time_signature is None else time_signature
        total_ticks = bars * time_signature.ticks_per_bar
        return total_ticks * (60 / (bpm * ppq))

    @staticmethod
    def bars2bst(bars: int | float, time_signature: TimeSignature = None) -> 'BarStepTick':
        time_signature = TimeSignature.default() if time_signature is None else time_signature
        return BarStepTick.tick2bst(bars * time_signature.ticks_per_bar, time_signature=time_signature)

    @staticmethod
    def relocate_seconds_on_target_time_signature(sec: int | float, bpm: int | float = 120, *, original_time_signature: TimeSignature, target_time_signature: TimeSignature) -> float:
        return BarStepTick.bars2sec(BarStepTick.sec2bars(sec, bpm, original_time_signature), bpm, target_time_signature)

BST = BarStepTick
