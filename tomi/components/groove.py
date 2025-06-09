from typing import Union
from . import MIDIType, TimeSignature, BarStepTick, GrooveSpeed
import numpy as np
from numpy.typing import NDArray


class GrooveBlock:
    Sustain = 0
    Pause = 1
    Next = 2
    Replay = 3
    StartBlocks = {2, 3}
    ActionBlocks = {1, 2, 3}


class Groove:
    def __init__(self, groove: Union[list[int], NDArray[np.int8], 'Groove'] = None, midi_type: MIDIType = None, is_plain_groove: bool = False, time_signature: TimeSignature = None):
        self.speed = self.progression_count = self.midi_type = self.bar_length = None
        self.time_signature = TimeSignature.default() if time_signature is None else time_signature
        self.is_plain_groove = is_plain_groove
        if groove is None:
            self.groove = np.ndarray([], dtype=np.int8)
        if isinstance(groove, list):
            groove = np.array(groove, dtype=np.int8)
        if isinstance(groove, np.ndarray):
            assert midi_type, "'midi_type' parameter cannot be None if a list is provided!"
            self.set_groove(groove, midi_type)
        elif isinstance(groove, Groove):
            self.set_groove(groove.get_groove(), groove.get_midi_type())

    def __lt__(self, other):
        return BarStepTick(self.bar_length, time_signature=self.time_signature) < BarStepTick(other.bar_length, time_signature=other.time_signature)

    def __hash__(self):
        return hash((tuple(self.groove), self.midi_type, self.progression_count, self.bar_length, self.time_signature.numerator, self.time_signature.denominator))

    def __getitem__(self, item):
        return self.groove[item]

    def __iter__(self):
        return iter(self.groove)

    def __len__(self):
        return self.groove.shape[0]

    def __str__(self):
        s = f"|Groove|type: {self.get_midi_type().name}|bars: {self.bar_length}|prog_count: {self.progression_count}|time_signature: {self.time_signature.numerator}/{self.time_signature.denominator}\n"
        started = False
        count = 0
        for b in self.groove:
            if count % 4 == 0:
                s += "|"
            if b == GrooveBlock.Next:
                s += "S"
                started = True
            elif b == GrooveBlock.Replay:
                s += "R"
                started = True
            elif b == GrooveBlock.Sustain:
                if started:
                    s += "="
                else:
                    s += " "
            else:
                s += "P"
                started = False
            count += 1
        return s

    def __eq__(self, other):
        return self.groove == other.groove and self.time_signature == other.time_signature

    def set_groove(self, groove: NDArray[np.int8], midi_type: MIDIType):
        self.verify_groove(groove)
        self.bar_length = self.calc_bar_length(groove)
        self.groove = groove[:int(self.bar_length * self.time_signature.steps_per_bar)]
        self.midi_type = midi_type
        self.progression_count = self.count_progression(self)
        self.speed = self.check_speed()

    def check_speed(self):
        if self.is_plain_groove:
            return GrooveSpeed.Plain
        start_count = sum(1 for i in self.groove if i in GrooveBlock.StartBlocks)
        if start_count <= self.bar_length:
            return GrooveSpeed.Normal
        elif start_count <= 2 * self.bar_length:
            return GrooveSpeed.Fast
        else:
            return GrooveSpeed.Rapid

    @staticmethod
    def verify_groove(groove_list: NDArray[np.int8]):
        f = False
        for groove in groove_list:
            if groove == GrooveBlock.Next:
                f = True
            elif not f and groove == GrooveBlock.Replay:
                raise ValueError("wrong groove list! no start-n before start-p")

    def calc_bar_length(self, groove: NDArray[np.int8]):
        g = groove.copy()
        l = g.size
        if l % self.time_signature.steps_per_bar != 0:
            raise ValueError("wrong groove list length!")
        i = int(l / 2)
        while i >= self.time_signature.steps_per_bar and g.size % self.time_signature.steps_per_bar == 0:
            if np.all(g[:i] == g[i:]):
                g = g[:i]
            else:
                break
            i = int(i / 2)
        bar_count = int((i * 2) / self.time_signature.steps_per_bar)
        return bar_count

    @staticmethod
    def count_progression(groove: 'Groove'):
        return sum(1 for i in groove.groove if i == GrooveBlock.Next)

    @staticmethod
    def compare_grooves(g1: 'Groove', g2: 'Groove') -> float:
        '''
        Calculate the groove difference value in the range between 0 (identical) and 1 (totally different).
        :param g1: The groove that you want to compare it with other grooves, the difference calculation is based on this groove.
        :param g2: The other groove to compare with.
        :return: a float in range [0,1].
        '''
        if g1.speed != g2.speed or g1.time_signature != g2.time_signature:
            return 1
        i1 = i2 = 0
        while i1 < len(g1) and g1[i1] != GrooveBlock.Next:
            i1 += 1
        while i2 < len(g2) and g2[i2] != GrooveBlock.Next:
            i2 += 1
        g1g = g1[i1:]
        g2g = g2[i2:]
        if len(g1g) < len(g2g):
            g2g = g2g[:len(g1g)]
        else:
            g1g = g1g[:len(g2g)]
        valid_g1g = len([x for x in g1g if x != GrooveBlock.Sustain])
        valid_g2g = len([x for x in g2g if x != GrooveBlock.Sustain])
        ind = 0
        diff_count = 0
        while ind < len(g1g):
            if g2g[ind] != g1g[ind]:
                diff_count += 1
            ind += 1
        if valid_g2g + valid_g1g == 0:
            return 0
        return diff_count / (valid_g1g + valid_g2g)

    def get_groove(self):
        return self.groove

    def get_midi_type(self) -> MIDIType:
        return self.midi_type

    def get_length(self):
        return self.bar_length

    def get_progression_count(self):
        return self.progression_count

    def get_rhythm(self) -> GrooveSpeed:
        return self.speed
