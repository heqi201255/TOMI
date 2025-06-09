from tomi import DEFAULT_TIME_SIGNATURE_DENOMINATOR, DEFAULT_TIME_SIGNATURE_NUMERATOR, make_property
from pretty_midi import TimeSignature as PrettyMidiTimeSignature


class TimeSignature:
    numerator = make_property(src_attr_name='_numerator', value_type=int, value_min=1, value_max=16)
    denominator = make_property(src_attr_name='_denominator', value_type=int, value_options=(2, 4, 8, 16))
    def __init__(self,
                 numerator: int | PrettyMidiTimeSignature = DEFAULT_TIME_SIGNATURE_NUMERATOR,
                 denominator: int = DEFAULT_TIME_SIGNATURE_DENOMINATOR,
                 ):
        self._numerator = self._denominator = 4
        if isinstance(numerator, PrettyMidiTimeSignature):
            self.numerator = numerator.numerator
            self.denominator = numerator.denominator
        else:
            self.numerator = numerator
            self.denominator = denominator

    def __eq__(self, other):
        return isinstance(other, TimeSignature) and self.numerator == other.numerator and self.denominator == other.denominator

    def __iter__(self):
        return iter((self.numerator, self.denominator))

    def __str__(self):
        return f"TimeSignature ({self.numerator}/{self._denominator})"

    def __hash__(self):
        return hash((self.numerator, self.denominator))

    @property
    def steps_per_bar(self) -> int:
        return int(self.numerator * self.denominator * ((4 / self.denominator) ** 2))

    @property
    def ticks_per_bar(self) -> int:
        return self.steps_per_bar * 24

    @property
    def beats_per_bar(self) -> int:
        return self.numerator

    @property
    def steps_per_beat(self) -> int:
        return self.steps_per_bar // self.beats_per_bar

    @property
    def ticks_per_beat(self) -> int:
        return self.steps_per_beat * 24

    @staticmethod
    def default():
        return TimeSignature(DEFAULT_TIME_SIGNATURE_NUMERATOR, DEFAULT_TIME_SIGNATURE_DENOMINATOR)

    def to_pretty_midi_time_signature(self):
        return PrettyMidiTimeSignature(self.numerator, self.denominator, 0.)
