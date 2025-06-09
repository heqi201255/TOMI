from typing import Union, Optional
import pretty_midi
from . import BarStepTick, TOMIEnum, TOMIEnumDescriptor, KeyMode, TimeSignature
from tomi import InvalidMIDIFileError, smart_div, make_property, better_property, printer
import numpy as np
import random
from collections import Counter


class QuantizationFactor(TOMIEnum):
    """
    Simple Enum class used for the "quantize" function of MIDIProcessor class, this class has 5 options to choose,
    the name means the magnitude of quantization, the class has a property "unit_steps" that is used to calculate the
    column length when creating a PianoRoll or getting the time location in unit of "steps"
    """
    Bar = TOMIEnumDescriptor('Bar')  # All notes are quantized to the nearest bar marker.
    HalfBar = TOMIEnumDescriptor('HalfBar')  # All notes are quantized to the nearest half-bar marker.
    Beat = TOMIEnumDescriptor('Beat')  # All notes are quantized to the nearest beat marker.
    HalfBeat = TOMIEnumDescriptor('HalfBeat')  # All notes are quantized to the nearest half-beat marker.
    Step = TOMIEnumDescriptor('Step')  # All notes are quantized to the nearest step marker.

    time_signature = make_property(src_attr_name='_time_signature', value_type=TimeSignature)

    def __init__(self, value=None):
        super(QuantizationFactor, self).__init__(value)
        self._time_signature = TimeSignature.default()

    def __hash__(self):
        return hash((self.__class__.__name__, self.name, self._time_signature))

    @property
    def unit_steps(self) -> int | float:
        match self.name:
            case 'Bar': return self.time_signature.steps_per_bar
            case 'HalfBar': return max(1, smart_div(self.time_signature.steps_per_bar, 2))
            case 'Beat': return self.time_signature.steps_per_beat
            case 'HalfBeat': return max(1, smart_div(self.time_signature.steps_per_beat, 2))
            case 'Step': return 1
        return None

    @property
    def unit_bst(self) -> BarStepTick:
        match self.name:
            case 'Bar': return BarStepTick(1, time_signature=self.time_signature)
            case 'HalfBar': return BarStepTick(step=max(1, smart_div(self.time_signature.steps_per_bar, 2)), time_signature=self.time_signature)
            case 'Beat': return BarStepTick(step=self.time_signature.steps_per_beat, time_signature=self.time_signature)
            case 'HalfBeat': return BarStepTick(step=max(1, smart_div(self.time_signature.steps_per_beat, 2)), time_signature=self.time_signature)
            case 'Step': return BarStepTick(step=1, time_signature=self.time_signature)
        return None

    @property
    def unit_value(self) -> int:
        return self.time_signature.steps_per_bar // self.unit_steps

    def __call__(self, time_signature: TimeSignature) -> 'QuantizationFactor':
        self.time_signature = time_signature
        return self

    def __truediv__(self, other):
        return self.unit_value / other

    def __rtruediv__(self, other):
        return other / self.unit_value

    def __floordiv__(self, other):
        return self.unit_value // other

    def __rfloordiv__(self, other):
        return other // self.unit_value

    def __mul__(self, other):
        return self.unit_value * other

    def __rmul__(self, other):
        return self.unit_value * other


class QuantizationMode(TOMIEnum):
    """
    Simple Enum for the "quantize" function of MIDIProcessor class.
    """
    StartEnd = TOMIEnumDescriptor('Start_end')  # Quantize all notes based on the start time and end time of each note.
    Length = TOMIEnumDescriptor('Length')  # Quantize all notes based on the length of each note.


class MIDINote:
    pitch = make_property(src_attr_name='_pitch', value_type=int, value_min=0, value_max=127)
    start = make_property(src_attr_name='_start', value_type=(int, float), value_min=0, value_max=lambda self: self._end)
    end = make_property(src_attr_name='_end', value_type=(int, float), value_min=lambda self: self._start)
    velocity = make_property(src_attr_name='_velocity', value_type=int, value_min=0, value_max=127)
    duration = property(fget=lambda self: self.end - self.start)
    def __init__(self, note: Union['MIDINote', list, tuple, np.ndarray, pretty_midi.Note] = None,
                 pitch: int = None, start: int | float = None, end: int | float = None, velocity: int = None):
        if note is not None:
            if isinstance(note, (list, tuple, np.ndarray)):
                assert len(note) == 4, "Wrong note format! Should be a list of [pitch, start, end, velocity]"
                pitch = note[0]
                start = note[1]
                end = note[2]
                velocity = note[3]
            elif isinstance(note, (pretty_midi.Note, MIDINote)):
                self._pitch = note.pitch
                self._start = note.start
                self._end = note.end
                self._velocity = note.velocity
                return
            else:
                raise ValueError('Wrong "note" parameter.')
        assert pitch is not None, "Note pitch is not provided."
        assert 0 <= pitch <= 127, "Note pitch out of range."
        assert start is not None, "Note start time is not provided."
        assert end is not None, "Note end time is not provided."
        assert end > start, "Note end time before note start time."
        assert velocity is not None, "Note velocity is not provided."
        assert 0 <= velocity <= 127, "Note velocity out of range."
        self._pitch = int(pitch)
        self._start = start
        self._end = end
        self._velocity = int(velocity)

    def __str__(self):
        return f"Note(pitch: {pretty_midi.note_number_to_name(self.pitch)}, start: {self.start}, end: {self.end}, velocity: {self.velocity})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other: 'MIDINote'):
        return (isinstance(other, MIDINote) and self.pitch == other.pitch and self.start == other.start and self.end == other.end
                and self.velocity == other.velocity)

    def __hash__(self):
        return hash((self.pitch, self.start, self.end, self.velocity))

    def __lt__(self, other):
        return self.pitch < other.pitch

    def __len__(self):
        return self.duration

    def __sub__(self, other):
        return self.pitch - other.pitch

    def copy(self):
        return MIDINote(self)

    def at_bar(self, bpm: float, time_signature: TimeSignature = None) -> bool:
        bst = self.bst_location(bpm, time_signature)
        return bst.step == 0 and bst.tick == 0

    def bst_location(self, bpm: float, time_signature: TimeSignature = None) -> BarStepTick:
        return BarStepTick.sec2bst(self.start, bpm, time_signature)

    def time_offset(self, time_offset: float) -> 'MIDINote':
        s = max(0., self.start + time_offset)
        e = max(0., self.end + time_offset)
        assert s < e, "Time offset out of range."
        self._start = s
        self._end = e
        return self

    def is_between(self, bounds: tuple[float, float]) -> bool:
        return bounds[0] <= self.start <= bounds[1] and bounds[0] <= self.end <= bounds[1]

    def crop(self, bounds: tuple[float, float]) -> Union['MIDINote', None]:
        if self.end <= bounds[0] or self.start >= bounds[1]:
            return None
        if self.is_between(bounds):
            return MIDINote([self.pitch, self.start, self.end, self.velocity])
        if bounds[0] <= self.start < bounds[1]:
            return MIDINote([self.pitch, self.start, bounds[1], self.velocity])
        if bounds[0] < self.end <= bounds[1]:
            return MIDINote([self.pitch, bounds[0], self.end, self.velocity])
        if self.start <= bounds[0] and self.end >= bounds[1]:
            return MIDINote([self.pitch, bounds[0], bounds[1], self.velocity])
        return None

    def to_pretty_midi_note(self) -> pretty_midi.Note:
        return pretty_midi.Note(self.velocity, self.pitch, self.start, self.end)


class MIDINoteList:
    start_time = property(fget=lambda self: min(note.start for note in self.notes) if self.notes else 0)
    end_time = property(fget=lambda self: max(x.end for x in self.notes) if self.notes else 0)
    start_bst = property(fget=lambda self: BarStepTick.sec2bst(self.start_time, self.bpm, self.time_signature), doc="Start time BST of the Midi.")
    end_bst = property(fget=lambda self: BarStepTick.sec2bst(self.end_time, self.bpm, self.time_signature), doc="End time BST of the Midi.")
    def __init__(self, note_list: list | set | pretty_midi.Instrument = None, bpm: int | float = 120, time_signature: TimeSignature = None) -> None:
        if note_list is None:
            note_list = []
        elif isinstance(note_list, pretty_midi.Instrument):
            note_list.remove_invalid_notes()
            note_list = [[note.pitch, note.start, note.end, note.velocity] for note in note_list.notes]
        elif not isinstance(note_list, (list, set)):
            raise TypeError("Wrong note_list, note_list should be a list or set of notes or a pretty_midi.Instrument instance.")
        self.notes = [MIDINote(x) for x in note_list]
        self._time_signature = TimeSignature.default() if time_signature is None else time_signature
        self._bpm = bpm

    def __iter__(self):
        return self.notes.__iter__()

    def __len__(self):
        return len(self.notes)

    def __hash__(self):
        return hash((self.bpm, (x.__hash__() for x in self.notes)))

    def __eq__(self, other):
        return isinstance(other, MIDINoteList) and self.bpm == other.bpm and len(self.notes) == len(other.notes) and all(note in other.notes for note in self.notes) and self.time_signature == other.time_signature

    def __contains__(self, item: MIDINote):
        return item in self.notes

    def __str__(self):
        return f"MIDINoteList({len(self.notes)} notes, {self.bpm} bpm, {self.time_signature})"

    def __repr__(self):
        return self.__str__()

    def print_notes(self):
        printer(f"MIDINoteList:\n{'\n'.join(str(n) for n in self.notes)}")

    def copy(self):
        return MIDINoteList(self.notes, self.bpm, self.time_signature)

    @property
    def time_gaps(self):
        if not self.notes:
            return []
        # Sort notes by start time
        sorted_events = sorted(self.notes, key=lambda x: x.start)
        # Initialize variables
        gaps = []
        current_max_end = sorted_events[0].end
        # Find gaps between events
        for i in range(1, len(sorted_events)):
            current_start = sorted_events[i].start
            current_end = sorted_events[i].end
            # If there's a gap between current_max_end and current_start
            if current_start > current_max_end:
                gaps.append((current_max_end, current_start))
            # Update current_max_end if necessary
            current_max_end = max(current_max_end, current_end)
        return gaps

    @better_property(src_attr_name='_bpm', value_type=(int, float), value_min=0, inclusive_boundary=False)
    def bpm(self):
        def fset(_self, target_bpm: float | int):
            for note in _self.notes:
                note._start = BarStepTick.sec2bst(note.start, _self._bpm).to_seconds(target_bpm)
                note._end = BarStepTick.sec2bst(note.end, _self._bpm).to_seconds(target_bpm)
            _self._bpm = target_bpm
        return fset

    @better_property(src_attr_name='_time_signature', value_type=TimeSignature)
    def time_signature(self):
        """
        Get & Set the current MIDINoteList's time_signature.
        If set to a new time_signature, it will to an auto-conversion for all notes' start & end timing.
        The conversion is based on bars, which means, if the original time signature is 4/4 and have 4 bars long, the result should be a 4 bars long midi of 3/4.
        If you want to change the time_signature without converting the notes, set '_time_signature' directly.
        """
        def fset(_self, time_signature: TimeSignature):
            _self.notes = [MIDINote([n.pitch,
                          BarStepTick.relocate_seconds_on_target_time_signature(n.start, _self.bpm, original_time_signature=_self.time_signature, target_time_signature=time_signature),
                          BarStepTick.relocate_seconds_on_target_time_signature(n.end, _self.bpm, original_time_signature=_self.time_signature, target_time_signature=time_signature),
                          n.velocity]) for n in _self.notes]
            _self._time_signature = time_signature
        return fset

    def append(self, note: MIDINote | pretty_midi.Note | list):
        self.notes.append(MIDINote(note))

    def extend(self, note_list: 'MIDINoteList'):
        original_bpm = note_list.bpm
        note_list.bpm = self.bpm
        self.notes.extend(MIDINote(note) for note in note_list)
        note_list.bpm = original_bpm

    def move_key(self, offset: int):
        for note in self.notes:
            note.pitch = (note.pitch + offset) % 128

    def time_offset(self, offset: float):
        for note in self.notes:
            note.time_offset(offset)

    def get_notes(self):
        return self.notes

    def duplicate(self):
        time_offset = BarStepTick(self.get_bar_len(), time_signature=self.time_signature).to_seconds(bpm=self.bpm)
        new_notelist = [MIDINote([n.pitch, n.start + time_offset, n.end + time_offset, n.velocity]) for n in self.notes]
        return MIDINoteList(self.notes + new_notelist, self.bpm, self.time_signature)

    def get_note_list(self):
        return [[x.pitch, x.start, x.end, x.velocity] for x in self.notes]

    def to_numpy(self):
        return np.array([np.array([[x.pitch, x.start, x.end, x.velocity]]) for x in self.notes])

    def slice_length(self, bars: int, inplace: bool = False):
        end_seconds = BarStepTick(bars, time_signature=self.time_signature).to_seconds(bpm=self.bpm)
        nl = self if inplace else MIDINoteList(bpm=self.bpm, time_signature=self.time_signature)
        nl.notes = [MIDINote([n.pitch, n.start, n.end if n.end <= end_seconds else end_seconds, n.velocity]) for n in self.notes if n.start < end_seconds]
        return nl

    def get_bar_len(self):
        bst = BarStepTick.sec2bst(self.end_time, self.bpm, self.time_signature)
        return bst.bar + 1 if bst.step > 0 else bst.bar

    def convert_time_signature(self, time_signature: TimeSignature, inplace: bool = False) -> 'MIDINoteList':
        """
        Convert the current MIDINoteList to the given time signature. The conversion is based on bars.
        That means, if the original time signature is 4/4 and have 4 bars long, the result should be a 4 bars long midi of 3/4.
        :param time_signature: TimeSignature.
        :param inplace: If True, the time signature will be modified inplace.
        :return: MIDINoteList.
        """
        assert isinstance(time_signature, TimeSignature)
        nl = self if inplace else MIDINoteList(bpm=self.bpm, time_signature=time_signature)
        nl.notes = [MIDINote([n.pitch,
                      BarStepTick.relocate_seconds_on_target_time_signature(n.start, self.bpm, original_time_signature=self.time_signature, target_time_signature=time_signature),
                      BarStepTick.relocate_seconds_on_target_time_signature(n.end, self.bpm, original_time_signature=self.time_signature, target_time_signature=time_signature),
                      n.velocity]) for n in self.notes] if time_signature != self.time_signature else [n.copy() for n in self.notes]
        nl.time_signature = time_signature
        return nl

    def quantize(self,
                 factor: QuantizationFactor = None,
                 mode: QuantizationMode = None,
                 remove_short_notes: bool = True,
                 inplace: bool = False) -> 'MIDINoteList':
        """
        Quantize all notes according the factor and mode parameters.
        :param factor: QuantizationFactor, means the 'granularity' of piano roll to be processed.
        :param mode: QuantizationMode, "start_end" will quantize the start location and end location for every note, "length" will quantize the length of all notes to be same as the quantization factor.
        :param remove_short_notes: if True, it will remove notes that is too short to be considered as valid notes.
        :param inplace:
        :return: MIDINoteList
        """
        def quantize_note(note: MIDINote):
            if remove_short_notes:
                if small_qf:
                    if note.duration < min_time:
                        return None
                elif note.duration < min_time_for_longer_factor:
                    return None
            note_st = round(note.start / time_unit) * time_unit
            if is_start_end_mode:
                note_ed = round(note.end / time_unit) * time_unit
                note_ed = note_ed + time_unit if note_st == note_ed else note_ed
            else:
                note_ed = note_st + time_unit
            return MIDINote((note.pitch, round(note_st, 5), round(note_ed, 5), note.velocity))
        factor = QuantizationFactor.Step(self.time_signature) if factor is None else factor(self.time_signature)
        mode = QuantizationMode.StartEnd if mode is None else mode
        time_unit = factor.unit_bst.to_seconds(bpm=self.bpm)
        min_time = BarStepTick(0, 0, 12, self.time_signature).to_seconds(bpm=self.bpm)
        min_time_for_longer_factor = BarStepTick(0, min(self.time_signature.beats_per_bar // factor.unit_value, self.time_signature.beats_per_bar), time_signature=self.time_signature).to_seconds(bpm=self.bpm)
        small_qf = factor.name in ('HalfBeat', 'Step')
        is_start_end_mode = mode == QuantizationMode.StartEnd
        q_nl = self if inplace else MIDINoteList(bpm=self.bpm, time_signature=self.time_signature)
        q_nl.notes = [n for n in (quantize_note(note) for note in self) if n is not None]
        return q_nl

    @staticmethod
    def load_midi_file(midi_path: str, instrument_index: int = None, original_bpm: float | int = None,
                       original_time_signature: TimeSignature = None) -> 'MIDINoteList':
        """
        Load a midi file to MIDINoteList.
        :param midi_path: MIDI file path.
        :param instrument_index: If given, will try to load only the specified instrument of the MIDI, if None, will load all instrument's notes as one MIDINoteList object.
        :param original_bpm: indicates the MIDI's original bpm, in some cases the MIDI file's tempo information is incorrect, use this parameter for correction.
        :param original_time_signature: the MIDI's original time signature, in some cases the MIDI file's time signature information is incorrect, use this parameter for correction.
        :return: a MIDINoteList object or a dict of MIDINoteList objects corresponding to multiple instruments, if 'load_meta' is True, also return the Midi meta.
        """
        if not midi_path.split('.')[-1].lower() in ('mid', "midi"):
            raise InvalidMIDIFileError(f"File '{midi_path}' is not a midi file!")
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
            return MIDINoteList.load_pretty_midi(midi, instrument_index=instrument_index, original_bpm=original_bpm, original_time_signature=original_time_signature)
        except:
            raise InvalidMIDIFileError(f"Failed to load file '{midi_path}'!")

    @staticmethod
    def load_pretty_midi(pm: pretty_midi.PrettyMIDI | pretty_midi.Instrument, instrument_index: int = None,
                         original_bpm: float | int = None, original_time_signature: TimeSignature = None) -> 'MIDINoteList':
        assert isinstance(pm, (pretty_midi.PrettyMIDI, pretty_midi.Instrument))
        try:
            if isinstance(pm, pretty_midi.PrettyMIDI):
                if original_bpm is None:
                    original_bpm = pm.get_tempo_changes()[-1][0]
                if original_time_signature is None:
                    original_time_signature = TimeSignature(pm.time_signature_changes[-1]) if pm.time_signature_changes else TimeSignature.default()
                assert isinstance(original_bpm, (float, int))
                pm.remove_invalid_notes()
                if instrument_index is not None:
                    note_list = set(pm.instruments[instrument_index].notes)
                else:
                    note_list = set(note for inst in pm.instruments for note in inst.notes)
                return MIDINoteList(note_list, original_bpm, original_time_signature)
            elif isinstance(pm, pretty_midi.Instrument):
                assert instrument_index is None, "Cannot set instrument_index because input object is a pretty_midi.Instrument."
                pm.remove_invalid_notes()
                original_bpm = 120 if original_bpm is None else original_bpm
                original_time_signature = TimeSignature.default() if original_time_signature is None else original_time_signature
                return MIDINoteList(set(pm.notes), original_bpm, original_time_signature)
        except:
            raise InvalidMIDIFileError(f"Failed to load pretty_midi object!")

    def transpose(self, target_key: KeyMode, original_key: KeyMode, inplace: bool = False) -> 'MIDINoteList':
        original_key = original_key.to_major()
        target_key = target_key.to_major()
        steps = original_key - target_key
        nl = self if inplace else self.copy()
        nl.move_key(steps)
        return nl

    def unify_velocity(self, velocity: int = 100, inplace: bool = False) -> 'MIDINoteList':
        '''
        Set the note velocity of all notes in a pretty_midi instrument to be a new value.
        '''
        nl = self if inplace else self.copy()
        for note in nl.notes:
            note.velocity = velocity
        return nl

    def analyze_key(self, is_drum: bool = False) -> Optional[KeyMode]:
        base_scale_index_count = Counter(note.pitch % 12 for note in self.notes)
        if not base_scale_index_count:
            return None
        if is_drum:
            return None
        total_notes = len(self.notes)
        best_key = KeyMode.BASE_SCALE[0]
        best_score = 0
        for k in KeyMode.BASE_SCALE:
            scale_pitches = KeyMode(k).get_scale_note_indexes()
            score = sum(base_scale_index_count[k] for k in base_scale_index_count if k in scale_pitches) / total_notes
            if score > best_score:
                best_key = k
                best_score = score
                if score == 1:
                    break
        return KeyMode(best_key)

    def pitch_quantize(self, key: KeyMode, inplace: bool = False) -> 'MIDINoteList':
        key = key.to_major()
        scale = key.scale
        nl = self if inplace else self.copy()
        for note in nl.notes:
            if pretty_midi.note_number_to_name(note.pitch)[:-1] not in scale:
                note.pitch = note.pitch + 1 if random.choice([True, False]) else note.pitch - 1
        return nl

    def ceil_bar_from_end_time(self) -> BarStepTick:
        bst = BarStepTick.sec2bst(self.end_time, self.bpm, self.time_signature)
        bar, step = bst.bar, bst.step
        if bar == 0:
            return BarStepTick(1, 0, 0, self.time_signature)
        elif step > 0 or bst.tick > 0:
            return BarStepTick(bar + 1, 0, 0, self.time_signature)
        else:
            return BarStepTick(bar, 0, 0, self.time_signature)

    def legato(self, inplace: bool = False) -> 'MIDINoteList':
        start_time = {}
        for note in self.notes:
            if note.start not in start_time:
                start_time[note.start] = []
            start_time[note.start].append(note)
        start_time = sorted(start_time.items(), key=lambda x: x[0])
        end_time = self.ceil_bar_from_end_time().to_seconds(self.bpm)
        notes = [MIDINote([note.pitch, note.start, end_time, note.velocity])
                 if tid + 1 == len(start_time)
                 else MIDINote([note.pitch, note.start, start_time[tid + 1][0], note.velocity])
                 for tid in range(len(start_time)) for note in start_time[tid][1]]
        nl = self if inplace else MIDINoteList(bpm=self.bpm, time_signature=self.time_signature)
        nl.notes = notes
        return nl

    def humanize(self, mode: tuple = (True, True), time_range: tuple = (-8, 12),
                 vel_range: tuple = (50, 120), inplace: bool = False) -> 'MIDINoteList':
        '''
        Humanize the midi notes to make it sounds more natural by moving each note separately by a random number of ticks
        selected from time_range and assign a random velocity according to vel_range. The number in time_range are tick numbers.
        The 'mode' tuple parameter indicates whether note time and note velocity will be modified respectively.
        '''
        if not isinstance(time_range, tuple) or abs(time_range[0]) > 23 or abs(time_range[1]) > 23 or time_range[1] < time_range[0] or time_range[0] % 1 != 0 or time_range[1] % 1 != 0:
            raise ValueError('time_range should be a pair tuple where each element should be an int in the range [0,23].')
        if not isinstance(vel_range, tuple) or not isinstance(vel_range[0], int) or not isinstance(vel_range[1], int) or vel_range[0] > vel_range[1] or not vel_range[0] in range(0, 128) or not vel_range[1] in range(0, 128):
            raise ValueError('vel_range should be a pair tuple where each element should be an int in the range [0,127].')
        notelist = self if inplace else self.copy()
        for note in notelist.notes:
            vel = random.randint(vel_range[0], vel_range[1])
            time = 1 / 192 * random.randint(time_range[0], time_range[1])
            if mode[1]:
                note.velocity = vel
            if mode[0]:
                note.start = note.start + time
                if note.duration > BarStepTick(0, 3, time_signature=self.time_signature).to_seconds(bpm=notelist.bpm):
                    note.end -= BarStepTick(0, 1, time_signature=self.time_signature).to_seconds(bpm=notelist.bpm)
        return notelist

    def to_pretty_midi_inst(self, name: str = "", program: int = 2, is_drum: bool = False):
        inst = pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)
        inst.notes.extend(note.to_pretty_midi_note() for note in self.notes)
        return inst

    def save_to_file(self, file_path:str):
        if not file_path.endswith((".mid", ".midi")):
            file_path += '.mid'
        pm = pretty_midi.PrettyMIDI(initial_tempo=self.bpm)
        pm.time_signature_changes = [self.time_signature.to_pretty_midi_time_signature()]
        pm.instruments.append(self.to_pretty_midi_inst())
        pm.write(file_path)
