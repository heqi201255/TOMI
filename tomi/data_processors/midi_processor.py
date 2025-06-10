from tomi import (BarStepTick, MIDINoteList, MIDINote, QuantizationFactor, QuantizationMode, Groove, GrooveBlock,
               KeyMode, Mode, TimeSignature, RollBlock, ConsoleRollViz, RollPrintingMode, MIDIType, InvalidMIDIFileError, printer)
import re
from music21 import chord as m21chord
from typing import Union
import numpy as np
from numpy.typing import NDArray
import pretty_midi
from copy import deepcopy
from dataclasses import dataclass


@dataclass
class VNoteGroupInfo:
    """
    Vertical note group in the piano roll, can contain one or more notes (chord).
    """
    col_id: int  # Column index of the note group in the piano roll.
    group_length: int  # Number of columns of the note group in the piano roll.
    pitches: list[int]  # The note pitches of this note group.


class PianoRoll:
    """
    This class represents a piano roll just like what you see in a real DAW, it sees Midi notes as "blocks" in a 2D
    graphical representation where rows represent the "pitch" and columns represent the timing. By using this technique
    it can achieve many high-level functionalities like separating the chord notes and the melody notes.

    Class attributes:
    "block_sign": a dictionary that is used when printing the roll to the console window.
    "default_note_velocity": an integer between 0 and 128 that is used to set the velocity of a note/roll when no
    velocity information is provided.
    """
    default_note_velocity = 100

    def __init__(self,
                 midi: Union[MIDINoteList, NDArray[np.int8], 'PianoRoll'],
                 parent: 'MIDIProcessor',
                 midi_type: MIDIType = None,
                 roll_note_velocity: NDArray[np.int8] = None):
        """
        :param midi: can be a MIDINoteList or a raw Numpy 2D array representing the 'roll' or another 'PianoRoll' object.
        :param parent: a MIDIProcessor object that the 'PianoRoll' use some of its attributes in its methods.
        :param midi_type: specifies the Midi type of current piano roll.
        :param roll_note_velocity: a 2D Numpy array with the same shape of the piano roll, Each value is an integer from
        0 to 128 representing the Midi note velocity and each value is mapped to a block of the piano roll with the same
        indexes. Meaning you can find the note velocity of any notes in the piano roll in 'velocity_roll' just by
        passing the same indexes to 'velocity_roll'. If parameter 'midi' is a MIDINoteList object and 'roll_note_
        velocity' is None, it will automatically generate the 'velocity_roll' attribute.
        """
        self.parent = parent
        self.midi_type = midi_type
        self.roll_note_velocity = None
        self.in_scale_row_inds = None
        if isinstance(midi, MIDINoteList):
            # Parse the input MIDINoteList to a 2D array as the piano roll.
            self.roll, self.roll_note_velocity = self.render_piano_roll(midi)
        elif isinstance(midi, np.ndarray):
            # If the input midi is a 2D array, use it as the piano roll.
            assert isinstance(midi[0], np.ndarray), "Wrong Piano Roll format: midi type should be 2D np.ndarray."
            assert midi.dtype == np.int8, "Wrong Piano Roll format: midi block type should be np.int8."
            assert len(midi.shape) == 2 and midi.shape[0] == 128, "Wrong Piano Roll format: midi piano roll shape must be (128, x)."
            assert set(np.unique(midi)).issubset({RollBlock.Empty, RollBlock.Start, RollBlock.Duration}), "Wrong Piano Roll format: piano roll contains invalid value."
            self.roll = midi
        elif isinstance(midi, PianoRoll):
            # If the input midi is another PianoRoll object, copy the elements from it.
            self.midi_type = midi.midi_type
            self.roll_note_velocity = midi.roll_note_velocity.copy()
            self.roll = midi.roll.copy()
        else:
            raise ValueError('Wrong input midi type.')
        if roll_note_velocity is not None:
            # if velocity_roll is not None, it will override self.velocity_roll.
            assert roll_note_velocity.dtype == np.int8 and self.roll.shape == roll_note_velocity.shape, "Wrong 'velocity_roll', it must be the same shape of the piano roll and the data type should be numpy.int8"
            self.roll_note_velocity = roll_note_velocity
        elif self.roll_note_velocity is None:
            # If velocity_roll is None, and self.velocity_roll is also None, this happens when the input midi parameter is just a piano roll array without velocity information, then it will generate a velocity roll with default velocity.
            self.roll_note_velocity = np.full(self.roll.shape, self.default_note_velocity, dtype=np.int8)

    def __len__(self):
        """
        :return: the number of rows in the piano roll, 128 for full piano roll, meaning 128 pitches.
        """
        return len(self.roll)

    def __getitem__(self, item):
        """
        Pass to piano roll array's __getitem__ method.
        :param item: index or slice.
        :return: array or value.
        """
        return self.roll.__getitem__(item)

    def __setitem__(self, key, value):
        """
        Set value to piano roll given index or slice.
        :param key: index or slice.
        :param value: an array or an integer
        :return: None
        """
        self.roll[key] = value

    def __iter__(self):
        """
        Pass to piano roll arrays' __iter__ method.
        :return: an iterator object of piano roll array.
        """
        return iter(self.roll)

    @property
    def time_signature(self) -> TimeSignature:
        """
        Pass time signature object from MIDIProcessor parent.
        :return: TimeSignature.
        """
        return self.parent.time_signature

    @property
    def quantize_factor(self) -> QuantizationFactor:
        """
        Pass quantization factor from MIDIProcessor parent.
        :return: QuantizationFactor.
        """
        return self.parent.quantize_factor

    @property
    def key(self) -> KeyMode:
        return self.parent.key

    @property
    def shape(self) -> tuple:
        """
        :return: the shape of the piano roll.
        """
        return self.roll.shape

    def print_raw_roll(self):
        """
        Print the piano roll attribute as 2D numpy array to the console.
        Mainly used for debugging.
        :return: None
        """
        printer.print(self.roll)

    def transpose(self, target_key: KeyMode):
        """
        Transpose all notes in current piano roll to the target key.
        :param target_key: the target key of type KeyMode.
        :return: None
        """
        okey = self.key.to_major()
        target_key = target_key.to_major()
        steps = okey - target_key
        if steps == 0:
            return
        if (steps > 0 and np.any((self.roll[:-steps] == RollBlock.Start) | (self.roll[:-steps] == RollBlock.Duration))) or (
                steps < 0 and np.any((self.roll[-steps:] == RollBlock.Start) | (self.roll[-steps:] == RollBlock.Duration))):
            steps = target_key - okey
        self.roll = np.vstack((self.roll[-steps:], self.roll[:-steps]))
        self.roll_note_velocity = np.vstack((self.roll_note_velocity[-steps:], self.roll_note_velocity[:-steps]))

    def to_notelist(self, qf: QuantizationFactor = None) -> MIDINoteList:
        """
        Convert current piano roll to a MIDINoteList object.
        :param qf: a QuantizationFactor object, if None, will use parent MIDIProcessor's qf.
        :return: a MIDINoteList object.
        """
        note_list = []
        qf = self.parent.quantize_factor if qf is None else qf
        # flatten to 1D array to improve efficiency.
        flat_arr = self.roll.flatten()
        # Find all indexes of starting note blocks.
        start_indices = np.where(flat_arr == RollBlock.Start)[0]
        shape = self.shape
        for start in start_indices:
            # For each starting note block, find its end block.
            end = start + 1
            while end < (start // shape[-1] + 1) * shape[-1] and flat_arr[end] not in (RollBlock.Start, RollBlock.Empty):
                end += 1
            # Now convert the index back to its original index in the 2D piano roll.
            start_i, end_i = np.unravel_index(start, shape), np.unravel_index(end - 1, shape)
            # Convert the index (bid) to bst first, then convert it to seconds.
            note_stime = self.bid2bst(start_i[1], qf).to_seconds(self.parent.bpm)
            note_etime = self.bid2bst(end_i[1] + 1, qf).to_seconds(self.parent.bpm)
            # Create a MIDINote object and put it in the notelist.
            note_list.append(MIDINote([start_i[0], note_stime, note_etime, self.roll_note_velocity[start_i]]))
        return MIDINoteList(note_list, bpm=self.parent.bpm, time_signature=self.time_signature)

    def copy(self) -> 'PianoRoll':
        """
        Shallow copy current PianoRoll object.
        :return: a PianoRoll object.
        """
        return PianoRoll(self, self.parent, self.midi_type, self.roll_note_velocity)
    
    def render_piano_roll(self, midi: MIDINoteList) -> tuple[NDArray[np.int8], NDArray[np.int8]]:
        """
        Convert a MIDINoteList object to a piano roll 2D array and a velocity roll. Must quantize the MIDINoteList before,
        you can use MIDIProcessor's
        :param midi: a quantized MIDINoteList object.
        :return: a tuple of (piano_roll, velocity_roll).
        """
        def get_ind(time: float) -> int:
            """
            Find the corresponding column index of the piano roll given the time in seconds.
            :param time: seconds
            :return: int
            """
            # first convert the time to BST format.
            bst = BarStepTick.sec2bst(time, midi.bpm, self.time_signature)
            return bst.bar * self.parent.quantize_factor + bst.step // self.parent.quantize_factor.unit_steps

        def trim(total_bars: int):
            """
            Make the length of piano roll's column to fit the 'total_bars', for example, if quantize_factor is Q16 (16 columns for one bar) and
            current piano roll's column length is 18, the piano roll will be appended by 14 columns to make up 32 columns if 'total_bars' is 2.
            :param total_bars: number of bars.
            :return: None
            """
            nonlocal roll, velocity_roll
            if roll.shape[1] <= total_bars * self.parent.quantize_factor:
                # if current piano roll column length is shorter than desired column length, append it by empty columns.
                supply_len = total_bars * self.parent.quantize_factor - roll.shape[1]
                roll = np.hstack((roll, np.zeros((roll.shape[0], supply_len), dtype=np.int8)))
                velocity_roll = np.hstack((velocity_roll, np.zeros((velocity_roll.shape[0], supply_len), dtype=np.int8)))
            else:
                # if current piano roll column length is longer than desired column length, remove the extra columns.
                remove_len = roll.shape[1] - total_bars * self.parent.quantize_factor
                roll = roll[:, -remove_len]
                velocity_roll = velocity_roll[:, -remove_len]
        # get the number of columns of the piano roll, where each column represents:
        # 1. 1 step if 'self.parent.quantize_factor' is QuantizeFactor.Step,
        # 2. 2 steps if 'self.parent.quantize_factor' is QuantizeFactor.HalfBeat (for 4/4 time signature),
        # 3. 4 steps if 'self.parent.quantize_factor' is QuantizeFactor.Beat (for 4/4 time signature),
        # 4. 8 steps (half bar) if 'self.parent.quantize_factor' is QuantizeFactor.HalfBar (for 4/4 time signature),
        # 5. 16 steps (1 bar) if 'self.parent.quantize_factor' is QuantizeFactor.Bar (for 4/4 time signature),
        length = get_ind(midi.end_time)
        # create an empty piano roll with shape (128, num_of_columns), 128 means there are 128 keys in total for midi.
        roll = np.zeros((128, length), dtype=np.int8)
        # create an empty velocity roll with same shape of piano roll, and fill it with default note velocity values.
        velocity_roll = np.full((128, length), self.default_note_velocity, dtype=np.int8)
        for note in midi.notes:
            # find the start column index and end column index of the note.
            starti, endi = get_ind(note.start), get_ind(note.end)
            # Make the first block in the row to be a Start block to represent the note start.
            roll[note.pitch, starti] = RollBlock.Start
            # Make the reset blocks of the note to be Duration blocks.
            roll[note.pitch, starti + 1: endi][roll[note.pitch, starti + 1: endi] == RollBlock.Empty] = RollBlock.Duration
            # Change the corresponding blocks in velocity roll to be the value of this note's velocity.
            velocity_roll[note.pitch, starti: endi] = note.velocity
        # now get the number of full bars represented of current piano roll.
        bar = int(length / self.parent.quantize_factor)
        # get the extra steps in current piano roll.
        step = (length - (bar * self.parent.quantize_factor)) * self.parent.quantize_factor.unit_steps
        # make both rolls to be at least 1 bar long, if there are extra steps, append columns to rolls to make the roll column length represent exact full bars.
        trim(1 if bar == 0 else bar if step == 0 else bar + 1)
        return roll, velocity_roll

    def get_compact_roll(self) -> dict[int, NDArray[np.int8]]:
        """
        Get the compact version of piano roll that only rows with notes are included.
        :return: dict[row_index: row array]
        """
        return {row_id: self.roll[row_id].copy() for row_id in range(128) if self.contain_note(self.roll[row_id])}

    def get_scale_row_ids(self, key: KeyMode = None) -> list[int]:
        """
        Find the row indexes that are in the key's scale, each row represents a midi pitch.
        :param key: KeyMode instance, if none, will choose to use self.key.
        :return: a list of row indexes.
        """
        key = self.key if key is None else key
        if key is None: return []
        scale = key.scale
        return [rid for rid in range(128) if pretty_midi.note_number_to_name(rid)[:-1].rstrip("-") in scale]

    def get_full_width_roll(self) -> NDArray[np.int8]:
        """
        Stretch the piano roll array's column length to make it 16 columns a bar. Used for printing.
        :return: The stretched piano roll array.
        """
        process_roll = self.roll.copy()
        quantize_multiplier = int(self.time_signature.steps_per_bar / self.quantize_factor)
        # find all indexes of Start blocks.
        mask = np.where(process_roll == RollBlock.Start)
        # now get the new Start block indexes for each note.
        row_inds, col_inds = mask[0], mask[1] * quantize_multiplier
        # use np.repeat to stretch the 2D array.
        process_roll = np.repeat(process_roll, quantize_multiplier, axis=1)
        # because it is stretched, there can be multiple continuous Start blocks, to fix it, change all Start blocks to Duration blocks.
        process_roll[process_roll == RollBlock.Start] = RollBlock.Duration
        # then use the new Start block indexes to change these blocks back to Start blocks.
        process_roll[row_inds, col_inds] = RollBlock.Start
        return process_roll

    def get_full_width_velocity(self) -> NDArray[np.int8]:
        """
        Stretch the velocity roll array's column length to make it 16 columns a bar. Used for printing.
        :return: The stretched velocity roll array.
        """
        process_velocity = self.roll_note_velocity.copy()
        quantize_multiplier = int(self.time_signature.steps_per_bar / self.quantize_factor)
        # use np.repeat to stretch the 2D array, we don't need to change the values of the velocity array.
        process_velocity = np.repeat(process_velocity, quantize_multiplier, axis=1)
        return process_velocity

    def print_roll(self, mode: RollPrintingMode = None, scale_shadow: bool = True,
                   show_bar: bool = True, show_velocity: bool = True, msg: str = ""):
        """
        Print the piano roll array on terminal for better visualization.
        :param mode: RollPrintingMode.Normal only shows rows between the highest note and the lowest note; RollPrintingMode.Compact only shows rows which contains notes, RollPrintingMode.Full shows all 128 pitch rows.
        :param scale_shadow: whether to add background color for rows (keys) in the scale.
        :param show_bar: whether to add vertical line to indicate bars.
        :param show_velocity: whether to color the note according to its velocity, if True, notes with higher velocity will have stronger color.
        :param msg: the title being printed above the piano roll.
        :return: None
        """
        if len(self.roll) == 0:
            printer.print(f"-------------------Empty Roll{f': {msg}' if msg else ''}-------------------")
            return
        if scale_shadow and self.in_scale_row_inds is None:
            self.in_scale_row_inds = self.get_scale_row_ids()
        header = [(f"**************************************************************** {msg} "
                   f"****************************************************************")] if msg else []
        bst_length = self.parent.ceil_bst
        mode = RollPrintingMode.Compact if mode is None else mode
        # show meta information of MIDI
        header.append(
            f"View: {mode.value}, Type: {self.midi_type.name if self.midi_type is not None else "Unknown"}, "
            f"MIDI: {self.parent.name}, TimeSignature: {self.time_signature.numerator}/{self.time_signature.denominator}\n"
            f"Quantization Mode: {self.parent.quantize_mode.value}"
            f", Quantization Factor: {self.parent.quantize_factor.unit_value}, Length: {bst_length.bar}bars "
            f"{bst_length.step}steps {bst_length.tick}ticks \n"
            f"key: {f'{self.key.to_major().__str__()}/{self.key.to_minor().__str__()}' if self.key is not None else 'Unknown'}")
        printer.print("\n".join(header))
        process_roll = self.get_full_width_roll()
        tr = ConsoleRollViz(roll=process_roll,
                            roll_color_strength_matrix=self.get_full_width_velocity() if show_velocity else None,
                            hbg_colors={tuple(self.in_scale_row_inds): (100, 100, 100)},
                            mode=mode,
                            show_bars=show_bar,
                            reverse=True,
                            channel_names_align='right',
                            time_signature=self.time_signature)
        tr.draw()

    def contain_note(self, r: NDArray[np.int8], note_type: Union[np.int8, int, list[np.int8 | int], NDArray[np.int8]] = None) -> bool:
        """
        Check whether a piano roll or a row of piano roll contains certain note(s).
        :param r: can be a piano roll (2d list) or a single row of piano roll (1d list)
        :param note_type: can be int or list. Check self.block_type for detail.
        :return: True if the given roll or row contains any note with such note_type. False otherwise.
        """
        if note_type is None:
            return np.any((r == RollBlock.Start) | (r == RollBlock.Duration))
        elif isinstance(note_type, np.int8):
            return np.any(r == note_type)
        elif isinstance(note_type, (list, np.ndarray)):
            return np.any(np.isin(r, note_type))
        return False

    def legato(self, skip_empty_bars_num: int = 2):
        """
        Legato all notes, stretch the duration of each note until its end time reaches the start time of another note.
        For long MIDI that contains empty bars, will not stretch the note across these empty bars.
        :param skip_empty_bars_num: How many continuous empty bars should be considered to be ignored.
        :return: None
        """
        begin_bid = self.bst2bid(self.parent.active_range[0])
        end_bid = self.bst2bid(self.parent.active_range[1])
        skip_blocks = self.bst2bid(BarStepTick(skip_empty_bars_num, time_signature=self.time_signature))
        start_block_inds = np.where(self.roll == RollBlock.Start)
        start_time: list = list(set(start_block_inds[1]))
        i = 0
        # for all empty regions that met the 'skip_empty_bars_num' requirements, add the start time of the region to 'start_time' list.
        # So that it will be treated as columns with Start blocks.
        while i <= self.shape[1] - skip_blocks:
            # Check if the next N columns contain all zeros
            if np.all(self.roll[:, i:i + skip_blocks] == 0):
                start_time.append(i)
                i += skip_blocks
            else:
                i += 1
        start_time.sort()
        for rid, cid in zip(*start_block_inds):
            end_ind = start_time.index(cid) + 1
            if end_ind == 1 and cid != begin_bid:
                self.roll[rid, begin_bid] = RollBlock.Start
                self.roll[rid, begin_bid + 1: cid + 1] = RollBlock.Duration
                self.roll_note_velocity[rid, begin_bid: cid + 1] = self.roll_note_velocity[rid, cid]
            if end_ind == len(start_time):
                if cid + 1 < end_bid:
                    self.roll[rid, cid + 1: end_bid] = RollBlock.Duration
                    self.roll_note_velocity[rid, cid + 1: end_bid] = self.roll_note_velocity[rid, cid]
            else:
                self.roll[rid, cid + 1: start_time[end_ind]] = RollBlock.Duration
                self.roll_note_velocity[rid, cid + 1: start_time[end_ind]] = self.roll_note_velocity[rid, cid]

    def bst2bid(self, bst: BarStepTick, qf: QuantizationFactor = None) -> int:
        """
        Convert BST instance to the corresponding block index (column index) of a 2D array.
        :param bst: BarStepTick
        :param qf: QuantizationFactor
        :return: block index (column index)
        """
        qf = self.quantize_factor if qf is None else qf
        return bst.to_steps() // qf.unit_steps

    def bid2bst(self, bid: int, qf: QuantizationFactor = None) -> BarStepTick:
        """
        Convert a block index (column index) to the corresponding BST object.
        :param bid: block index (column index)
        :param qf: QuantizationFactor
        :return: BarStepTick
        """
        assert bid >= 0
        qf = self.parent.quantize_factor if qf is None else qf
        return BarStepTick.step2bst(bid * qf.unit_steps, time_signature=self.time_signature)

    def reform_root_notes(self, dividers: list = None):
        """
        Reorganize the root notes, make all root notes' start and end location to be on bars.
        :param dividers: the specified separation indexes of root notes.
        :return: None
        """
        qf = self.parent.quantize_factor
        if dividers is None:
            dividers = [qf* (2 * i) for i in range(1, int(self.parent.ceil_bst.bar / 2))]
        bar_dividers = [qf * i for i in range(1, self.parent.ceil_bst.bar)]
        col_notes_count = np.count_nonzero(self.parent.piano_roll.roll == RollBlock.Start, axis=0)
        for row in self.roll:
            f = False
            for cid in range(self.shape[1]):
                if row[cid] == RollBlock.Start:
                    if f and cid not in dividers:
                        if not (cid in bar_dividers and col_notes_count[cid] >= 3):
                            row[cid] = RollBlock.Duration
                    else:
                        f = True
                elif row[cid] == RollBlock.Empty:
                    f = False

    def get_col_notes(self, col: int, note_scope: Union[list[np.int8 | int], np.int8] = None) -> NDArray[np.int8]:
        """
        Given a column index, get the row indexes that contains note block specified in 'note_scope'.
        :param col: column index.
        :param note_scope: the specified note block types (integers).
        :return: an array of row indexes.
        """
        note_scope = [RollBlock.Start, RollBlock.Duration] if note_scope is None else note_scope
        return np.flatnonzero(np.isin(self.roll[:, col], note_scope)) if isinstance(note_scope, list) else np.flatnonzero(self.roll[:, col] == note_scope)

    def _preprocess_remove_very_top(self, top_thresh: int):
        """
        Preprocess for parsing chord notes, remove notes on higher pitch rows that are too far from its lower pitches (distance >= top_thresh), these notes will be considered as melody notes.
        :param top_thresh: the threshold value.
        :return: None
        """
        self.roll = self.roll[::-1]
        traversed_col = []
        for cid in range(self.shape[1]):
            top = None
            if cid in traversed_col:
                continue
            for rid in range(self.shape[0]):
                if self.roll[rid, cid] in RollBlock.NoteBlocks:
                    if top is not None:
                        if abs(rid - top) >= top_thresh:
                            indexes = self.remove_note(top, cid)
                            traversed_col.extend(indexes)
                        break
                    elif self.roll[rid, cid] == RollBlock.Start:
                        top = rid
        self.roll = self.roll[::-1]

    def _postprocess_remove_very_top(self, top_thresh: int = 7, loop: int = 3):
        """
        Postprocess for parsing chord notes, this time focus on the continuity pitch level of notes, remove notes upper the threshold.
        You may adjust 'loop' and 'top_thresh' to get better result.
        :param top_thresh: threshold value.
        :param loop: how many times to check the roll, check only once may not remove all unneeded notes.
        :return: None
        """
        def get_prev_top(col_index: int) -> Union[None, int]:
            if col_index == 0:
                return None
            for r in range(self.shape[0]):
                if self.roll[r, col_index] in RollBlock.NoteBlocks:
                    return r
            return None

        def get_next_top(row_index: int, col_index: int) -> Union[None, int]:
            try:
                for c in range(col_index + 1, self.shape[1]):
                    r_indices = np.where(self.roll[:, c] == RollBlock.Start)[0]
                    for r in r_indices:
                        if r == row_index:
                            break
                        else:
                            return r
                return None
            except:
                return None

        self.roll = self.roll[::-1]
        median_height, col_heights = self.get_chord_median_height()
        for _ in range(loop):
            traversed_col = set()
            for rid in range(self.shape[0]):
                found_before = False
                if len(traversed_col) == self.shape[1]:
                    break
                for cid in range(self.shape[1]):
                    if cid in traversed_col or self.roll[rid, cid] != RollBlock.Start:
                        continue
                    too_high = col_heights[cid] > median_height + 7
                    prev_top = get_prev_top(cid)
                    next_top = get_next_top(rid, cid)
                    traversed_col.update(self._get_note_indexes(rid, cid))
                    if ((next_top is not None and next_top - rid >= top_thresh) or
                        (prev_top is not None and prev_top - rid >= top_thresh)) and too_high and not found_before:
                        self.remove_note(rid, cid)
                        continue
                    found_before = True
        self.roll = self.roll[::-1]

    def get_chord_median_height(self) -> tuple[int, list]:
        """
        Get the median height and each individual height of all chords in the piano roll.
        :return: tuple[medium height value, list of heights of each chord]
        """
        heights = [self.get_chord_height(col) for col in self.roll.T]
        return np.median(heights), heights

    def get_chord_height(self, col: NDArray[np.int8]) -> int:
        """
        Get single chord height given the column index of the piano roll.
        :param col: column index.
        :return: chord height.
        """
        tb = self.get_chord_top_bottom_row_indexes(col)
        return abs(tb[0] - tb[1]) + 1 if tb else 0

    def add_notes(self, notes: Union[int, list, tuple, NDArray[np.int8]], start: int, length: int,
                  velocities: Union[int, list, tuple, NDArray[np.int8]] = None):
        """
        Add single/multiple notes to the piano roll, if 'velocities' is specified, also add velocity values to velocity roll.
        :param notes: note pitches.
        :param start: start column index.
        :param length: length of note in terms of piano roll columns.
        :param velocities: velocity value or list of velocities corresponding to each note.
        :return: None
        """
        if velocities is None:
            velocities = self.default_note_velocity
        if isinstance(notes, (tuple, list, np.ndarray)):
            if isinstance(velocities, (tuple, list, np.ndarray)):
                assert len(notes) == len(velocities), "Shape of 'notes' and 'velocities' do not match!"
                for i in range(len(notes)):
                    self.add_notes(notes[i], start, length, velocities[i])
            else:
                for n in notes:
                    self.add_notes(n, start, length, velocities)
            return
        self.roll[notes, start] = RollBlock.Start
        self.roll[notes, start + 1:start + length] = RollBlock.Duration
        self.roll_note_velocity[notes, start: start + length] = velocities

    def remove_note(self, rid: int, cid: int) -> list:
        """
        Remove the note given a row index and column index in piano roll. The corresponding value in piano_roll[rid, cid] must be a Start block.
        :param rid: row index.
        :param cid: column index.
        :return: a list of column indexes of the removed note had occupied before.
        """
        indexes = self._get_note_indexes(rid, cid)
        self.roll[rid, indexes] = RollBlock.Empty
        self.roll_note_velocity[rid, indexes] = self.default_note_velocity
        return indexes

    def _get_note_indexes(self, rid: int, cid: int) -> list:
        """
        Given the row index and column index of a Start block in the piano roll, the note may have Duration blocks after this block, get all block column indexes belongs to this note.
        :param rid: row index.
        :param cid: column index.
        :return: the list of all column indexes of this note.
        """
        row = self.roll[rid]
        assert row[cid] == RollBlock.Start, "roll[rid, cid] is not a valid 'RollBlock.Start' block!"
        col_indexes = [cid]
        current = cid + 1
        while current < self.shape[1]:
            if row[current] == RollBlock.Duration:
                col_indexes.append(current)
            else:
                break
            current += 1
        return col_indexes

    def get_chord_top_bottom_row_indexes(self, col: NDArray[np.int8] | int) -> Union[tuple[int, int], None]:
        """
        Given a column or a column index that the column may contain a chord, find the top row index (highest pitch) and bottom row index (lowest pitch).
        :param col: column array or column index.
        :return: tuple[top row index, bottom row index].
        """
        if isinstance(col, int):
            col = self.roll[:, col]
        rows = np.flatnonzero((col == RollBlock.Start) | (col == RollBlock.Duration))
        top, bottom = (rows[-1], rows[0]) if len(rows) > 0 else (None, None)
        return (top, bottom) if bottom is not None and top is not None else None # top>=bottom

    def get_piano_roll_metrics(self):
        """
        Calculate and return the length, dynamic, complexity metrics of the piano roll.
        :return: dict[zm(length), zd(dynamic), zs(complexity)].
        """
        is_chord = self.midi_type == MIDIType.Chord
        zm = self.piano_roll_length_metric()  # length in bars
        zd = self.piano_roll_dynamic_metric(is_chord)  # average height
        zs = self.piano_roll_complexity_metric(zd, zm, is_chord)  # average note density
        return {"zm": zm, "zd": round(zd, 2), "zs": round(zs, 2)}

    def piano_roll_length_metric(self) -> int:
        """
        Get the length in bars of the piano roll.
        :return: int
        """
        contain_notes_indexes = np.flatnonzero(np.any((self.roll.T == RollBlock.Start) | (self.roll.T == RollBlock.Duration), axis=1))
        start = contain_notes_indexes[0] if contain_notes_indexes.size > 0 else None
        end = contain_notes_indexes[-1] if contain_notes_indexes.size > 0 else None
        zm = int(self.bid2bst(end - start + 1).bar) if start != end else 0
        return zm

    def piano_roll_dynamic_metric(self, is_chord: bool = False) -> int:
        """
        Get the average 'height' of the piano roll.
        :param is_chord: whether the roll contains a chord or a chord progression.
        :return: int
        """
        if is_chord:
            _, heights = self.get_chord_median_height()
            heights = [x for x in heights if x > 0]
            try:
                zd = int(sum(heights) / len(heights))
            except ZeroDivisionError:
                zd = 0
        else:
            top = -1
            bottom = -1
            for rid in range(self.shape[0]):
                if any(self.roll[rid, cid] in RollBlock.NoteBlocks for cid in range(self.shape[1])):
                    if bottom == -1:
                        bottom = rid
                    top = rid
            zd = abs(top - bottom)
        return zd

    def piano_roll_complexity_metric(self, zd: int = None, zm: int = None, is_chord: bool = False) -> float:
        """
        Get the average note density of the piano roll.
        :param zd: dynamic metric.
        :param zm: length metric.
        :param is_chord: whether the roll contains a chord or a chord progression.
        :return: float
        """
        if is_chord:
            if zd is None:
                zd = self.piano_roll_dynamic_metric()
            note_nums = [len(c.pitches) for c in self.parent.note_groups]
            zs = 0 if zd == 0 else (sum(note_nums) / len(note_nums)) / zd
        else:
            if zm is None:
                zm = self.piano_roll_length_metric()
            note_nums = len(np.flatnonzero(self.roll == RollBlock.Start))
            zs = 0 if zm == 0 else note_nums / zm
        return zs

    def analyze_note_groups(self) -> list[VNoteGroupInfo]:
        """
        Parse the piano roll to find all notes and group them according to their start time.
        :return: list[VNoteGroupInfo]
        """
        attack_cols = np.flatnonzero(np.any(self.roll.T == RollBlock.Start, axis=1)).tolist()
        lens: list[int] = [attack_cols[i + 1] - attack_cols[i] for i in range(len(attack_cols) - 1)]
        if attack_cols: lens.append(self.shape[1] - attack_cols[-1])
        return [VNoteGroupInfo(col_id=cid, group_length=lens[chord_id], pitches=np.flatnonzero(self.roll.T[cid] == RollBlock.Start).tolist()) for chord_id, cid in enumerate(attack_cols)]
        # return [((cid, lens[chord_id]), np.flatnonzero(self.roll.T[cid] == RollBlock.Start).tolist()) for chord_id, cid in enumerate(attack_cols)]

    def apply_groove(self, groove: Groove, keep_original_len=True) -> MIDINoteList:
        """
        Given a Groove object, apply it to the piano roll and generate a new MIDINoteList object.
        The Groove must be compatible with the midi, by compatible it means the groove's progression_count must be divisible by the midi's progression count or a multiple of the midi's progression_count.'
        :param groove: Groove instance.
        :param keep_original_len:
        :return: MIDINoteList
        """
        assert groove.time_signature == self.time_signature, "Groove time signature is different."
        note_groups = self.analyze_note_groups()
        more_prog = len(note_groups) >= groove.get_progression_count()
        more_bars = self.parent.ceil_bst.bar >= groove.get_length()
        assert len(note_groups) % groove.get_progression_count() == 0 if more_prog else groove.get_progression_count() % len(
            note_groups) == 0, "Number of note groups does not match the given groove!"
        row_len = self.parent.ceil_bst.bar * 16 if more_bars or keep_original_len else groove.get_length() * 16
        new_roll = PianoRoll(np.zeros((self.shape[0], row_len), dtype=np.int8), self.parent, MIDIType.Chord)
        chord_ind, prelen = -1, 0
        loop = 1 if not more_prog else int(len(note_groups) / groove.get_progression_count())
        for _ in range(loop):
            in_note = False
            note_start = 0
            for cid, g_block in enumerate(groove.groove):
                if keep_original_len and cid + prelen >= self.parent.ceil_bst.bar * 16:
                    break
                if g_block != GrooveBlock.Sustain:
                    if in_note:
                        notes = note_groups[chord_ind].pitches
                        new_roll.add_notes(notes, note_start, cid + prelen - note_start,
                                           self.roll_note_velocity[notes, note_groups[chord_ind].col_id])
                    in_note = True
                    note_start = cid + prelen
                if g_block == GrooveBlock.Next:
                    chord_ind = (chord_ind + 1) % len(note_groups)
                elif g_block == GrooveBlock.Pause:
                    in_note = False
                if cid + 1 == len(groove) and in_note:
                    notes = note_groups[chord_ind].pitches
                    new_roll.add_notes(notes, note_start, cid + prelen + 1 - note_start,
                                       self.roll_note_velocity[notes, note_groups[chord_ind].col_id])
            prelen += len(groove)
        return new_roll.to_notelist()

    def analyze_midi_type(self) -> MIDIType:
        """
        Try to find the correct MIDIType based on the piano roll. For now, it can only check for Drummer, Bass, Melody, and Composite.
        :return: MIDIType
        """
        non_zero_counts_col = np.count_nonzero(self.roll, axis=0)
        non_zero_counts_row = np.count_nonzero(self.roll, axis=1)
        single_note_cols = np.count_nonzero(non_zero_counts_col == 1)
        all_note_cols = np.count_nonzero(non_zero_counts_col)
        all_note_rows = np.count_nonzero(non_zero_counts_row)
        row_indexes = np.flatnonzero(non_zero_counts_row)
        mid_row = sum([non_zero_counts_row[i] * i for i in row_indexes]) / np.sum(non_zero_counts_row)
        if all_note_rows <= 2:
            start_notes = np.count_nonzero(self.roll == RollBlock.Start)
            return MIDIType.Drummer if start_notes >= 4 else MIDIType.Bass
        elif single_note_cols / all_note_cols >= 0.9:
            return MIDIType.Bass if mid_row <= 50 else MIDIType.Melody
        return MIDIType.Composite


class MIDIProcessor:
    def __init__(self,
                 midi: str | MIDINoteList | pretty_midi.PrettyMIDI | pretty_midi.Instrument,
                 target_bpm: int | float = 120,
                 midi_type: MIDIType = None,
                 fit_beat: bool = False,
                 original_bpm: int | float = None,
                 original_time_signature: TimeSignature = None,
                 remove_short_notes: bool = False,
                 quantize_factor: QuantizationFactor = None,
                 quantize_mode: QuantizationMode = None,
                 name: str = None,
                 instrument_index: int = None,
                 force_key: KeyMode = None):
        """
        Given a midi file or a MIDINoteList instance, analyze and process it in many aspects.
        """
        if isinstance(midi, str):
            midi, midi_name = self.load_midi_file(midi, original_bpm, original_time_signature, instrument_index)
            self.name = midi_name if name is None else name
            self.instrument_index = instrument_index
        elif isinstance(midi, (pretty_midi.PrettyMIDI, pretty_midi.Instrument)):
            midi = MIDINoteList.load_pretty_midi(midi, instrument_index)
            self.instrument_index = instrument_index
            self.name = "MIDIProcessor" if name is None else name
        elif isinstance(midi, MIDINoteList):
            assert instrument_index is None, "'instrument_index' should only be used when 'midi' parameter is a file path or a prettymidi.PrettyMIDI object."
            self.name = "MIDIProcessor" if name is None else name
        self.time_signature = midi.time_signature
        self._midi_type = midi_type
        self.bpm = target_bpm
        self.midi = midi
        self.midi.bpm = self.bpm
        self._quantize_factor = QuantizationFactor.Step(self.time_signature) if quantize_factor is None else quantize_factor(self.time_signature)
        self._quantize_mode = QuantizationMode.StartEnd if quantize_mode is None else quantize_mode
        self.remove_short_notes = remove_short_notes
        self.quantized_midi = self.midi.quantize(self.quantize_factor, self.quantize_mode, self.remove_short_notes, inplace=False)
        self.fit_beat_toggle = fit_beat
        self.nl_offset = 0
        self.ceil_bst = self.quantized_midi.ceil_bar_from_end_time()
        self.key = force_key if force_key is not None and isinstance(force_key, KeyMode) else None
        self.piano_roll = self.chord_roll = self.bass_roll = self.melody_roll = None
        self.progression_nums = {'major': (), 'minor': ()}
        self.note_groups = self.progression_names = None
        self.groove = self.drum_groove = self.plain_groove = self.chord_groove = self.bass_groove = self.melody_groove = None
        self.active_range = None
        self.progression_count = 0
        self.update()
    
    def update(self):
        """
        Update the MIDIProcessor instance, reanalyze the piano rolls, grooves, etc.
        :return: None
        """
        def parse_chord_stuff():
            self.bass_roll = self.parse_bass_roll()
            self.chord_roll = self.parse_chord_roll()
            self.progression_nums = self.analyze_root_progression_numbers()
            self.note_groups = self.chord_roll.analyze_note_groups()
            self.progression_names = self.analyze_chord_name_progression()
            self.progression_count = len(self.progression_nums['major'])
            self.bass_groove = self.parse_bass_groove()
            self.chord_groove = self.parse_chord_groove()
        self.nl_offset = self.fit_beat() if self.fit_beat_toggle else 0
        self.piano_roll = PianoRoll(self.quantized_midi, self, self.midi_type)
        if self._midi_type is None:
            self._midi_type = self.parse_midi_type()
            self.piano_roll.midi_type = self._midi_type
        self.active_range = self.get_active_range()
        if self.key is None:
            self.key = self.midi.analyze_key(self._midi_type.is_drum())
        if self.midi_type.is_drum():
            self.note_groups = self.piano_roll.analyze_note_groups()
            self.groove = self.drum_groove = self.parse_drum_groove()
        elif self.is_type([MIDIType.Melody, MIDIType.Arp]):
            self.melody_roll = self.piano_roll
            self.note_groups = self.melody_roll.analyze_note_groups()
            self.groove = self.melody_groove = self.parse_midi_groove(self.melody_roll)
        elif self.is_type(MIDIType.Composite):
            parse_chord_stuff()
            self.groove = self.plain_groove = self.parse_plain_groove()
            self.melody_roll = self.parse_melody_roll()
            self.melody_groove = self.parse_midi_groove(self.melody_roll)
        elif self.is_type(MIDIType.Chord):
            parse_chord_stuff()
            self.groove = self.plain_groove = self.parse_plain_groove()
        elif self.is_type(MIDIType.Bass):
            self.bass_roll = self.parse_bass_roll()
            self.progression_nums = self.analyze_root_progression_numbers()
            self.note_groups = self.bass_roll.analyze_note_groups()
            self.groove = self.bass_groove = self.parse_midi_groove(self.bass_roll)
    
    def print_all(self, show_velocity_color: bool = True):
        if self.is_progression():
            self.piano_roll.print_roll(mode=RollPrintingMode.Compact, msg=f"Original MIDI (Q-{self.quantize_factor.name})", show_velocity=show_velocity_color)
            self.bass_roll.print_roll(mode=RollPrintingMode.Compact, msg="Bass MIDI", show_velocity=show_velocity_color)
            self.chord_roll.print_roll(mode=RollPrintingMode.Compact, msg="Chord MIDI", show_velocity=show_velocity_color)
            self.melody_roll.print_roll(mode=RollPrintingMode.Compact, msg="Melody MIDI", show_velocity=show_velocity_color)
            printer.print(f"Chord Progression: {self.progression_names}\n"
                          f"Type: {self.midi_type.name}\n"
                          f"Has melody: {self.has_melody()}\n"
                          f"Chord Metrics: {self.chord_roll.get_piano_roll_metrics()}\n"
                          f"Bass Metrics: {self.bass_roll.get_piano_roll_metrics()}\n"
                          f"Melody Metrics: {self.melody_roll.get_piano_roll_metrics()}\n"
                          f"\n"
                          f"Plain Groove: {self.plain_groove}\n"
                          f"Chord Groove: {self.chord_groove}\n"
                          f"Bass Groove: {self.bass_groove}\n"
                          f"Melody Groove: {self.melody_groove}\n")
        elif self.is_type(MIDIType.Bass):
            self.piano_roll.print_roll(mode=RollPrintingMode.Compact, msg=f"Original MIDI (Q-{self.quantize_factor.name})", show_velocity=show_velocity_color)
            self.bass_roll.print_roll(mode=RollPrintingMode.Compact, msg="Bass MIDI", show_velocity=show_velocity_color)
            printer.print(f"Type: {self.midi_type.name}\n"
                          f"Bass Metrics: {self.bass_roll.get_piano_roll_metrics()}\n"
                          f"\n"
                          f"Bass Groove: {self.bass_groove}\n")
        elif self.midi_type.is_drum():
            self.piano_roll.print_roll(mode=RollPrintingMode.Compact,
                                       msg=f"Original MIDI (Q-{self.quantize_factor.name})", show_velocity=show_velocity_color)
            printer.print(f"Type: {self.midi_type.name}\n"
                          f"\n"
                          f"Drum Groove: {self.drum_groove}\n")
        else:
            self.piano_roll.print_roll(mode=RollPrintingMode.Compact,
                                       msg=f"Original MIDI (Q-{self.quantize_factor.name})", show_velocity=show_velocity_color)
            printer.print(f"Type: {self.midi_type.name}\n"
                          f"\n"
                          f"Melody Groove: {self.melody_groove}\n")

    def has_melody(self) -> bool:
        return self.melody_roll is not None and self.melody_roll.piano_roll_length_metric() >= self.ceil_bst.bar / 2

    def transpose(self, key: KeyMode):
        self.midi.transpose(key, self.key, inplace=True)
        self.quantized_midi.transpose(key, self.key, inplace=True)
        self.key = key
        self.update()

    def move_octave(self, octave: int = 0):
        pitch_offset = octave * 12
        self.midi.move_key(pitch_offset)
        self.quantized_midi.move_key(pitch_offset)
        self.update()

    def legato(self):
        self.midi.legato(inplace=True)
        self.quantized_midi.legato(inplace=True)
        self.update()

    def get_key(self, mode: Mode | str = 'str') -> KeyMode:
        mode = Mode(mode) if isinstance(mode, str) else mode
        return self.key.to_mode(mode)

    def get_name(self) -> Union[str, None]:
        return self.name

    @property
    def midi_type(self) -> MIDIType:
        return self._midi_type

    @midi_type.setter
    def midi_type(self, midi_type: MIDIType):
        assert isinstance(midi_type, MIDIType)
        if self._midi_type != midi_type:
            self._midi_type = midi_type
            self.update()

    def is_progression(self):
        return self.midi_type == MIDIType.Chord or self.midi_type == MIDIType.Composite

    def get_notelist(self) -> MIDINoteList:
        return self.midi

    def get_quantized_notelist(self) -> MIDINoteList:
        return self.quantized_midi

    def get_chord_notelist(self) -> MIDINoteList:
        return self.chord_roll.to_notelist() if self.chord_roll is not None else MIDINoteList(bpm=self.bpm, time_signature=self.time_signature)

    def get_bass_notelist(self) -> MIDINoteList:
        return self.bass_roll.to_notelist() if self.bass_roll is not None else MIDINoteList(bpm=self.bpm, time_signature=self.time_signature)

    def get_melody_notelist(self) -> MIDINoteList:
        return self.melody_roll.to_notelist() if self.melody_roll is not None else MIDINoteList(bpm=self.bpm, time_signature=self.time_signature)

    def get_processor_by_type(self, midi_type: MIDIType):
        match midi_type:
            case MIDIType.Chord: return self.get_chord_processor()
            case MIDIType.Bass: return self.get_bass_processor()
            case MIDIType.Melody: return self.get_melody_processor()
            case MIDIType.Arp: return self.get_melody_processor()
            case _: return deepcopy(self)

    def get_chord_processor(self) -> 'MIDIProcessor':
        return MIDIProcessor(self.get_chord_notelist(), self.bpm, MIDIType.Chord, False,
                             remove_short_notes=self.remove_short_notes, quantize_factor=self.quantize_factor,
                             quantize_mode=self.quantize_mode, name=f"{self.name}_Chord", force_key=self.key)

    def get_bass_processor(self) -> 'MIDIProcessor':
        return MIDIProcessor(self.get_bass_notelist(), self.bpm, MIDIType.Bass, False,
                             remove_short_notes=self.remove_short_notes, quantize_factor=self.quantize_factor,
                             quantize_mode=self.quantize_mode, name=f"{self.name}_Bass", force_key=self.key)

    def get_melody_processor(self) -> 'MIDIProcessor':
        return MIDIProcessor(self.get_melody_notelist(), self.bpm, MIDIType.Melody, False,
                             remove_short_notes=self.remove_short_notes, quantize_factor=self.quantize_factor,
                             quantize_mode=self.quantize_mode, name=f"{self.name}_Melody", force_key=self.key)

    @property
    def quantize_factor(self) -> QuantizationFactor:
        return self._quantize_factor

    @quantize_factor.setter
    def quantize_factor(self, factor: QuantizationFactor):
        assert isinstance(factor, QuantizationFactor)
        if factor != self._quantize_factor:
            self._quantize_factor = factor
            self._quantize_factor.time_signature = self.time_signature
            self.quantized_midi = self.midi.quantize(self.quantize_factor, self.quantize_mode, self.remove_short_notes, inplace=False)
            self.update()

    @property
    def quantize_mode(self) -> QuantizationMode:
        return self._quantize_mode

    @quantize_mode.setter
    def quantize_mode(self, mode: QuantizationMode):
        assert isinstance(mode, QuantizationMode)
        if mode != self._quantize_mode:
            self._quantize_mode = mode
            self.quantized_midi = self.midi.quantize(self.quantize_factor, self.quantize_mode, self.remove_short_notes, inplace=False)
            self.update()

    def get_active_range(self):
        start = BarStepTick.sec2bst(self.quantized_midi.start_time, self.bpm, self.time_signature)
        end = BarStepTick.sec2bst(self.quantized_midi.end_time, self.bpm, self.time_signature)
        return start, end

    def is_type(self, midi_type: Union[MIDIType, list[MIDIType], tuple]):
        return self.midi_type == midi_type if isinstance(midi_type, MIDIType) else self.midi_type in midi_type

    @staticmethod
    
    def load_midi_file(midi_file: str, original_bpm: float | int = None, original_time_signature: TimeSignature = None, instrument_index: int = None) -> tuple[MIDINoteList, str]:
        try:
            midi_nl = MIDINoteList.load_midi_file(midi_file, instrument_index, original_bpm=original_bpm, original_time_signature=original_time_signature)
            name = midi_file if instrument_index is None else f"{midi_file}_{pretty_midi.PrettyMIDI(midi_file).instruments[instrument_index].name}"
            return midi_nl, name
        except:
            raise InvalidMIDIFileError(f"Failed to load file '{midi_file}'!")

    def parse_midi_type(self) -> MIDIType:
        if self.name:
            # Define type mappings
            type_keywords = {
                MIDIType.Arp: {'ARP', 'ARPS', 'ARPEGGIO'},
                MIDIType.Hihat: {'HIHAT', 'HAT', 'HATS', 'HIHATS', 'RIDE'},
                MIDIType.Kick: {'KICK', 'KICKS'},
                MIDIType.ClapSnare: {'SNARE', 'SNARES', 'CLAP', 'CLAPS'},
                MIDIType.Bass: {'BASS', 'BASSLINE'},
                MIDIType.Composite: {'CHORD', 'CHORDS', 'PROGRESSION'},
                MIDIType.Drummer: {'DRUMMER', 'DRUMKIT', 'MULTI', 'FILL', 'DRUMS'}
            }
            # Get filename tokens
            path_parts = self.name.rstrip(".mid").split("/")
            if len(path_parts) > 4 and path_parts[-4] == 'drummer': return MIDIType.Drummer
            name = "".join(path_parts[-2:]) if len(path_parts) >= 2 else "".join(path_parts)
            tokens = [x.upper() for x in re.split(r'-|\s|\*|_|/|&', name) if x]
            if 'BASS' in tokens:
                ti = tokens.index('BASS')
                if ti != 0 and tokens[ti-1] in {'FUTURE', 'DEEP', 'UK', 'ACID', 'HARD', 'MIAMI', 'TECH'}:
                    type_keywords.pop(MIDIType.Bass)
            tokens = set(tokens)
            # Check for each type
            for midi_type, keywords in type_keywords.items():
                if keywords & tokens:  # Set intersection
                    return midi_type
        return self.piano_roll.analyze_midi_type()
    
    def fit_beat(self) -> int:
        map_count = {i: 0 for i in range(self.time_signature.steps_per_bar)}
        note_bsts = (note.bst_location(self.bpm, self.time_signature) for note in self.quantized_midi)
        for nb in note_bsts:
            map_count[(self.time_signature.steps_per_bar - nb.step) % self.time_signature.steps_per_bar] += 1
        offset_steps = max(map_count, key=map_count.get)
        offset_sec = BarStepTick.step2sec(offset_steps, self.bpm)
        empty_bar = BarStepTick.sec2bst(self.midi.start_time + offset_sec, self.bpm, time_signature=self.time_signature).bar
        if empty_bar:
            extra_steps = empty_bar * self.time_signature.steps_per_bar
            offset_steps -= extra_steps
            offset_sec -= BarStepTick.step2sec(extra_steps, self.bpm)
        self.quantized_midi.time_offset(offset_sec)
        self.midi.time_offset(offset_sec)
        return offset_steps
    
    def parse_bass_roll(self):
        bass_roll = []
        velocity_roll = []
        started = False
        count = 0
        ended = False
        for rid, row in enumerate(self.piano_roll):
            if ended:
                bass_roll.append(np.zeros(self.piano_roll.shape[1], dtype=np.int8))
                velocity_roll.append(np.full(self.piano_roll.shape[1], PianoRoll.default_note_velocity, dtype=np.int8))
                continue
            bass_roll.append(row.copy())
            velocity_roll.append(self.piano_roll.roll_note_velocity[rid])
            if self.piano_roll.contain_note(row):
                if not started:
                    started = True
                    count += 1
                else:
                    count += 1
                    if count == 13:
                        ended = True
            else:
                if started:
                    count += 1
                    if count == 13:
                        ended = True
        bass_roll = PianoRoll(np.array(bass_roll, dtype=np.int8), self, MIDIType.Bass,
                              np.array(velocity_roll, dtype=np.int8))
        bass_roll.legato()
        for col in range(bass_roll.shape[1]):
            for row in range(bass_roll.shape[0]):
                if bass_roll[row, col] != RollBlock.Empty:
                    if row < bass_roll.shape[0] - 1:
                        bass_roll[row + 1:, col] = RollBlock.Empty
                        bass_roll.roll_note_velocity[row + 1, col] = PianoRoll.default_note_velocity
                        break
        bass_roll.reform_root_notes()
        return bass_roll
    
    def parse_chord_roll(self, top_thresh: int = 8):
        if not self.bass_roll:
            raise ValueError("Bass progression not found! Parse bass notes first.")
        chord_roll = self.bass_roll.copy()
        chord_roll.midi_type = MIDIType.Chord
        reference_roll = self.piano_roll.copy()
        reference_roll._preprocess_remove_very_top(top_thresh)

        def add_chord():
            nonlocal chord, duration
            if chord:
                best_one = sorted(chord.items(), key=lambda x: x[1][0], reverse=True)[0]
                chord_notes, velocities = best_one[0], best_one[1][1]
                chord_roll.add_notes(chord_notes, duration[0], len(duration), velocities)
            chord = {}
            duration = []

        for row in self.bass_roll:
            duration = []
            chord = {}
            started = False
            prev_col_notes = []
            for cid in range(self.bass_roll.shape[1]):
                if row[cid] == RollBlock.Start:
                    if started:
                        add_chord()
                    started = True
                if row[cid] == RollBlock.Empty:
                    started = False
                    add_chord()
                if started:
                    duration.append(cid)
                    notes = reference_roll.get_col_notes(cid, note_scope=RollBlock.Start).tolist()
                    dnotes = reference_roll.get_col_notes(cid, note_scope=RollBlock.Duration)
                    notes.extend(x for x in dnotes if x in prev_col_notes)
                    if len(notes) >= 3:
                        notes.sort()
                        prev_col_notes = notes
                        notes = tuple(notes)
                        if notes in chord.keys():
                            chord[notes][0] += 1
                        else:
                            velocities = reference_roll.roll_note_velocity[notes, cid]
                            chord[notes] = [1, velocities]
                    else:
                        prev_col_notes = []
                    if cid + 1 == self.bass_roll.shape[1]:
                        add_chord()
        chord_roll._postprocess_remove_very_top()
        return chord_roll
    
    def parse_melody_roll(self) -> PianoRoll:
        if self.is_type([MIDIType.Melody, MIDIType.Arp]):
            return self.piano_roll
        melody_roll = np.zeros(self.chord_roll.shape, dtype=np.int8)
        velocity_roll = np.full(self.chord_roll.shape, PianoRoll.default_note_velocity, dtype=np.int8)
        for cid in range(self.chord_roll.shape[1]):
            tb = self.chord_roll.get_chord_top_bottom_row_indexes(cid)
            if tb is None:
                continue
            top = tb[0]
            if top + 1 == self.chord_roll.shape[0]:
                continue
            for rid in range(top + 1, self.chord_roll.shape[0]):
                melody_roll[rid, cid] = self.piano_roll[rid, cid]
                velocity_roll[rid, cid] = self.piano_roll.roll_note_velocity[rid, cid]
        return PianoRoll(melody_roll, self, MIDIType.Melody, velocity_roll)
    
    def parse_drum_groove(self) -> Groove:
        g = np.ones(self.piano_roll.shape[1], dtype=np.int8)
        started = False
        for cid in range(self.piano_roll.shape[1]):
            if np.any(self.piano_roll[:, cid] == RollBlock.Start):
                if started:
                    g[cid] = GrooveBlock.Replay
                else:
                    g[cid] = GrooveBlock.Next
                    started = True
        return self._stretch_to_groove(g, self.midi_type, self.quantize_factor)

    def parse_midi_groove(self, roll: PianoRoll = None) -> Groove:
        if roll is None:
            roll = self.melody_roll
        g = np.zeros(self.piano_roll.shape[1], dtype=np.int8)
        prev_notes = None
        paused = False
        for cid, col in enumerate(roll.roll.T):
            notes = np.flatnonzero(col == RollBlock.Start)
            if notes.size > 0:
                g[cid] = GrooveBlock.Next if prev_notes is None or notes.shape != prev_notes.shape or np.any(
                    notes != prev_notes) else GrooveBlock.Replay
                prev_notes = notes
                paused = False
            elif not np.any(col == RollBlock.Duration) and prev_notes is not None:
                if not paused:
                    g[cid] = GrooveBlock.Pause
                    paused = True
        return self._stretch_to_groove(g, self.midi_type, self.quantize_factor)

    def parse_plain_groove(self) -> Groove:
        g = np.zeros(self.piano_roll.shape[1], dtype=np.int8)
        attack_rows = []
        relay_rows = []
        last = []
        for c in self.note_groups:
            if c.pitches != last:
                attack_rows.append(c.col_id)
            else:
                relay_rows.append(c.col_id)
            last = c.pitches
        g[attack_rows] = GrooveBlock.Next
        g[relay_rows] = GrooveBlock.Replay
        return self._stretch_to_groove(g, MIDIType.Chord, self.quantize_factor, is_plain_groove=True)

    def parse_bass_groove(self) -> Groove:
        g = np.zeros(self.piano_roll.shape[1], dtype=np.int8)
        pause = False
        for c in self.note_groups:
            rootid = c.pitches[0]
            start = c.col_id
            length = c.group_length
            first = True
            for cid in range(start, start + length):
                if self.piano_roll[rootid, cid] == RollBlock.Start:
                    pause = False
                    if first:
                        g[cid] = GrooveBlock.Next
                        first = False
                    else:
                        g[cid] = GrooveBlock.Replay
                elif self.piano_roll[rootid, cid] == RollBlock.Empty and not pause and not first:
                    g[cid] = GrooveBlock.Pause
                    pause = True
        return self._stretch_to_groove(g, MIDIType.Bass, self.quantize_factor)
    
    def parse_chord_groove(self) -> Groove:
        g = np.zeros(self.piano_roll.shape[1], dtype=np.int8)
        pause = False
        for c in self.note_groups:
            first = True
            chords = c.pitches
            start = c.col_id
            length = c.group_length
            filter = True
            if len(chords) <= 3:
                filter = False
            for cid in range(start, start + length):
                start_note_hits = np.count_nonzero(self.piano_roll[chords, cid] == RollBlock.Start)
                note_hits = np.count_nonzero(
                    np.isin(self.piano_roll[chords, cid], [RollBlock.Start, RollBlock.Duration]))
                is_start = (start_note_hits >= len(chords) * 0.7) if filter else (start_note_hits == len(chords))
                is_note = (note_hits >= len(chords) * 0.7) if filter else (note_hits == len(chords))
                if is_start:
                    pause = False
                    if first:
                        g[cid] = GrooveBlock.Next
                        first = False
                    else:
                        g[cid] = GrooveBlock.Replay
                else:
                    if not is_note and not pause and not first:
                        g[cid] = GrooveBlock.Pause
                        pause = True
        return self._stretch_to_groove(g, MIDIType.Chord, self.quantize_factor)

    def _stretch_to_groove(self, groove: NDArray[np.int8], midi_type: MIDIType, qf: QuantizationFactor, is_plain_groove: bool = False) -> Groove:
        """
        :param groove:
        :param midi_type:
        :param qf: quantize factor
        :param is_plain_groove:
        :return:
        """
        g = []
        for b in groove:
            if b in GrooveBlock.ActionBlocks:
                g.append(b)
                g.extend(GrooveBlock.Sustain for _ in range(int(self.time_signature.steps_per_bar / qf) - 1))
            else:
                g.extend(b for _ in range(int(self.time_signature.steps_per_bar / qf)))
        return Groove(np.array(g, dtype=np.int8), midi_type, is_plain_groove, self.time_signature)
    
    def analyze_root_progression_numbers(self) -> dict:
        if self.midi_type.name not in ("Composite", "Chord", "Bass") or self.key is None:
            return {'major': (), 'minor': ()}
        start_blocks = [rid for col in self.bass_roll.roll.T for rid, block in enumerate(col) if block == RollBlock.Start]
        notes = [pretty_midi.note_number_to_name(x)[:-1] for x in start_blocks]
        maj_scale = self.key.to_major().scale
        min_scale = self.key.to_minor().scale
        maj_prog = [maj_scale.index(x) + 1 for x in notes if x in maj_scale]
        min_prog = [min_scale.index(x) + 1 for x in notes if x in min_scale]
        if maj_prog and len(maj_prog) > 4:
            i = int(len(maj_prog) / 2)
            while i >= 3:
                if maj_prog[:i] == maj_prog[i:]:
                    maj_prog = maj_prog[:i]
                    min_prog = min_prog[:i]
                else:
                    break
                i = int(i / 2)
        return {"major": tuple(maj_prog), "minor": tuple(min_prog)}

    def analyze_chord_name_progression(self) -> list:
        if self.midi_type.name not in ("Composite", "Chord"):
            return []
        chord_names = []
        for p in self.note_groups:
            ch = m21chord.Chord(p.pitches)
            chord_names.append(ch.pitchedCommonName if ch else "N/A")
        return chord_names
