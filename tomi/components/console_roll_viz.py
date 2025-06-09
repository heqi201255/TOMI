from . import TOMIEnum, TOMIEnumDescriptor, BarStepTick, TimeSignature
from tomi import printer
import numpy as np
from dataclasses import dataclass
from pretty_midi import note_number_to_name
from IPython.display import display, HTML
from ansi2html import Ansi2HTMLConverter


class RollBlock:
    """
    This class represents a single block of a 2D graphical representation of a piano roll (PianoRoll class), there are
    three types: "Empty", "Duration", and "Start". Note that "Duration" type block should never be appeared without a
    "Start" block or another "Duration" block in front of it. This class provides some helper functions for checking
    the type of any RollBlock.
    """
    Empty = 0  # empty block
    Duration = 1  # note duration block
    Start = 2  # note attack (start) block
    NoteBlocks = (1, 2)


RollBlockMIDITypeConverter = {
    "Composite": 1,
    "Chord": 2,
    "Bass": 3,
    "Melody": 4,
    "Arp": 5,
    "Kick": 6,
    "ClapSnare": 7,
    "Hihat": 8,
    "Drummer": 9
}


_MIDITypeColor = {
    0: (0, 255, 255),
    1: (0, 255, 255),
    2: (255, 255, 0),
    3: (255, 0, 0),
    4: (255, 0, 255),
    5: (190, 0, 255),
    6: (0, 30, 210),
    7: (190, 232, 1),
    8: (17, 229, 42),
    9: (232, 0, 100)
}


class RollPrintingMode(TOMIEnum):
    """
    Simple Enum class used as the "mode" for PianoRoll to print into the console window.
    """
    Normal = TOMIEnumDescriptor('Normal')  # Print all key rows between the highest pitch and the lowest pitch of the PianoRoll inclusively.
    Compact = TOMIEnumDescriptor('Compact')  # Print only the key rows which includes at least one Midi note of the PianoRoll.
    Full = TOMIEnumDescriptor('Full')  # Print all key rows of the PianoRoll.


@dataclass
class RollSectionInfo:
    length: BarStepTick
    text: str
    color: tuple[int, int, int] = None


@dataclass
class RollBlockInfo:
    row_index: int
    start_index: int
    arr: np.ndarray
    converter: dict[int, str] = None


@dataclass
class RollBlockGroupInfo:
    blocks: list[RollBlockInfo]
    info: str = None
    color: tuple[int, int, int] = None
    bgcolor: tuple[int, int, int] = None

    @property
    def shape(self):
        shapes = np.array([(b.row_index + 1, b.start_index + b.arr.size) for b in self.blocks], dtype=np.int16)
        return (0, 0) if shapes.size == 0 else (shapes[:, 0].max(), shapes[:, 1].max())

    def to_num_roll(self, shape):
        block_roll = np.zeros(shape, dtype=np.int8)
        for block in self.blocks:
            block_roll[block.row_index, block.start_index:block.start_index + len(block.arr)] = block.arr
        return block_roll

    def convert_blocks(self):
        l = printer
        block_roll = {}
        for block in self.blocks:
            converter = block.converter if block.converter else ConsoleRollViz.DEFAULT_CONVERTER
            block_arr = np.vectorize(converter.get)(block.arr).astype('U50')
            block_roll[(block.row_index, block.start_index, block.start_index + len(block.arr))] = block_arr
            if self.color or self.bgcolor:
                zero_indices = [-1] + np.where(block_arr == ' ')[0].tolist() + [block_arr.size]
                indices = np.array([(zero_indices[i] + 1, zero_indices[i + 1] - 1) for i in range(len(zero_indices) - 1) if
                                   zero_indices[i] + 1 <= zero_indices[i + 1] - 1])
                if indices.size > 0:
                    start_indices, end_indices = indices[:, 0], indices[:, 1]
                    if self.color:
                        block_arr[start_indices] = np.char.add(l.rgb("", self.color, True), block_arr[start_indices])
                        block_arr[end_indices] = np.char.add(block_arr[end_indices], l.resetF)
                    if self.bgcolor:
                        block_arr[start_indices] = np.char.add(l.Brgb("", self.bgcolor, True), block_arr[start_indices])
                        block_arr[end_indices] = np.char.add(block_arr[end_indices], l.resetB)
        return block_roll


def in_jupyter_notebook():
    try:
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'  # Jupyter Notebook 或 JupyterLab
    except NameError:
        return False  # 非 IPython 环境（如脚本、终端等）


class ConsoleRollViz:
    DEFAULT_CONVERTER = {
                RollBlock.Start: '█',
                RollBlock.Duration: '▆',
                RollBlock.Empty: ' ',
            }

    def __init__(self,
                 roll: np.ndarray | list[RollBlockGroupInfo],
                 roll_color_matrix: np.ndarray = None,
                 roll_color_strength_matrix: np.ndarray = None,
                 channel_names: list = None,
                 row_length: int = None,
                 roll_info: str = None,
                 arrangement_info: list[RollSectionInfo] = None,
                 start_loc: BarStepTick = None,
                 mode: RollPrintingMode = None,
                 vbg_colors: dict[tuple, tuple[int, int, int]] = None,
                 hbg_colors: dict[tuple, tuple[int, int, int]] = None,
                 show_bars: bool = True,
                 reverse: bool = False,
                 channel_names_align: str = 'left',
                 time_signature: TimeSignature = None):
        self.l = printer
        self.show_bars = show_bars
        self.arrangement_info = arrangement_info
        self.roll_info = roll_info
        self.roll = roll
        self.roll_color_matrix = None
        self.roll_color_strength_matrix = None
        if roll_color_strength_matrix is not None:
            if isinstance(roll, np.ndarray):
                assert roll_color_strength_matrix.shape == roll.shape, f"'roll_color_strength_matrix' of shape {roll_color_strength_matrix.shape} does not match with 'roll' shape {roll.shape}"
                self.roll_color_strength_matrix = roll_color_strength_matrix
            else:
                self.l.print(self.l.yellow("'roll_color_strength_matrix' will not be used because 'roll' is not a numpy array."))
        if roll_color_matrix is not None:
            if isinstance(roll, np.ndarray):
                assert roll_color_matrix.shape == roll.shape, f"'roll_color_matrix' of shape {roll_color_matrix.shape} does not match with 'roll' shape {roll.shape}"
                self.roll_color_matrix = roll_color_matrix
            else:
                self.l.print(self.l.yellow("'roll_color_matrix' will not be used because 'roll' is not a numpy array."))
        self.channel_names = channel_names
        self.mode = RollPrintingMode.Full if mode is None else mode
        self.time_signature = TimeSignature.default() if time_signature is None else time_signature
        self.start_loc = BarStepTick() if start_loc is None else start_loc
        self.start_loc.time_signature = self.time_signature
        self.vbg_colors = vbg_colors
        self.hbg_colors = hbg_colors
        self.reverse = reverse
        assert channel_names_align in ['left', 'right', 'center']
        self.channel_names_align = channel_names_align
        # Get front padding length
        self.roll_length = row_length
        self.roll_shape: tuple[int, int] = self.get_roll_shape()
        self.lines = []
        self.front_length = 11
        self.roll_arr = np.full(self.roll_shape, " ", dtype=np.dtype('U50'))
        self.info_arr = np.full(self.roll_shape, " ", dtype=np.dtype('U50'))
        if isinstance(self.roll, np.ndarray):
            self.process_roll()
        else:
            self.process_roll_block_group()
        self.roll_arr = self.set_bar_ticks(self.roll_arr)
        self.paint_background_colors()

    def draw(self):
        channel_names, roll, info = self.get_channel_and_roll_for_printing()
        if self.reverse:
            roll = roll[::-1]
            info = info[::-1]
            channel_names = channel_names[::-1]
        self.front_length = max([11 if self.roll_info else 0, max([0] + [len(channel_name) + 2 for channel_name in channel_names])])
        top_unit, bottom_unit, labels = self.get_unit_lines()
        if self.roll_info and self.arrangement_info:
            self.lines.append(self.get_top_edge_line())
            self.get_info_line()
            self.lines.append(self.get_mid_edge_line())
            self.get_arrangement_line()
        elif self.roll_info:
            self.lines.append(self.get_top_edge_line())
            self.get_info_line()
        elif self.arrangement_info:
            self.lines.append(self.get_top_edge_line())
            self.get_arrangement_line()
        self.lines.append(top_unit)
        self.lines.extend(self.convert_roll_to_lines(channel_names, roll, info))
        self.lines.append(bottom_unit)
        self.lines.append(labels)
        if in_jupyter_notebook():
            ansi_html = Ansi2HTMLConverter(inline=True).convert("\n".join(self.lines), full=False)
            display(HTML(f'''
                    <div style="
                        overflow-x: auto;
                        white-space: pre;
                        font-family: monospace;
                        line-height: 1.1;
                        font-size: 16px;
                        padding: 4px;
                        margin: 0;
                    ">{ansi_html}</div>
                    '''))
        else:
            self.l.print("\n".join(self.lines))

    def get_roll_shape(self):
        if isinstance(self.roll, np.ndarray):
            shape = self.roll.shape
        else:
            shapes = np.array([bg.shape for bg in self.roll], dtype=np.int16)
            shape = (shapes[:, 0].max(), shapes[:, 1].max())
        if self.channel_names:
            assert len(self.channel_names) == shape[0]
        if self.roll_length:
            self.roll_length = max([self.roll_length, shape[1]])
            shape = (shape[0], self.roll_length)
        else:
            self.roll_length = shape[1]
        return shape

    def paint_background_colors(self):
        def paint_bg(txt: str):
            nonlocal color
            return self.l.Brgb(txt, color, no_end=True)
        if self.vbg_colors:
            for inds, color in self.vbg_colors.items():
                for ind in inds:
                    start, end = ind[0], ind[1]
                    self.roll_arr[:, start] = np.vectorize(paint_bg)(self.roll_arr[:, start])
                    self.roll_arr[:, end-1] = np.char.add(self.roll_arr[:, end-1], self.l.resetB)
                    self.info_arr[:, start] = np.vectorize(paint_bg)(self.info_arr[:, start])
                    self.info_arr[:, end-1] = np.char.add(self.info_arr[:, end-1], self.l.resetB)
        if self.hbg_colors:
            for inds, color in self.hbg_colors.items():
                if not inds: continue
                valid_inds = [i for i in inds if i < self.roll_shape[0]]
                self.roll_arr[valid_inds, 0] = np.vectorize(paint_bg)(self.roll_arr[valid_inds, 0])
                self.roll_arr[valid_inds, -1] = np.char.add(self.roll_arr[valid_inds, -1], self.l.resetB)

    def get_top_edge_line(self):
        return f"┏{'━' * self.front_length}┳{'━' * self.roll_length}┓"

    def get_mid_edge_line(self):
        return f"┣{'━' * self.front_length}╋{'━' * self.roll_length}┫"

    def get_info_line(self):
        header = "TOMIStudio".center(self.front_length)
        header = f"┃{self.l.Sitalic(self.l.Brgb(self.l.rgb(header, (255, 255, 255)), (66, 215, 245)))}┃"
        info = self.roll_info.ljust(self.roll_length) if len(self.roll_info) <= self.roll_length else self.roll_info[:self.roll_length]
        info_line = f"{header}{info}┃"
        self.lines.append(info_line)

    def get_line_front_part(self, text: str = ""):
        match self.channel_names_align:
            case 'left': return f"{text.ljust(self.front_length)}"
            case 'right': return f"{text.rjust(self.front_length)}"
            case _: return f"{text.center(self.front_length)}"

    def set_bar_ticks(self, roll: np.ndarray):
        def _set_bar_ticks(line: np.ndarray):
            condition = (np.arange(line.shape[-1])) % self.time_signature.steps_per_bar == 0
            line[condition] = np.char.replace(line[condition], " ", "|")
            return line
        return roll if not self.show_bars else np.apply_along_axis(_set_bar_ticks, axis=-1, arr=roll)

    def get_unit_lines(self):
        # Process upper and lower unit lines and bottom label line
        top = "━" * self.front_length
        top, top_last = (f"┏{top}┳", "┓") if self.roll_info is None and self.arrangement_info is None else (f"┣{top}╋", "┫")
        bottom = f"┗{"━" * self.front_length}┻"
        labels = f"{' ' * (self.front_length - 1)}"
        top_arr = np.array(['━'] * (self.roll_length + 1))
        bottom_arr = np.array(['━'] * (self.roll_length + 1))
        top_arr[self.time_signature.steps_per_beat::self.time_signature.steps_per_beat], top_arr[0::self.time_signature.steps_per_bar], top_arr[-1] = "┯", "┳", top_last
        bottom_arr[self.time_signature.steps_per_beat::self.time_signature.steps_per_beat], bottom_arr[0::self.time_signature.steps_per_bar], bottom_arr[-1] = "┷", "┻", "┛"
        label_arr = [" "] * (self.roll_length + 1)
        for ind, i in enumerate(range(1, self.roll_length + 2, self.time_signature.steps_per_bar)):
            label_arr[i:i + 3] = list(str(ind).rjust(3))
        labels += "".join(label_arr)
        return top + "".join(top_arr), bottom + "".join(bottom_arr), labels

    def get_channel_and_roll_for_printing(self):
        if self.mode == RollPrintingMode.Compact:
            keep_rids = np.flatnonzero(np.any(np.char.find(self.roll_arr, '█') >= 0, axis=1)).tolist()
            channel_names = [self.channel_names[row_id] for row_id in keep_rids] if self.channel_names else [note_number_to_name(row_id) for row_id in keep_rids]
            process_roll = self.roll_arr[keep_rids]
            process_info = self.info_arr[keep_rids]
        elif self.mode == RollPrintingMode.Normal:
            keep_rids = np.flatnonzero(np.any(np.char.find(self.roll_arr, '█') >= 0, axis=1)).tolist()
            channel_names = [self.channel_names[row_id] for row_id in range(keep_rids[0], keep_rids[-1] + 1)] if self.channel_names else [note_number_to_name(row_id) for row_id in range(keep_rids[0], keep_rids[-1] + 1)]
            process_roll = self.roll_arr[keep_rids[0]:keep_rids[-1] + 1]
            process_info = self.info_arr[keep_rids[0]:keep_rids[-1] + 1]
        else:
            channel_names = self.channel_names if self.channel_names else [note_number_to_name(row_id) for row_id in range(self.roll_shape[0])]
            process_roll = self.roll_arr
            process_info = self.info_arr
        return channel_names, process_roll, process_info

    def process_roll(self):
        def _color_block(block: str, color: int = None, strength: int = None) -> str:
            if block == ' ':
                return block
            color = (0, 255, 255) if color is None else _MIDITypeColor[color]
            color = (min(255, color[0]+strength), min(255, color[1]+strength), min(255, color[2]+strength)) if strength is not None else color
            return block if block == ' ' else self.l.rgb(block, color)
        self.roll_arr = np.vectorize(self.DEFAULT_CONVERTER.get)(self.roll).astype('U50')
        if self.roll_color_matrix is not None or self.roll_color_strength_matrix is not None:
            self.roll_arr = np.vectorize(_color_block)(self.roll_arr, self.roll_color_matrix, self.roll_color_strength_matrix).astype('U50')

    def process_roll_block_group(self):
        def _merge(b1: str, b2: str) -> str:
            l1, l2 = self.l.remove_ansi_codes(b1), self.l.remove_ansi_codes(b2)
            return b1 if (l1 == ' ' and l2 == ' ') or l1 == '█' or l2 == ' ' else b2
        for i, block_group in enumerate(self.roll):
            block_roll = block_group.convert_blocks()
            if block_group.info:
                for block in block_group.blocks:
                    start_indices = np.where(block.arr == RollBlock.Start)
                    if len(start_indices) > 0 and len(start_indices[0]) > 0:
                        ind = block.start_index + start_indices[0][0]
                        self.info_arr[block.row_index][ind:ind + len(block_group.info)] = list(block_group.info)
            for inds, arr in block_roll.items():
                self.roll_arr[inds[0], inds[1]:inds[2]] = np.vectorize(_merge)(self.roll_arr[inds[0], inds[1]:inds[2]], arr).astype('U50')

    def convert_roll_to_lines(self, channel_names, roll, info):
        lines = []
        clean_info_arr = np.vectorize(self.l.remove_ansi_codes)(info) if len(info) != 0 else np.array([], dtype=np.dtype('U50'))
        for tid in range(roll.shape[0]):
            s = f"┃{self.get_line_front_part(channel_names[tid])}┃{"".join(roll[tid])}┃"
            lines.append(s)
            if np.any(clean_info_arr[tid] != " "):
                lines.append(f"┃{self.get_line_front_part()}┃{''.join(self.set_bar_ticks(info[tid]))}┃")
        return lines

    def to_num_roll(self):
        def char_to_num(char: str):
            match char:
                case '█': return RollBlock.Start
                case ' ': return RollBlock.Empty
                case _:return RollBlock.Duration
        roll = np.vectorize(self.l.remove_ansi_codes)(self.roll_arr)
        roll = np.char.replace(roll, "|", " ")
        roll = np.vectorize(char_to_num, otypes=[np.int8])(roll)
        if roll.shape[0] != 128:
            roll = np.vstack((roll, np.zeros((128 - roll.shape[0], roll.shape[1]), dtype=np.int8)), dtype=np.int8)
        return roll

    def get_arrangement_line(self):
        # Process arrangement line
        arrangement_line = self.get_line_front_part(" Arrangement")
        sap_line = np.array([" "] * self.roll_length, dtype=np.dtype('U50'))
        current_bar = self.start_loc
        for sec in self.arrangement_info:
            start = current_bar.to_steps()
            current_bar += sec.length
            end = current_bar.to_steps()
            sap_line[end if start <= 0 else [start, end]] = "┃"
            sap_text = np.array(list(sec.text), dtype=np.dtype('U50'))
            if sec.color:
                sap_text[0] = self.l.Brgb(str(sap_text[0]), sec.color, True)
                sap_line[end] = sap_line[end] + self.l.reset
            sap_line[start + 1:start + 1 + len(sap_text)] = sap_text
        sap_line = "".join(sap_line)
        arrangement_line = f"┃{arrangement_line}┃{sap_line}┃"
        self.lines.append(arrangement_line)

