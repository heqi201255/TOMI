# Author: Qi He heqi201255@icloud.com 2024

from tqdm import tqdm
import sys
import re
import os
import threading


class TOMIPrinter:
    _instance = None
    _lock = threading.Lock()  # This lock ensures thread-safety during instance creation
    ansi_colors = {
        "foreground": {
            "black": "\033[30m",
            "red": "\033[31m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "blue": "\033[34m",
            "magenta": "\033[35m",
            "cyan": "\033[36m",
            "white": "\033[37m",
            "bright_black": "\033[90m",  # gray
            "bright_red": "\033[91m",
            "bright_green": "\033[92m",
            "bright_yellow": "\033[93m",
            "bright_blue": "\033[94m",
            "bright_magenta": "\033[95m",
            "bright_cyan": "\033[96m",
            "bright_white": "\033[97m",
        },
        "background": {
            "black": "\033[40m",
            "red": "\033[41m",
            "green": "\033[42m",
            "yellow": "\033[43m",
            "blue": "\033[44m",
            "magenta": "\033[45m",
            "cyan": "\033[46m",
            "white": "\033[47m",
            "bright_black": "\033[100m",  # gray
            "bright_red": "\033[101m",
            "bright_green": "\033[102m",
            "bright_yellow": "\033[103m",
            "bright_blue": "\033[104m",
            "bright_magenta": "\033[105m",
            "bright_cyan": "\033[106m",
            "bright_white": "\033[107m",
        },
        "style": {
            "bold": "\033[1m",
            "dim": "\033[2m",
            "italic": "\033[3m",
            "underline": "\033[4m",
            "blink": "\033[5m",
            "inverted": "\033[7m",
            "hidden": "\033[8m",
            "strike_through": "\033[9m",
        }
    }
    resetB = "\033[49m"
    resetF = "\033[39m"
    resetS = "\033[22;23;24;25;27;28;29m"
    reset = "\033[0m"

    def __new__(cls, *args, **kwargs):
        # Double-checked locking to improve performance when _instance is already initialized
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-check
                    cls._instance = super(TOMIPrinter, cls).__new__(cls)
                    cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if not self.__initialized:
            self.logs = []
            self.progress = lambda iterable=None, desc=None, unit=None, total=None: tqdm(iterable, desc=desc, file=self, unit=unit, total=total)
            self._in_progress = False
            self.__initialized = True

    def __getattr__(self, name):
        if name.startswith("B"):
            if name[1:] in self.ansi_colors['background']:
                return self._add_background_color(name)
        if name.startswith('S'):
            if name[1:] in self.ansi_colors['style']:
                return self._add_style(name)
        elif name in self.ansi_colors['foreground']:
            return self._add_foreground_color(name)
        return super().__getattribute__(name)

    def __call__(self, text: str):
        self.print(text)

    def _add_background_color(self, name: str):
        def add_color(text, no_end=False):
            return self.color_func(text, self.ansi_colors['background'][name[1:]], self.resetB, no_end)
        return add_color

    def _add_foreground_color(self, name: str):
        def add_color(text, no_end=False):
            return self.color_func(text, self.ansi_colors['foreground'][name], self.resetF, no_end)
        return add_color

    def _add_style(self, name: str):
        def add_style(text, no_end=False):
            return self.color_func(text, self.ansi_colors['style'][name[1:]], self.resetS, no_end)
        return add_style

    def rgb(self, text: str, rgb: tuple[int, int, int], no_end=False) -> str:
        trim_text = text[: -len(self.resetF)] if text.endswith(self.resetF) else text
        r, g, b = rgb
        return f"\033[38;2;{r};{g};{b}m{trim_text}{"" if no_end else self.resetF}"

    def Brgb(self, text: str, rgb: tuple[int, int, int], no_end=False) -> str:
        trim_text = text[: -len(self.resetB)] if text.endswith(self.resetB) else text
        r, g, b = rgb
        return f"\033[48;2;{r};{g};{b}m{trim_text}{"" if no_end else self.resetB}"

    def color_func(self, text: str, color: str, reset: str, no_end: bool = False) -> str:
        trim_text = text[: -len(reset)] if text.endswith(reset) else text
        return f"{color}{trim_text}{"" if no_end else reset}"

    def len(self, text: str) -> int:
        return len(self.remove_ansi_codes(text))

    @staticmethod
    def remove_ansi_codes(text):
        ansi_escape_pattern = r'\033\[[0-9;]*m'
        clean_text = re.sub(ansi_escape_pattern, '', text)
        return clean_text

    @staticmethod
    def contains_ansi_codes(text):
        # Regular expression to match ANSI escape codes
        ansi_escape_pattern = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        # Search the string for ANSI escape codes
        return bool(ansi_escape_pattern.search(text))

    @staticmethod
    def extract_ansi_codes(text):
        # Regular expression to match ANSI escape codes
        ansi_escape_pattern = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        # Find all matches in the string
        return ansi_escape_pattern.findall(text)

    @staticmethod
    def remove_foreground_reset(text):
        return text.replace(TOMIPrinter.resetF, '')

    @staticmethod
    def remove_background_reset(text):
        return text.replace(TOMIPrinter.resetB, '')

    def write(self, message):
        if message.strip():
            m = message.split(":")
            m[0] = self.Bblue(self.bright_white(m[0]))
            m2 = m[1].split('|')
            m2[1] = re.sub(r'\S', self.red("▇"), m2[1])
            m[1] = "|".join(m2)
            msg = f"{m[0]}{"".join(m[1:])}"
            print(msg, end="")
            self.logs.append(self.remove_ansi_codes(str(msg)))
            self._in_progress = True

    def flush(self):
        sys.stdout.flush()

    def print(self, msg = "", end="\n"):
        if self._in_progress:
            self._in_progress = False
            print()
        print(msg, end=end)
        self.logs.append(self.remove_ansi_codes(str(msg)))

    def header(self, txt: str, background_func = None, text_func = None, sides_len: int = 8):
        if background_func is None:
            background_func = self.Bbright_white
        if text_func is None:
            text_func = self.black
        return background_func(text_func(self.Sbold(f"{'-' * sides_len}{self.Sitalic(txt)}{'-' * sides_len}")))

    def save_printed_msg_to_file(self, filename: str):
        """
        It's not a true "log", it just saves everything printed on the console to a file.
        """
        os.makedirs(filename, exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join([msg.strip() for msg in self.logs]))


