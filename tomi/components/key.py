from typing import Union
from . import TOMIEnum, TOMIEnumDescriptor


class Mode(TOMIEnum):
    Major = TOMIEnumDescriptor('Major')
    Minor = TOMIEnumDescriptor('Minor')

    def set_value(self, mode: str, *args):
        alias_map = {'Maj': 'Major', 'Min': 'Minor', 'M': 'Minor'}
        super(Mode, self).set_value(mode.lower().capitalize(), alias_map)


class Key(TOMIEnum):
    C = TOMIEnumDescriptor('C')
    Cs = TOMIEnumDescriptor('C#')
    D = TOMIEnumDescriptor('D')
    Ds = TOMIEnumDescriptor('D#')
    E = TOMIEnumDescriptor('E')
    F = TOMIEnumDescriptor('F')
    Fs = TOMIEnumDescriptor('F#')
    G = TOMIEnumDescriptor('G')
    Gs = TOMIEnumDescriptor('G#')
    A = TOMIEnumDescriptor('A')
    As = TOMIEnumDescriptor('A#')
    B = TOMIEnumDescriptor('B')
    Unknown = TOMIEnumDescriptor('Unknown')

    def set_value(self, key: str, *args):
        alias_map = {'C': 'C', 'C#': 'C#', 'Cb': 'B', 'Db': 'C#', 'D': 'D', 'D#': 'D#', 'Eb': 'D#', 'E': 'E',
                     'F': 'F', 'F#': 'F#', 'Fb': 'E', 'Gb': 'F#', 'G': 'G',
                     'G#': 'G#', 'Ab': 'G#', 'A': 'A', 'A#': 'A#', 'Bb': 'A#', 'B': 'B', 'Unknown': 'Unknown'}
        super(Key, self).set_value(key.lower().capitalize().replace('s', '#'), alias_map)


class KeyMode:
    """Used to represent the 'key signature'"""
    min_to_maj: dict[str, str]  = {'C': 'D#', 'C#': 'E', 'Db': 'E', 'D': 'F', 'D#': 'F#', 'Eb': 'F#', 'E': 'G', 'F': 'G#',
                  'F#': 'A', 'Gb': 'A', 'G': 'A#', 'G#': 'B', 'Ab': 'B', 'A': 'C', 'A#': 'C#', 'Bb': 'C#',
                  'B': 'D'}
    maj_to_min: dict[str, str] = {'D#': 'C', 'E': 'C#', 'F': 'D', 'F#': 'D#', 'G': 'E', 'G#': 'F', 'A': 'F#', 'A#': 'G',
                  'B': 'G#', 'C': 'A', 'C#': 'A#', 'D': 'B', 'Db': 'A#', 'Eb': 'C', 'Gb': 'D#', 'Ab': 'F',
                  'Bb': 'G'}
    BASE_SCALE: tuple[str, ...] = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
    _maj_scale_indexes: tuple[int, ...] = (0, 2, 4, 5, 7, 9, 11)
    _min_scale_indexes: tuple[int, ...] = (0, 2, 3, 5, 7, 8, 10)
    def __init__(self, key: Union[str, Key] = None, mode: Union[str, Mode] = None):
        """
        A simple KeyMode() would return an instance of KeyMode of C Major.
        :param key: Can be a string or an instance of Key. Will cast string to 'Key'.
        :param mode: Can be a string or an instance of Mode. Will cast string to 'Mode'.
        """
        if key is None:
            key = Key.C
        elif isinstance(key, str):
            key = key.lower()
            if key.__contains__('m'):
                key, mode = key[:key.index('m')], key[key.index('m'):]
                mode = Mode(mode)
            key = Key(key)
        if mode is None:
            mode = Mode.Major
        elif isinstance(mode, str):
            mode = Mode(mode)
        self.key, self.mode = key, mode

    def __repr__(self):
        return f"KeyMode: {self.key.value}{self.mode.value}"

    def __str__(self):
        return f"{self.key.value}{self.mode.value}"

    def __sub__(self, other: 'KeyMode'):
        """
        Get the steps needed for current KeyMode to move to the other KeyMode, if the result is positive, it means to
        move up, move down otherwise.
        :param other: the other KeyMode object.
        :return: an integer represents the 'steps'.
        """
        if self.key == Key.Unknown or other.key == Key.Unknown:
            return 0
        c_other = other.to_mode(self.mode) if self.mode != other.mode else other
        up = self.full_scale.index(c_other.key.value)
        down = 12 - up
        return -down if down < up else up

    def __eq__(self, other: 'KeyMode'):
        return self.mode == other.mode and self.key == other.key

    def __hash__(self):
        return hash((self.key, self.mode))

    def to_mode(self, mode: Mode):
        """
        Since each major scale matches the keys of a minor scale (e.g. CMaj and Amin), this method allows you to get a
        KeyMode object that matches the keys in current object's scale but its mode is 'mode'.
        :param mode: a Mode object.
        :return: If current KeyMode is 'C Major' and the 'mode' parameter is 'Mode.Minor', it will return 'A Minor' as a
        KeyMode object
        """
        return self.to_major() if mode == Mode.Major else self.to_minor()

    def transpose(self, steps: int) -> 'KeyMode':
        k = self.full_scale[self.full_scale.index(self.key.value) + steps % len(self.full_scale)]
        return KeyMode(k, self.mode)

    def is_major(self) -> bool:
        return self.mode == Mode.Major

    def is_minor(self) -> bool:
        return self.mode == Mode.Minor

    def to_major(self):
        key = self.key if self.mode == Mode.Major or self.key == Key.Unknown else Key(self.min_to_maj[self.key.value])
        return KeyMode(key, Mode.Major)

    def to_minor(self):
        key = self.key if self.mode == Mode.Minor or self.key == Key.Unknown else Key(self.maj_to_min[self.key.value])
        return KeyMode(key, Mode.Minor)

    @property
    def scale(self) -> list[str]:
        key = Key.C if self.key.value == 'Unknown' else self.key
        base_scale = KeyMode.BASE_SCALE
        root_i = base_scale.index(key.value)
        if self.mode.value == "Major":
            scale = [base_scale[(root_i + i) % 12] for i in self._maj_scale_indexes]
        else:
            scale = [base_scale[(root_i + i) % 12] for i in self._min_scale_indexes]
        return scale

    def get_scale_note_indexes(self) -> list[int]:
        """
        Get the indexes of notes in current KeyMode scale in base scale (CMajor).
        For example, if the KeyMode is AMajor, you will get (9, 11, 0, 2, 4, 5, 7).
        :return: list of indexes
        """
        key = Key.C if self.key.value == 'Unknown' else self.key
        base_scale = KeyMode.BASE_SCALE
        root_i = base_scale.index(key.value)
        if self.mode.value == "Major":
            scale_indexes = [(root_i + i) % 12 for i in self._maj_scale_indexes]
        else:
            scale_indexes = [(root_i + i) % 12 for i in self._min_scale_indexes]
        return scale_indexes

    @property
    def full_scale(self):
        key = Key.C if self.key.value == 'Unknown' else self.key
        base_scale = KeyMode.BASE_SCALE
        root_i = base_scale.index(key.value)
        return base_scale[root_i:] + base_scale[:root_i]

if __name__ == '__main__':
    km = KeyMode('C', Mode.Major)
    print(km.transpose(-1))