class InvalidMIDIFileError(Exception):
    pass


class InvalidNumeratorError(Exception):
    pass


class InvalidDenominatorError(Exception):
    pass


class UnsupportedNodeTypeError(Exception):
    pass


class MidiNotFoundError(Exception):
    pass


class UnsupportedMIDITypeError(Exception):
    pass


class NoAvailableSamplesError(Exception):
    pass


class InvalidParentNodeError(Exception):
    pass


class InvalidChildNodeError(Exception):
    pass


class ExcessiveParentError(Exception):
    pass


class UnableToLoadPluginError(Exception):
    pass


class AddInstrumentOnAudioTrackError(Exception):
    pass


class InvalidPresetError(Exception):
    pass
