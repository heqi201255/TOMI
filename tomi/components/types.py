from . import TOMIEnum, TOMIEnumDescriptor


__all__ = ["GrooveSpeed", "NodeType", "LinkType", "MIDIType", "SectionType", "SongGenre", "TrackType",
           "AudioType", "InstrumentType", "PluginFormat"]

T = TOMIEnumDescriptor

class GrooveSpeed(TOMIEnum):
    Plain = T('Plain') # Similar to GrooveSpeed.Normal, but is processed to be less rhythmic.
    Normal = T('Normal') # Ordinary rhythmic.
    Fast = T('Fast') # Quite rhythmic.
    Rapid = T('Rapid') # Jumpy.


class NodeType(TOMIEnum):
    Section = T("Section")
    GeneralTransform = T("GeneralTransform")
    DrumTransform = T("DrumTransform")
    FxTransform = T("FxTransform")
    FillTransform = T("FillTransform")
    MidiClip = T("MidiClip")
    AudioClip = T("AudioClip")
    Track = T("Track")


class LinkType(TOMIEnum):
    ArrangementLink = T("ArrangementLink")


class MIDIType(TOMIEnum):
    Composite = T('Composite')
    Chord = T('Chord')
    Bass = T('Bass')
    Melody = T('Melody')
    Arp = T('Arp')
    Kick = T('Kick')
    ClapSnare = T('Snare')
    Hihat = T('Hihat')
    Drummer = T('Drummer')
    def is_drum(self) -> bool:
        return self in [MIDIType.ClapSnare, MIDIType.Kick, MIDIType.Hihat, MIDIType.Drummer]


class SectionType(TOMIEnum):
    Intro = T('Intro')
    Verse = T('Verse')
    PreChorus = T('PreChorus')
    Chorus = T('Chorus')
    Bridge = T('Bridge')
    Outro = T('Outro')


class TrackType(TOMIEnum):
    Midi = T("Midi")
    Audio = T("Audio")


class SongGenre(TOMIEnum):
    Pop = T("Pop")
    EDM = T("EDM")
    FutureBass = T("FutureBass")
    Cyberpunk = T("Cyberpunk")
    House = T("House")
    Emotional = T("Emotional")
    Funk = T("Funk")
    RnB = T("RnB")
    Hiphop = T("Hiphop")
    Trap = T("Trap")
    LoFi = T("LoFi")
    Unspecified = T("Unspecified")


class AudioType(TOMIEnum):
    Keys = T("Keys")
    AcousticGuitar = T("AcousticGuitar")
    ElectricGuitar = T("ElectricGuitar")
    MutedGuitar = T("MutedGuitar")
    BassGuitar = T("BassGuitar")
    String = T("String")
    Horn = T("Horn")
    Kick = T("Kick")
    Snare = T("Snare")
    Clap = T("Clap")
    Snap = T("Snap")
    ClosedHihat = T("ClosedHihat")
    OpenHihat = T("OpenHihat")
    Rides = T("Rides")
    Percussion = T("Percussion")
    Breakbeat = T("Breakbeat")
    Drummer = T("Drummer")
    Foley = T("Foley")
    Cymbal = T("Cymbal")
    DrumFill = T("DrumFill")
    BuildUp = T("BuildUp")
    DrumTop = T("DrumTop")
    DrumFull = T("DrumFull")
    Texture = T("Texture")
    Bass = T("Bass")
    Bass808 = T("Bass808")
    Melody = T("Melody")
    Vocal = T("Vocal")
    Arp = T("Arp")
    Noise = T("Noise")
    SweepUp = T("SweepUp")
    SweepDown = T("SweepDown")
    Riser = T("Riser")
    ReversedSynth = T("ReversedSynth")
    ReversedVocal = T("ReversedVocal")
    ReversedGuitar = T("ReversedGuitar")
    ReverseCymbal = T("ReverseCymbal")
    Stab = T("Stab")
    Impact = T("Impact")
    Ambiance = T("Ambiance")
    SubDrop = T("SubDrop")
    ReverseSynth = T('ReverseSynth')


class InstrumentType(TOMIEnum):
    GrandPiano = T("GrandPiano")
    SoftPiano = T("SoftPiano")
    ElectricPiano = T("ElectricPiano")
    Rhodes = T("Rhodes")
    AcousticGuitar = T("AcousticGuitar")
    ElectricGuitar = T("ElectricGuitar")
    MutedGuitar = T("MutedGuitar")
    BassGuitar = T("BassGuitar")
    Violin = T("Violin")
    Viola = T("Viola")
    Cello = T("Cello")
    Drummer = T("Drummer")
    ChordSynth = T("ChordSynth")
    Bass808 = T("Bass808")
    BassSynth = T("BassSynth")
    PluckSynth = T("PluckSynth")
    BellSynth = T("BellSynth")
    LeadSynth = T("LeadSynth")
    PadSynth = T("PadSynth")
    OtherSynth = T("OtherSynth")
    Erhu = T("Erhu")
    Koto = T("Koto")
    Shamisen = T("Shamisen")
    Brass = T("Brass")
    Horn = T("Horn")
    Sax = T("Sax")
    Flute = T("Flute")
    Unknown = T("Unknown")


class PluginFormat(TOMIEnum):
    AU = T('AU')
    VST = T('VST')
    VST3 = T('VST3')
