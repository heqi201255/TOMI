import random
from tomi import (Project, TOMIEngine, SongGenre, SectionNode, TrackType, MIDIType,
                   AudioType, BarStepTick, SectionType, NodeFactory)
from .tomi_llm_request import TOMILLMRequest
import numpy as np


class SimpleLoFiSong:
    """
    Rule-based generation case.
    """
    def __init__(self, song_name: str = "simple_lofi_song", song_bar_length:int = 92, seed: int = None):
        if seed:
            np.random.seed(seed)
            random.seed(seed)
        self.llm = TOMILLMRequest()
        self.project = Project(song_name, genre=SongGenre.LoFi)
        self.engine = TOMIEngine(self.project)
        self.arrangement = SectionNode.generate_arrangement(self.project, song_bar_length)
        self.project.set_arrangement(self.arrangement)
        self.factory = NodeFactory(self.project)
        self.gui = None

    def gen(self, stream_output: bool = False, open_editor: bool = True):
        graph = self.project.node_graph
        # Setting track nodes
        tr_melody = self.factory.track(TrackType.Audio)
        tr_chord = self.factory.track(TrackType.Midi)
        tr_bass = self.factory.track(TrackType.Midi)
        tr_kick = self.factory.track(TrackType.Audio)
        tr_topdrum = self.factory.track(TrackType.Audio)
        tr_rise = self.factory.track(TrackType.Audio)
        tr_fall = self.factory.track(TrackType.Audio)
        tr_fill = self.factory.track(TrackType.Audio)
        # Setting transformation nodes
        cp = self.factory.general_transform()
        rfx = self.factory.fx_transform()
        ffx = self.factory.fx_transform(is_faller=True)
        fp = self.factory.fill_transform(loop_bars=8)
        # Setting clip nodes
        chord = self.factory.midi(MIDIType.Chord, BarStepTick(4))
        melody = self.factory.audio(AudioType.ElectricGuitar, ['guitar', 'lead'], loop=True, minlen=BarStepTick(4), fit_key=True, fit_tempo=True)
        bass = self.factory.midi(MIDIType.Bass, dependent_node=chord, dependent_type=MIDIType.Bass)
        topdrum = self.factory.audio(AudioType.DrumTop, ['drums', 'hihat'], loop=True, minlen=BarStepTick(4), fit_tempo=True)
        kick = self.factory.audio(AudioType.Kick, ['drums', 'kick'], loop=True, minlen=BarStepTick(4), fit_tempo=True)
        fill = self.factory.audio(AudioType.DrumFill, ['fill', 'drums'], loop=True, minlen=BarStepTick(0, 4), maxlen=BarStepTick(0, 4), fit_tempo=True)
        rise = self.factory.audio(AudioType.Riser, ['sweep', 'house'], loop=False, maxlen=BarStepTick(4, 0), fit_tempo=True)
        fall = self.factory.audio(AudioType.Impact, ['impact', 'crash'], loop=False, maxlen=BarStepTick(2, 0), fit_tempo=False)
        # Linking nodes
        for sec in self.project.section_nodes:
            graph.add(sec >> cp >> topdrum & chord >> tr_topdrum & tr_chord)
            if sec.section_type in (SectionType.Chorus, SectionType.Verse):
                graph.add(sec
                          >>
                          cp | rfx | ffx | fp
                          >>
                          melody & kick & bass | rise | fall | fill
                          >>
                          tr_melody & tr_kick & tr_bass | tr_rise | tr_fall | tr_fill)
        self.project.remove_unused_nodes()
        self.engine.run_reaper(stream_output=stream_output)
        if open_editor:
            from tomi.editor import TOMIEditor
            self.gui = TOMIEditor(self.engine)
            self.gui.run()


if __name__ == '__main__':
    rg = SimpleLoFiSong(seed=20240)
    rg.gen()

