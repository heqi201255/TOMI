import random
from tomi import (Project, TOMIEngine, SongGenre, SectionNode, TrackType, MIDIType, DEFAULT_INSTRUMENT_PLUGIN, DEFAULT_INSTRUMENT_PRESET,
                   AudioType, BarStepTick, NodeFactory, GrooveSpeed, Key)
from experiment_section_patterns import form_structure


class RandomSong:
    def __init__(self, song_name: str = "random_song", key: Key = None, song_bar_length:int = 92, seed: int = None, section_pattern: str = None):
        self.random_seed = seed
        self.rand_id = 0
        key = Key.C if key is None else key
        self.project = Project(song_name, genre=SongGenre.Pop, key=key)
        self.engine = TOMIEngine(self.project)
        if section_pattern is not None:
            self.arrangement = form_structure(section_pattern, self.project)
        else:
            self.arrangement = SectionNode.generate_arrangement(self.project, song_bar_length)
        self.project.set_arrangement(self.arrangement)
        self.factory = NodeFactory(self.project)
        self.gui = None

    def randint(self, low, high):
        if self.random_seed is not None and isinstance(self.random_seed, int):
            random.seed(self.random_seed + self.rand_id)
            self.rand_id += 1
        return random.randint(low, high)

    def randbool(self):
        if self.random_seed is not None and isinstance(self.random_seed, int):
            random.seed(self.random_seed + self.rand_id)
            self.rand_id += 1
        return random.choice((True, False))

    def random_choice(self, iterable, weights = None):
        if self.random_seed is not None and isinstance(self.random_seed, int):
            random.seed(self.random_seed + self.rand_id)
            self.rand_id += 1
        if weights is None:
            return random.choice(iterable)
        return random.choices(iterable, weights)

    def gen_midi_tracks(self, midi_track_num: int, graph, transform_node, same_chord_in_section: bool = True):
        section_chord_clips = {} # Section: Chord MIDI
        section_bass_clips = {} # Section: Bass MIDI
        midi_clips = {MIDIType.Chord: [], MIDIType.Bass: [], MIDIType.Melody: []}
        midi_clips_chord = (MIDIType.Chord, MIDIType.Composite)
        midi_clips_melody = (MIDIType.Melody, MIDIType.Arp)
        for midi_track_i in range(midi_track_num):
            if not midi_clips[MIDIType.Chord]:
                m_type = MIDIType.Chord
            else:
                m_type = self.random_choice((MIDIType.Chord, MIDIType.Bass, MIDIType.Melody))
            if m_type == MIDIType.Chord:
                m_type_spec = self.random_choice(midi_clips_chord)
            elif m_type == MIDIType.Melody:
                m_type_spec = self.random_choice(midi_clips_melody)
            else:
                m_type_spec = m_type # Bass
            midi_track = self.factory.track(TrackType.Midi, plugin_name=DEFAULT_INSTRUMENT_PLUGIN, plugin_preset=DEFAULT_INSTRUMENT_PRESET)
            for sec_i, section in enumerate(self.project.section_nodes):
                if m_type == MIDIType.Chord:
                    add_clip = True if sec_i == 0 and midi_track_i == 0 else self.randbool()
                    if add_clip:
                        if not same_chord_in_section or section not in section_chord_clips:
                            add_new_clip = self.randbool()
                            if add_new_clip or not midi_clips[MIDIType.Chord]:
                                m_clip = self.factory.midi(m_type_spec, BarStepTick(self.random_choice((4, 8))),
                                                           groove_speed=self.random_choice((GrooveSpeed.Normal, GrooveSpeed.Fast, GrooveSpeed.Rapid)))
                                midi_clips[MIDIType.Chord].append(m_clip)
                            else:
                                m_clip = self.random_choice(midi_clips[MIDIType.Chord])
                            if same_chord_in_section:
                                section_chord_clips[section] = m_clip
                        else:
                            m_clip = section_chord_clips[section]
                        graph.add(section >> transform_node >> m_clip >> midi_track)
                elif m_type == MIDIType.Bass:
                    add_clip = self.randbool()
                    if add_clip:
                        if not same_chord_in_section or section not in section_bass_clips:
                            dependent_chord = section_chord_clips[section] if same_chord_in_section and section in section_chord_clips else self.random_choice(midi_clips[MIDIType.Chord])
                            m_clip = self.factory.midi(m_type_spec, dependent_node=dependent_chord, dependent_type=MIDIType.Bass)
                            midi_clips[MIDIType.Bass].append(m_clip)
                            if same_chord_in_section:
                                section_bass_clips[section] = m_clip
                        else:
                            m_clip = section_bass_clips[section]
                        graph.add(section >> transform_node >> m_clip >> midi_track)
                else:
                    add_clip = self.randbool()
                    if add_clip:
                        add_new_clip = self.randbool()
                        if add_new_clip or not midi_clips[MIDIType.Melody]:
                            m_clip = self.factory.midi(m_type_spec, BarStepTick(self.random_choice((4, 8))),
                                                       groove_speed=self.random_choice((GrooveSpeed.Normal, GrooveSpeed.Fast, GrooveSpeed.Rapid)))
                            midi_clips[MIDIType.Melody].append(m_clip)
                        else:
                            m_clip = self.random_choice(midi_clips[MIDIType.Melody])
                        graph.add(section >> transform_node >> m_clip >> midi_track)

    def gen_audio_tracks(self, audio_track_num:int, graph, transform_general, transform_riser, transform_faller, transform_fill):
        audio_clips_melodic = (AudioType.Melody, AudioType.Keys, AudioType.AcousticGuitar, AudioType.ElectricGuitar, AudioType.String, AudioType.Horn, AudioType.MutedGuitar)
        audio_clips_drum = (AudioType.Breakbeat, AudioType.Clap, AudioType.ClosedHihat, AudioType.OpenHihat, AudioType.Kick,
                            AudioType.DrumFull, AudioType.DrumTop, AudioType.Drummer, AudioType.Snare, AudioType.Snap, AudioType.Foley, AudioType.Percussion,
                            AudioType.Rides)
        audio_clips_riser = (AudioType.Riser, AudioType.BuildUp, AudioType.ReverseCymbal, AudioType.ReversedGuitar, AudioType.ReversedSynth, AudioType.ReversedVocal, AudioType.SweepUp)
        audio_clips_faller = (AudioType.SweepDown, AudioType.Impact, AudioType.Cymbal, AudioType.Stab, AudioType.SubDrop)
        audio_clips_fill = (AudioType.DrumFill, )
        audio_clips_texture = (AudioType.Texture, AudioType.Noise, AudioType.Ambiance)
        for audio_track_i in range(audio_track_num):
            a_type = self.random_choice(("melodic", "drum", "riser", "faller", "fill", "texture"))
            if a_type == "melodic":
                a_type_spec = self.random_choice(audio_clips_melodic)
                loop = True
                fit_key = True
                fit_tempo = True
                trans = transform_general
            elif a_type == "drum":
                a_type_spec = self.random_choice(audio_clips_drum)
                loop = True
                fit_key = False
                fit_tempo = True
                trans = transform_general
            elif a_type == "riser":
                a_type_spec = self.random_choice(audio_clips_riser)
                loop = False
                fit_key = False
                fit_tempo = False
                trans = transform_riser
            elif a_type == "faller":
                a_type_spec = self.random_choice(audio_clips_faller)
                loop = False
                fit_key = False
                fit_tempo = False
                trans = transform_faller
            elif a_type == "fill":
                a_type_spec = self.random_choice(audio_clips_fill)
                loop = False
                fit_key = False
                fit_tempo = True
                trans = transform_fill
            else:
                a_type_spec = self.random_choice(audio_clips_texture)
                loop = True
                fit_key = True
                fit_tempo = True
                trans = transform_general
            audio_track = self.factory.track(TrackType.Audio)
            a_clip = self.factory.audio(a_type_spec, loop=loop, fit_key=fit_key, fit_tempo=fit_tempo)
            add_to_section = self.random_choice(self.project.section_nodes)
            graph.add(add_to_section >> trans >> a_clip >> audio_track)
            for sec_i, section in enumerate(self.project.section_nodes):
                if section == add_to_section:
                    continue
                add_clip = self.randbool()
                if add_clip:
                    graph.add(section >> trans >> a_clip >> audio_track)

    def gen(self, stream_output: bool = False, open_editor: bool = True, same_chord_in_section: bool = True):
        graph = self.project.node_graph
        transform_general = self.factory.general_transform(loop=True, node_name="transform_general")
        transform_riser = self.factory.fx_transform(node_name="transform_riser")
        transform_faller = self.factory.fx_transform(is_faller=True, node_name="transform_faller")
        transform_fill = self.factory.fill_transform(loop_bars=8, node_name='transform_fill')
        total_track_num = self.randint(15, 25)
        midi_track_num = self.randint(3, 15)
        audio_track_num = total_track_num - midi_track_num
        self.gen_midi_tracks(midi_track_num, graph, transform_general, same_chord_in_section)
        self.gen_audio_tracks(audio_track_num, graph, transform_general, transform_riser, transform_faller, transform_fill)
        self.project.remove_unused_nodes()
        self.engine.run_reaper(stream_output=stream_output)
        if open_editor:
            from tomi.editor import TOMIEditor
            self.gui = TOMIEditor(self.engine)
            self.gui.run()


if __name__ == '__main__':
    rg = RandomSong(seed=20240)
    rg.gen()

