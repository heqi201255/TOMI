from . import MIDIProcessor, PianoRoll
from tomi import InvalidMIDIFileError, compute_md5, printer, Groove, RollBlock, MIDIType, KeyMode
from .db_queries import *
import os
import sqlite3
import numpy as np


def is_proper_midi(midi_processor: MIDIProcessor) -> tuple[bool, str]:
    if midi_processor.midi_type != midi_processor.piano_roll.midi_type:
        return False, "processor MIDI type != piano roll MIDI type."
    if midi_processor.quantized_midi.end_time == 0:
        return False, "MIDI end time is 0."
    total_notes = np.count_nonzero(midi_processor.piano_roll.roll == RollBlock.Start)
    if total_notes < 4:
        return False, "total notes < 4."
    if not midi_processor.is_type((MIDIType.Chord, MIDIType.Composite)):
        return True, ""
    em = np.zeros(midi_processor.piano_roll.shape[0], dtype=bool)
    em[midi_processor.piano_roll.get_scale_row_ids()] = True
    out_of_key_notes = np.count_nonzero(midi_processor.piano_roll.roll[~em] == RollBlock.Start)
    if out_of_key_notes >= total_notes * 0.2:
        return False, "too much notes out of key."
    if len(midi_processor.note_groups) <= 3:
        return False, f"{midi_processor.midi_type} contains less than 3 chords."
    if len(midi_processor.progression_nums["major"]) == 0:
        return False, f"{midi_processor.midi_type} contains no root progression."
    if midi_processor.midi_type.name == "Bass":
        if any(np.all(r != RollBlock.Empty) for r in midi_processor.piano_roll.roll):
            return False, f"{midi_processor.midi_type} one or more rows full of notes."
    return True, ""


class MIDISampleDB:
    db_name_map = {
        'genres': 'TOMI_Genres',
        'grooves': 'TOMI_MIDI_Grooves',
        'midi': 'TOMI_MIDI',
        'midi_genres': 'TOMI_MIDI_File_Genres'
    }
    def __init__(self, database_path: str):
        self.database_path = database_path
        self.bad_midi_count = self.success_midi_count = 0
        self.conn = self.cursor = None
        self._insert_midi_genre = self.__insert_and_relate_tag_with_file(insert_genres, insert_midi_file_genres, 'genres', 'genre')
        self._init_connection()

    def _init_connection(self):
        if self.cursor is None:
            self.conn = sqlite3.connect(self.database_path)
            self.conn.execute("PRAGMA foreign_keys = ON;")
            self.cursor = self.conn.cursor()

    def begin_transaction(self):
        self.execute("BEGIN TRANSACTION")

    def commit(self):
        self.conn.commit()

    def process_midi_folder(self, folder_path: str, overwrite: bool = False):
        def _walk_dir(root_path: str):
            for file in os.listdir(root_path):
                file_path = os.path.join(root_path, file)
                if os.path.isdir(file_path):
                    _walk_dir(file_path)
                elif file.endswith('.mid'):
                    to_be_processed.append(file_path)
        assert os.path.exists(folder_path) and os.path.isdir(folder_path), f"Path '{folder_path}' is invalid!"
        if not self.check_all_tables_exist():
            self._create_all_tables()
        if overwrite:
            self.delete_all_midis_loaded_from_midi_files()
        to_be_processed = []
        _walk_dir(folder_path)
        self.begin_transaction()
        loop = printer.progress(to_be_processed, desc="Processing MIDI files", unit="files")
        for midi_file in loop:
            loop.set_postfix({'path': midi_file})
            self._process_midi_file(midi_file)
        self.commit()
        loop.close()
        printer.print(f"Successfully loaded and processed {self.success_midi_count} midis, failed to load {self.bad_midi_count} midis.")
        self.bad_midi_count = self.success_midi_count = 0

    def _process_midi_file(self, file_path: str):
        try:
            mp = MIDIProcessor(file_path, fit_beat=False)
        except InvalidMIDIFileError:
            printer.print(f"Bad midi '{file_path}' File corrupt!")
            self.bad_midi_count += 1
            return
        is_proper, msg = is_proper_midi(mp)
        if not is_proper:
            printer.print(f"Bad midi '{file_path}': {msg}")
            self.bad_midi_count += 1
            return
        file_name = os.path.basename(file_path)
        genres = []
        try:
            s = file_path.index("[")
            e = file_path.index("]")
            assert e > s
            g_str = file_path[s+1:e].lower()
            genres = g_str.split(", ")
        except:
            pass
        genres = ["".join(g.split('-')).lower() for g in genres]
        self._insert_midi(mp, midi_file_name=file_name, print_roll=True, midi_genres=genres)
        self.success_midi_count += 1

    def __insert_and_relate_tag_with_file(self, insert_tag_query: str, insert_file_tag_query: str, tag_db: str, key_name: str):
        def func(file_id: int, tag: str):
            item_id = self.retrieve(self.db_name_map[tag_db],'id', {key_name: tag})
            if item_id:
                item_id = item_id[0]
            else:
                self.execute(insert_tag_query, (tag,))
                item_id = self.retrieve(self.db_name_map[tag_db],'id', {key_name: tag})[0]
            self.execute(insert_file_tag_query, (file_id, item_id))
        return func

    def retrieve(self, table_name: str, keys: str | tuple[str, ...] | list[str] = None, conditions: dict = None, fetchone: bool = True):
        keys = "*" if keys is None else keys if isinstance(keys, str) else ", ".join(keys)
        conditions_q, parameters = self._make_condition_clause(conditions)
        self.execute(f"SELECT {keys} FROM {table_name} {conditions_q};", parameters)
        return self.fetchone() if fetchone else self.fetchall()

    @staticmethod
    def _make_condition_clause(conditions: dict = None):
        if conditions is None:
            return "", None
        cond = []
        parameters = []
        for k, v in conditions.items():
            if v is None:
                cond.append(f"{k} IS NULL")
            else:
                cond.append(f"{k}=?")
                parameters.append(v)
        return f"WHERE {" AND ".join(cond)}", parameters

    def delete(self, table_name: str, conditions: dict = None, commit: bool = False):
        conditions_q, parameters = self._make_condition_clause(conditions)
        self.execute(f"DELETE FROM {table_name} {conditions_q};", parameters)
        if commit: self.conn.commit()

    def _insert_midi(self, midi: MIDIProcessor, audio_file_id: int = None, audio_file_name: str = None, midi_file_name: str = None,
                     print_roll: bool = False, audio_key: KeyMode = None, midi_genres: list[str] = None):
        def __insert_groove(groove: Groove) -> int | None:
            if groove is None:
                return None
            gmt = groove.midi_type.name
            g = str(groove.get_groove().tolist())
            g_md5 = compute_md5(g)
            groove_id = self.retrieve(self.db_name_map['grooves'],'id',
                                      {'midi_type': gmt, 'time_numerator': groove.time_signature.numerator,
                                       'time_denominator': groove.time_signature.denominator, 'groove_md5': g_md5})
            if groove_id:
                return groove_id[0]
            else:
                self.execute(insert_midi_groove, (gmt, g, groove.speed.name, groove.time_signature.numerator,
                                                  groove.time_signature.denominator, groove.bar_length, groove.progression_count, g_md5))
                return self.retrieve(self.db_name_map['grooves'],'id',
                                     {'midi_type': gmt, 'time_numerator': groove.time_signature.numerator,
                                      'time_denominator': groove.time_signature.denominator, 'groove_md5': g_md5})[0]

        def __insert(pr: PianoRoll, midi_nl: list = None, from_midi_id: int = None, groove: Groove = None) -> tuple[int] | None:
            midi_z = pr.get_piano_roll_metrics()
            midi_nl = midi_nl if midi_nl is not None else pr.to_notelist().get_note_list()
            if not midi_nl:
                return None
            midi_nl = str(midi_nl)
            midi_md5 = compute_md5(midi_nl)
            if pr.midi_type.name in ('Composite', 'Chord', 'Bass'):
                maj_rp, min_rp = maj_root_progression, min_root_progression
                maj_rp_md5 = compute_md5(maj_root_progression) if maj_root_progression is not None else None
            else:
                maj_rp, min_rp, maj_rp_md5 = None, None, None
            if pr.midi_type.name in ('Composite', 'Chord'):
                pn, pc = progression_names, progression_count
            else:
                pn, pc = None, None
            self.execute(insert_midi, (
                    audio_file_id,
                    from_midi_id,
                    file_name,
                    pr.midi_type.name,
                    midi_nl,
                    pr.parent.ceil_bst.to_bars(),
                    pr.time_signature.numerator,
                    pr.time_signature.denominator,
                    key.to_major().key.value if key is not None else None,
                    key.to_minor().key.value if key is not None else None,
                    __insert_groove(groove),
                    maj_rp, min_rp, pn, pc,
                    midi_z['zm'], midi_z['zd'], midi_z['zs'],
                    midi_md5, maj_rp_md5
                )
            )
            return self.retrieve(self.db_name_map['midi'], 'id', {'from_midi_id': from_midi_id, 'file_name': file_name, 'midi_type': pr.midi_type.name, 'note_list_md5': midi_md5})
        if print_roll:
            midi.piano_roll.print_roll()
        file_name = midi_file_name if midi_file_name is not None else audio_file_name
        key = audio_key if audio_key is not None else midi.key
        maj_root_progression = str(midi.progression_nums['major']) if midi.progression_nums['major'] else None
        min_root_progression = str(midi.progression_nums['minor']) if midi.progression_nums['minor'] else None
        progression_names = str(midi.progression_names) if midi.progression_names else None
        progression_count = midi.progression_count
        inserted_ids = []
        mid = __insert(midi.piano_roll, midi_nl=midi.get_notelist().get_note_list(), groove=midi.groove)
        if mid is not None:
            mid = mid[0]
            inserted_ids.append(mid)
        if midi.midi_type in (MIDIType.Composite, MIDIType.Chord):
            if midi.midi_type == MIDIType.Composite and midi.has_melody():
                stem_mid = __insert(midi.melody_roll, from_midi_id=mid, groove=midi.melody_groove)
                if stem_mid is not None:
                    inserted_ids.append(stem_mid[0])
            if midi.chord_roll is not None:
                stem_mid = __insert(midi.chord_roll, from_midi_id=mid, groove=midi.chord_groove)
                if stem_mid is not None:
                    inserted_ids.append(stem_mid[0])
            if midi.bass_roll is not None:
                stem_mid = __insert(midi.bass_roll, from_midi_id=mid, groove=midi.bass_groove)
                if stem_mid is not None:
                    inserted_ids.append(stem_mid[0])
        if midi_genres is not None:
            for iid in inserted_ids:
                for genre in midi_genres:
                    self._insert_midi_genre(iid, genre)

    def execute(self, query: str, parameters: list | tuple = None):
        if parameters is not None:
            return self.cursor.execute(query, parameters)
        else:
            return self.cursor.execute(query)

    def execute_multi(self, query: str):
        self.cursor.executescript(query)

    def fetchone(self):
        return self.cursor.fetchone()

    def fetchall(self):
        return self.cursor.fetchall()

    def delete_all_midis_loaded_from_midi_files(self):
        self.execute(delete_all_midis_loaded_from_midi_files)
        self.commit()

    def check_all_tables_exist(self):
        self.execute(get_all_tables)
        tables = self.fetchall()
        tables = {row[0] for row in tables}
        return set(self.db_name_map.values()).issubset(tables)

    def _drop_all_tables(self):
        self.begin_transaction()
        for query in drop_all_tables:
            self.execute_multi(query)
        self.commit()

    def _create_all_tables(self):
        self.begin_transaction()
        for query in create_all_tables:
            self.execute_multi(query)
        self.commit()

    def reset_database(self):
        i = input("[WARNING] Resetting database. Are you sure? (y/n)")
        if i.lower() == 'y':
            self._drop_all_tables()
            self._create_all_tables()
