get_all_tables = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"

create_genres = '''
CREATE TABLE IF NOT EXISTS TOMI_Genres
(
    id INTEGER PRIMARY KEY,
    genre TEXT UNIQUE,
    audio_use_count INTEGER DEFAULT 0,
    midi_use_count INTEGER DEFAULT 0
); 
'''

drop_genres = "DROP TABLE IF EXISTS TOMI_Genres"

insert_genres = 'INSERT OR IGNORE INTO TOMI_Genres (genre) VALUES (?);'

create_midi_grooves = '''
CREATE TABLE IF NOT EXISTS TOMI_MIDI_Grooves
(
    id INTEGER PRIMARY KEY,
    midi_type TEXT NOT NULL ,
    groove_arr TEXT NOT NULL,
    speed CHAR(10) NOT NULL,
    time_numerator INTEGER NOT NULL,
    time_denominator INTEGER NOT NULL,
    bar INTEGER NOT NULL,
    note_group_count INTEGER NOT NULL,
    use_count INTEGER DEFAULT 0,
    groove_md5 INTEGER NOT NULL,
    UNIQUE(midi_type, time_numerator, time_denominator, groove_md5)
);

CREATE INDEX IF NOT EXISTS "grooves-midi_type-groove_md5"
    ON TOMI_MIDI_Grooves (midi_type, groove_md5);

CREATE INDEX IF NOT EXISTS "grooves-midi_type-speed_bar-note_group_count"
    ON TOMI_MIDI_Grooves (midi_type, speed, bar, note_group_count);

CREATE INDEX IF NOT EXISTS "grooves-time_numerator_time_denominator"
    ON TOMI_MIDI_Grooves (time_numerator, time_denominator);

CREATE INDEX IF NOT EXISTS "grooves-use_count"
    ON TOMI_MIDI_Grooves (use_count);
'''

drop_midi_groove = """
DROP TABLE IF EXISTS TOMI_MIDI_Grooves;
DROP INDEX IF EXISTS "grooves-midi_type-groove_md5";
DROP INDEX IF EXISTS "grooves-midi_type-speed_bar-note_group_count";
DROP INDEX IF EXISTS "grooves-time_numerator_time_denominator";
DROP INDEX IF EXISTS "grooves-use_count";
"""

insert_midi_groove = '''
INSERT OR IGNORE INTO TOMI_MIDI_Grooves 
    (midi_type, groove_arr, speed, time_numerator, time_denominator, bar, note_group_count, groove_md5) 
VALUES 
    (?, ?, ?, ?, ?, ?, ?, ?);
'''

create_midi = '''
CREATE TABLE IF NOT EXISTS TOMI_MIDI
(
    id INTEGER PRIMARY KEY,
    audio_file_id INTEGER,
    from_midi_id INTEGER REFERENCES TOMI_MIDI(id) ON DELETE CASCADE,
    file_name TEXT NOT NULL,
    midi_type TEXT NOT NULL,
    note_list TEXT NOT NULL,
    bar INTEGER,
    time_numerator INTEGER,
    time_denominator INTEGER,
    maj_key CHAR(2),
    min_key CHAR(2),
    groove_id INTEGER NULL REFERENCES TOMI_MIDI_Grooves(id) ON DELETE SET NULL,
    maj_root_prog TEXT,
    min_root_prog TEXT,
    progression_names TEXT,
    progression_count INTEGER,
    midi_zm REAL,
    midi_zd REAL,
    midi_zs REAL,
    note_list_md5 INTEGER,
    maj_root_prog_md5 INTEGER,
    UNIQUE(audio_file_id, midi_type, note_list_md5),
    UNIQUE(from_midi_id, file_name, midi_type, note_list_md5)
);

CREATE INDEX IF NOT EXISTS "midi-file_name-midi_type-note_list_md5"
    ON TOMI_MIDI (file_name, midi_type, note_list_md5);

CREATE INDEX IF NOT EXISTS "midi-from_midi_id-file_name-midi_type-note_list_md5"
    ON TOMI_MIDI (from_midi_id, file_name, midi_type, note_list_md5);

CREATE INDEX IF NOT EXISTS "midi-audio_file_id"
    ON TOMI_MIDI (audio_file_id);
    
CREATE INDEX IF NOT EXISTS "midi-note_list_md5"
    ON TOMI_MIDI (note_list_md5);

CREATE INDEX IF NOT EXISTS "midi-midi_type"
    ON TOMI_MIDI (midi_type);
    
CREATE INDEX IF NOT EXISTS "midi-bar"
    ON TOMI_MIDI (bar);

CREATE INDEX IF NOT EXISTS "midi-time_numerator-time_denominator"
    ON TOMI_MIDI (time_numerator, time_denominator);

CREATE INDEX IF NOT EXISTS "midi-maj_root_prog_md5"
    ON TOMI_MIDI (maj_root_prog_md5);
    
CREATE TRIGGER IF NOT EXISTS increment_groove_use_count
    AFTER INSERT ON TOMI_MIDI
    FOR EACH ROW
    WHEN NEW.groove_id IS NOT NULL
BEGIN
    UPDATE TOMI_MIDI_Grooves
    SET use_count = use_count + 1
    WHERE id = NEW.groove_id;
END;

CREATE TRIGGER IF NOT EXISTS decrement_groove_use_count
    AFTER DELETE ON TOMI_MIDI
    FOR EACH ROW
    WHEN OLD.groove_id IS NOT NULL
BEGIN
    UPDATE TOMI_MIDI_Grooves
    SET use_count = use_count - 1
    WHERE id = OLD.groove_id;

    DELETE FROM TOMI_MIDI_Grooves
    WHERE id = OLD.groove_id AND use_count <= 0;
END;
'''

drop_midi = """
DROP TABLE IF EXISTS TOMI_MIDI;
DROP INDEX IF EXISTS "midi-file_name-midi_type-note_list_md5";
DROP INDEX IF EXISTS "midi-from_midi_id-file_name-midi_type-note_list_md5";
DROP INDEX IF EXISTS "midi-audio_file_id";
DROP INDEX IF EXISTS "midi-note_list_md5";
DROP INDEX IF EXISTS "midi-midi_type";
DROP INDEX IF EXISTS "midi-bar";
DROP INDEX IF EXISTS "midi-time_numerator-time_denominator";
DROP INDEX IF EXISTS "midi-maj_root_prog_md5";
DROP TRIGGER IF EXISTS increment_groove_use_count;
DROP TRIGGER IF EXISTS decrement_groove_use_count;
"""

insert_midi = '''
INSERT OR IGNORE INTO TOMI_MIDI
    (audio_file_id, from_midi_id, file_name, midi_type, note_list, bar, time_numerator, time_denominator, maj_key, min_key,
    groove_id, maj_root_prog, min_root_prog, progression_names, progression_count, midi_zm, midi_zd, midi_zs, note_list_md5, 
     maj_root_prog_md5)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?); 
'''

create_midi_file_genres = '''
CREATE TABLE IF NOT EXISTS TOMI_MIDI_File_Genres
(
    id INTEGER PRIMARY KEY,
    file_id INTEGER REFERENCES TOMI_MIDI(id) ON DELETE CASCADE,
    genre_id INTEGER REFERENCES TOMI_Genres(id) ON DELETE CASCADE,
    UNIQUE(file_id, genre_id)
);

CREATE TRIGGER IF NOT EXISTS increment_midi_genres_use_count
    AFTER INSERT ON TOMI_MIDI_File_Genres
    FOR EACH ROW
BEGIN
    UPDATE TOMI_Genres
    SET midi_use_count = midi_use_count + 1
    WHERE id = NEW.genre_id;
END;

CREATE TRIGGER IF NOT EXISTS decrement_midi_genres_use_count
    AFTER DELETE ON TOMI_MIDI_File_Genres
    FOR EACH ROW
    WHEN OLD.genre_id IS NOT NULL
BEGIN
    UPDATE TOMI_Genres
    SET midi_use_count = midi_use_count - 1
    WHERE id = OLD.genre_id;

    DELETE FROM TOMI_Genres
    WHERE id = OLD.genre_id AND midi_use_count <= 0 AND audio_use_count <= 0;
END;
'''

drop_midi_file_genres = """
DROP TABLE IF EXISTS TOMI_MIDI_File_Genres;
DROP TRIGGER IF EXISTS increment_midi_genres_use_count;
DROP TRIGGER IF EXISTS decrement_midi_genres_use_count;
"""

insert_midi_file_genres = 'INSERT OR IGNORE INTO TOMI_MIDI_File_Genres (file_id, genre_id) VALUES (?, ?);'

delete_all_midis_loaded_from_midi_files = 'DELETE FROM TOMI_MIDI WHERE audio_file_id IS NULL;'

delete_all_midis_loaded_from_audio_files = 'DELETE FROM TOMI_MIDI WHERE audio_file_id IS NOT NULL;'

drop_all_tables = (
    drop_genres, drop_midi_groove, drop_midi, drop_midi_file_genres
)

create_all_tables = (
    create_genres, create_midi_grooves, create_midi, create_midi_file_genres
)