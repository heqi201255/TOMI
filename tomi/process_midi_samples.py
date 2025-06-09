from tomi import MIDISampleDB, PROGRAM_DIR
import os

# Enter the directory path of your MIDI packs.
midi_packs_dir = ""

db_dir = os.path.join(PROGRAM_DIR, "data/databases")
db_name = "tomi_db.db3" # Do not change


if __name__ == '__main__':
    if midi_packs_dir == "":
        raise ValueError("Please specify midi_packs_dir first.")
    elif not os.path.isdir(midi_packs_dir):
        raise ValueError("The midi_packs_dir must be a valid directory.")
    os.makedirs(db_dir, exist_ok=True)
    db = MIDISampleDB(os.path.join(db_dir, db_name))
    db.process_midi_folder(midi_packs_dir)
