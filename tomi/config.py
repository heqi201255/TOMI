import os

# The project directory of TOMI. Do not change.
PROGRAM_DIR = file_path = os.path.abspath(__file__).rstrip("config.py")

# TOMI V1 retrieve clips from your own local MIDI/Audio sample collections for music generation, you must have a MIDI sample dataset
# (all should be short clips, e.g., chord progressions, melody phrases) and process your MIDI dataset into an SQLite3 database via
# "process_midi_samples.py".
MIDI_DB_ADDRESS = os.path.expanduser(os.path.join(PROGRAM_DIR, "data/databases/tomi_db.db3"))

# The ADSR Sample Manager sqlite3 database path, please install ADSR Sample Manager first, then drag your local audio sample collections
# to the app and wait for it to analyze all the meta information of all samples.
SAMPLE_DB_ADDRESS = '~/Library/Application Support/ADSR/adsr_1_8.db3'
SAMPLE_DB_ADDRESS = os.path.expanduser(SAMPLE_DB_ADDRESS)

# If you set the seed to a value other than None, the selected sample of each clip will always be the same when the database is not changed.
FIX_SEED = None

# The generated song time signature is set to 4/4, other time signatures are not tested yet.
DEFAULT_TIME_SIGNATURE_NUMERATOR = 4
DEFAULT_TIME_SIGNATURE_DENOMINATOR = 4

# If you want to have some empty bars before the generated song, increase this parameter.
ARRANGEMENT_FRONT_PADDING_BARS = 0

# Reaper App Path
REAPER_PATH = "/Applications/REAPER.app"

# LLM Settings. If you want to use OpenAI models, make sure to set the 'MODEL_PROVIDER' to 'OpenAI';
# if you want to use Claude models, make sure to set 'MODEL_PROVIDER' to 'Claude'.
# The 'LLM_MODEL' is the model name of text-based LLM.
MODEL_PROVIDER = 'OpenAI' # or 'Claude'
LLM_MODEL = 'gpt-4o-2024-11-20'
API_BASE = ""
API_KEY = ""

# TOMI currently does not support automatic instrument assignment yet.
# We will use the following provided synth plugin and preset for all MIDI tracks,
# if not provided, there will be no sound for MIDI tracks, and you can change the instrument manually inside the DAW.
# Set the plugin and preset like this:
# DEFAULT_INSTRUMENT_PLUGIN = "Serum"
# DEFAULT_INSTRUMENT_PRESET = "/Library/Audio/Presets/Xfer Records/Serum Presets/Presets/Factory/Pads/PD Jp8k [7S].fxp"
DEFAULT_INSTRUMENT_PLUGIN = None
DEFAULT_INSTRUMENT_PRESET = None