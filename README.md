# TOMI

[Demo](https://tomi-2025.github.io/) | [Paper]()

This is the code repository of the paper:

> Qi He, Ziyu Wang, and Gus Xia. TOMI: Transforming and Organizing Music Ideas for Multi-Track Compositions with Full-Song Structure. ISMIR 2025.
---
# Quick Start
1. Install Python 3.12 and Git, then run the following commands:
    ```
    ## Clone the repo
    git clone https://github.com/heqi201255/TOMI_v1.git
    
    ## Install dependencies
    cd TOMI_v1
    pip install -r requirements.txt
    ```
2. Download REAPER from the [official website](https://www.reaper.fm/) and install REAPER.
3. One great thing about REAPER is that you can automate and control it with code via [ReaScript API](https://www.reaper.fm/sdk/reascript/reascript.php). 
We use [reapy](https://github.com/RomeoDespres/reapy) (a pythonic wrapper of ReaScript) in Python. To set up for the first time, open REAPER, then open a 
terminal and run `python -c "import reapy; reapy.configure_reaper()"` to let REAPER know reapy is available, you only need to do this once. Then, 
restart REAPER. For more details, please refer to [reapy's documentation](https://python-reapy.readthedocs.io/en/latest/install_guide.html).
4. Due to copyright restrictions, we are unable to share the sample datasets used in our experiments. Please prepare your own MIDI dataset and audio sample dataset, each MIDI/audio clip should be a 
short segment commonly used in music production (suitable samples can be found on platforms like [Splice](https://splice.com/) and [loopmasters](https://www.loopmasters.com/)). 
The quality of the generated song depends on the quantity, diversity, and quality of your sample datasets.
5. To extract features from audios, download and Install ADSR Sample Manager from the [official website](https://www.adsrsounds.com/product/software/adsr-sample-manager/).
6. Open ADSR Sample Manager, add the path of your **audio sample dataset** in the application, and wait for it to parse the meta-information for all audio samples.
7. After finishing processing the audio samples, the default database containing the meta data is located in:
   >   MacOS: `'~/Library/Application Support/ADSR/adsr_1_8.db3'`;
   > 
   >   Windows: `'C:\Users\{USER_NAME}\AppData\Roaming\ADSR\adsr_1_8.db3'`.

   If you cannot find the database, you can open the setting window in ADSR Sample Manager and click "EXPORT DATABASE" to export the same database to a custom location.
   In the directory of this repo, open `config.py` and replace the value of `SAMPLE_DB_ADDRESS` to the path of your audio sample database.
8. Open `process_midi_samples.py`, enter the local path of your MIDI dataset to `midi_packs_dir`, and run the file. Wait until the processing finished, then a MIDI database file will be created 
in the path `tomi/data/databases/tomi_db.db3` in the code directory.

   >**Note**: currently we have not implement the mechanism for MIDI genre detection, you can add one or more genre tags to a MIDI pack folder by manually adding "[tag_name, ...]" in front of the folder name before processing it. For example, 
rename the folder `RnB progression MIDIs` to `[RnB, Pop] RnB progression MIDIs`, this will allow the script to recognize the suitable genres for the MIDI files inside the folder and can help for better 
retrieval results. We release our code for analyzing and extracting musical features from MIDI files in `tomi/data_processors/midi_processor.py`, see the "MIDIProcessor" section below for further details.
9. Open `config.py`, set `SAMPLE_DB_ADDRESS`, `REAPER_PATH`, default instrument and LLM API settings.
10. To generate a song in REAPER, open REAPER first, then open `generate.py`, you will see:
   ```
   tomi_song = TOMISong(
       song_name="tomi_song",
       song_genre=SongGenre.Pop,
       song_blocks=None,
       key=KeyMode('C', 'major')
   )
   
   tomi_song.gen(stream_output=False, open_editor=False)
   ```
   For the `song_blocks` parameter in `TOMISong`, you can use the path of a previously generated JSON file; if None, it will generate a new song arrangement in TOMI data structure and save it as a JSON file in `tomi/model_outputs/TOMILLMRequest/`.
   In `TOMISONG.gen()` method, `stream_output` is used to just mimic the stream output style of text LLMs in REAPER, setting it to False can load all song data much faster; we also provide a basic user interface for visualizing the current TOMI data structure, if `open_editor` is set to True, the interface window will be open once the data is loaded into REAPER, see the "TOMIEditor" section below for further details.
---
# MIDIProcessor
We implement the `MIDIProcessor` class to extract features from a MIDI file, including the Bar-Step-Length duration (the duration metric often used in DAW), stems (chord, bass, and melody) and groove. It loads the file using the `prettymidi` package, then converts the 