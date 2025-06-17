# TOMI

[Demo](https://tomi-2025.github.io/) | [Paper]()

This is the code repository of the paper:

> Qi He, Ziyu Wang, and Gus Xia. TOMI: Transforming and Organizing Music Ideas for Multi-Track Compositions with Full-Song Structure. ISMIR 2025.
---
# Introduction
We propose the TOMI (Transforming and Organizing Music Ideas) paradigm for high-level music data representation.
TOMI models a musical piece as a sparse, four-dimensional space defined by **clips** (short audio or MIDI segments),
**sections** (temporal positions), **tracks** (instrument layers), and **transformations** (elaboration methods).
We represent these concepts as nodes in our data structure and define **composition links**, each a quadruple of nodes, to specify a music clip (what) to be placed in a particular section (when) and on a specific track (where), undergoing certain transformations (how). Nodes are reusable across links, forming a structured representation of the complete composition.
Based on this, we achieve the first electronic music generation system
to produce long-term, multi-track compositions containing both MIDI and audio clips, while achieving **robust structural consistency** and **high music quality**. We use
a foundation text-based large language model (LLM) with TOMI data structure and achieve multi-track electronic music generation with full-song structure through **in-context-learning** (ICL) and sample retrieval.
Moreover, we integrate TOMI with the REAPER digital audio workstation (DAW) to allow for human-AI co-creation and high-resolution audio rendering.


# Installation
Install Python 3.12 and Git, then run the following commands:
```
    ## Clone the repo
    git clone https://github.com/heqi201255/TOMI.git
    
    ## Install dependencies
    cd TOMI
    pip install -r requirements.txt
   ```

# Quick Start
1. Download REAPER from the [official website](https://www.reaper.fm/) and install REAPER.
2. One great thing about REAPER is that you can automate and control it with code via [ReaScript API](https://www.reaper.fm/sdk/reascript/reascript.php). 
We use [reapy](https://github.com/RomeoDespres/reapy) (a pythonic wrapper of ReaScript) in Python. To set up for the first time, open REAPER, then open a 
terminal and run `python -c "import reapy; reapy.configure_reaper()"` to let REAPER know reapy is available, you only need to do this once. Then, 
restart REAPER. For more details, please refer to [reapy's documentation](https://python-reapy.readthedocs.io/en/latest/install_guide.html).
3. Due to copyright restrictions, we are unable to share the sample datasets used in our experiments. Please prepare your own MIDI dataset and audio sample dataset, each MIDI/audio clip should be a 
short segment commonly used in music production (suitable samples can be found on platforms like [Splice](https://splice.com/) and [loopmasters](https://www.loopmasters.com/)). 
The quality of the generated song depends on the quantity, diversity, and quality of your sample datasets.
4. To extract features from audios, download and Install ADSR Sample Manager from the [official website](https://www.adsrsounds.com/product/software/adsr-sample-manager/).
5. Open ADSR Sample Manager, add the path of your **audio sample dataset** in the application, and wait for it to extract the meta-information for all audio samples.
6. After finishing processing the audio samples, the default database containing the meta data is located in:
   >   MacOS: `'~/Library/Application Support/ADSR/adsr_1_8.db3'`;
   > 
   >   Windows: `'C:\Users\{USER_NAME}\AppData\Roaming\ADSR\adsr_1_8.db3'`.

   If you cannot find the database, you can open the setting window in ADSR Sample Manager and click "EXPORT DATABASE" to export the same database to a custom location.
   In the directory of this repo, open `config.py` and replace the value of `SAMPLE_DB_ADDRESS` to the path of your audio sample database.
7. Open `process_midi_samples.py`, enter the local path of your MIDI dataset to `midi_packs_dir`, and run the file. Wait until the processing finished, then a MIDI database file will be created 
in the path `tomi/data/databases/tomi_db.db3` in the code directory.

   >**Note**: currently we have not implement the mechanism for MIDI genre detection, you can add one or more genre tags to a MIDI pack folder by manually adding "[tag_name, ...]" in front of the folder name before processing it. For example, 
rename the folder `RnB progression MIDIs` to `[RnB, Pop] RnB progression MIDIs`, this will allow the script to recognize the suitable genres for the MIDI files inside the folder and can help for better 
retrieval results. We release our code for analyzing and extracting musical features from MIDI files in `tomi/data_processors/midi_processor.py`, see the "MIDIProcessor" section below for further details.
8. Open `config.py`, set `SAMPLE_DB_ADDRESS`, `REAPER_PATH`, default instrument and LLM API settings.
9. To generate a song in REAPER, open REAPER first, then open `generate.py`, you will see:
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

# MIDIProcessor
We implement the `MIDIProcessor` class in `tomi/data_processors/midi_processor.py` to extract features given a MIDI file, including the MIDI type (composite, chord, bass, melody, 
arp, and drums, "composite" type means the MIDI may contain both chord progression and melody), duration, stems (chord, bass, and melody) and 
groove. It loads the file using the [pretty-midi](https://github.com/craffel/pretty-midi) package, then converts the MIDI note sequence 
to a 2D array which is essentially the piano roll. We use the bar-step-tick concept in TOMI to adopt metrical time instead of absolute 
time. In `MIDIProcessor`, currently the start/end position of each MIDI note is quantized to its nearest step rather than ticks to reduce the size 
of the piano roll. 

[//]: # (The algorithms we developed to extract features are all rule-based and are performed on the piano roll rather than the note sequence, )
[//]: # (and we have not)

# TOMI Editor GUI Demo

# Limitations
TOMI is a continuous research project. Currently, our framework is capable of generating long-term full-song composition with a text LLM using local sample materials, but has the following limitations:
1. The quality of the final song heavily relies on the quality and quantity of the user's local sample library. If you have a small sample collection, you will likely to have bad results.
2. The sample retrieval process is not robust enough. The harmonic coherence in our results can occasionally be disrupted by randomness and limited features during sample retrieval. You may need to try a few times to get a song of reasonable quality. If you have a small sample collection, some clips may have no matching samples after retrieval.
3. The instrument assignment for MIDI tracks is currently NOT automated. TOMI will use the `DEFAULT_INSTRUMENT_PLUGIN` with `DEFAULT_INSTRUMENT_PRESET` settings in `tomi/config.py` for all MIDI tracks, and you can manually change the instruments in REAPER.
4. Our data structure only focuses on the arrangement of the composition. Sound design, mixing, and parameter automation are not supported.

The code base has not been fully tested yet. If you find yourself confused or encountering any issues/bugs, feel free to create an issue on our repository for assistance.

# Acknowledgement
Our digital audio workstation integration is realized through [REAPER](https://www.reaper.fm/) with their powerful [ReaScript API](https://www.reaper.fm/sdk/reascript/reascript.php).

We build and manage the audio sample database using the versatile [ADSR Sample Manager](https://www.adsrsounds.com/product/software/adsr-sample-manager/).

# License
This project is licensed under the GNU GPLv3 License - see the [LICENSE](LICENSE) file for details.

[//]: # (# Contributing)

# Citation

If you find TOMI useful for your research, please consider citing:


```


```