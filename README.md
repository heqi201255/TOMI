# TOMI

[Demo](https://tomi-2025.github.io/) | [Paper](https://arxiv.org/abs/2506.23094)

This is the code repository of the paper:

> Qi He, Gus Xia, and Ziyu Wang. TOMI: Transforming and Organizing Music Ideas for Multi-Track Compositions with Full-Song Structure. ISMIR 2025.
---
## Introduction
We propose the **TOMI** (**T**ransforming and **O**rganizing **M**usic **I**deas) paradigm for high-level music data representation.
TOMI models a musical piece as a sparse, four-dimensional space defined by **clips** (short audio or MIDI segments),
**sections** (temporal positions), **tracks** (instrument layers), and **transformations** (elaboration methods).
We represent these concepts as nodes in our data structure and define **composition links**, each a quadruple of nodes, to specify a music clip (what) to be placed in a particular section (when) and on a specific track (where), undergoing certain transformations (how). Nodes are reusable across links, forming a structured representation of the complete composition.
Based on this, we achieve the first electronic music generation system
to produce long-term, multi-track compositions containing both MIDI and audio clips, while achieving **robust structural consistency** and **high music quality**. We use
a foundation text-based large language model (LLM) with TOMI data structure and achieve multi-track electronic music generation with full-song structure through **in-context-learning** (ICL) and sample retrieval.
Moreover, we integrate TOMI with the REAPER digital audio workstation (DAW) to allow for human-AI co-creation and high-resolution audio rendering.


## Installation
Install Python 3.12 and Git, then run the following commands:
```shell
## Clone the repo
git clone https://github.com/heqi201255/TOMI.git

## Install dependencies
cd TOMI
pip install -r requirements.txt
```

### Preparation
1. Download REAPER from the [official website](https://www.reaper.fm/) and install REAPER.
2. One great feature about REAPER is that you can automate and control it with code via [ReaScript API](https://www.reaper.fm/sdk/reascript/reascript.php).
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

## Generate a Composition
1. Open `config.py`, set `SAMPLE_DB_ADDRESS`, `REAPER_PATH`, default instrument and LLM API settings.
2. To generate a song in REAPER, open REAPER first, then open `tomi/demos/generate.py`, you will see:
   ```python
   tomi_song = TOMISong(
       song_name="tomi_song",
       song_genre=SongGenre.Pop,
       song_blocks=None,
       key=KeyMode('C', 'major')
   )
   
   tomi_song.gen(stream_output=False, open_editor=False)
   ```
For the `song_blocks` parameter in `TOMISong`, you can use the path of a previously generated JSON file; if None, it will generate a new song arrangement in TOMI data structure and save it as a JSON file in `tomi/model_outputs/TOMILLMRequest/`.
In `TOMISONG.gen()` method, `stream_output` is used to just mimic the stream output style of text LLMs in REAPER, setting it to False can load all song data much faster; we also provide a basic user interface for visualizing the current TOMI data structure, if `open_editor` is set to True, the interface window will be open once the data is loaded into REAPER. For more details, please refer to the [TOMI Editor](tomi/editor/README.md) page.

## MIDIProcessor
The `MIDIProcessor` class, implemented in `tomi/data_processors/midi_processor.py`, is designed to extract musical features from a given MIDI file. For more details, please refer to the [MIDIProcessor demo](tomi/demos/midi_stem_extraction_demo/midi_stem_extraction_demo.ipynb) notebook.

## Limitations
TOMI is an evolving research framework. The current version represents our first system capable of generating full compositions using a text-based LLM and local samples.
TOMI is still under active development and will continue to be extended and improved in future work. At present, the system has the following limitations:
1. Sample Library Dependency:
The quality of generated music strongly depends on the size and quality of your local sample library. Smaller libraries may result in suboptimal outputs.

2. Sample Retrieval Robustness:
The sample retrieval process is not fully robust. Harmonic coherence may be disrupted by randomness or limited feature representation. You may need to generate multiple times to achieve a satisfactory result. In small libraries, some clips may fail to find suitable matches.

3. Manual Instrument Assignment:
MIDI instrument assignment is not yet automated. All MIDI tracks use the `DEFAULT_INSTRUMENT_PLUGIN` and `DEFAULT_INSTRUMENT_PRESET` specified in `tomi/config.py`. You can manually adjust instruments in REAPER.

4. Arrangement-Focused Data Structure:
The system currently focuses on compositional arrangement. Sound design, mixing, and parameter automation are not supported.

5. No Cross-Section Composition Links:
Composition links are currently restricted to within-section arrangements.

6. Temporal Precision Limitations:
Transformations and MIDI notes are quantized to 16th-note resolution.

Note: The codebase has not been fully tested yet. If you encounter issues/bugs or have questions, feel free to create an issue on our repository for assistance.

# Acknowledgement
Our digital audio workstation integration is realized through [REAPER](https://www.reaper.fm/) with their powerful [ReaScript API](https://www.reaper.fm/sdk/reascript/reascript.php).

We build and manage the audio sample database using the versatile [ADSR Sample Manager](https://www.adsrsounds.com/product/software/adsr-sample-manager/).

# License
This project is licensed under the GNU GPLv3 License - see the [LICENSE](LICENSE) file for details.

[//]: # (# Contributing)

# Citation

If you find TOMI useful for your research, please consider citing:


```bibtex
@misc{he2025tomitransformingorganizingmusic,
      title={TOMI: Transforming and Organizing Music Ideas for Multi-Track Compositions with Full-Song Structure}, 
      author={Qi He and Gus Xia and Ziyu Wang},
      year={2025},
      eprint={2506.23094},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2506.23094}, 
}
```