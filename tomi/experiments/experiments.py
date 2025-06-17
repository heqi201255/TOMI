from tomi import SongGenre, config, Key
import json
import os
import random
import datetime
from tomi.experiments.random_song import RandomSong
from tomi.experiments.tomi_song import TOMISong
from tomi.experiments.standalone_llm_song import StandaloneLLMSong, StandaloneLLMRequest
from tomi.experiments.tomi_llm_request import TOMILLMRequest
from tomi.experiments.experiment_section_patterns import *


def generate_tomi_song(section_pattern: str, key: str, demo_id: int):
    song_name = f"tomi_{section_pattern}_{key}_{demo_id}"
    dir_path = f"./demos/TOMI/{section_pattern}/{song_name}/"
    os.makedirs(dir_path, exist_ok=True)
    random.seed(f"{datetime.datetime.now()}")
    if os.path.exists(os.path.join(dir_path, f'{song_name}_config.json')):
        print("Loading existing config")
        gen_config = json.load(open(os.path.join(dir_path, f'{song_name}_config.json'), "r"))
    else:
        user_prompt = "Please make a instrumental high quality Pop song. Feel free to choose any instruments you like on your own. The tempo is about 120. Your generation should be completely provided, and should be close to real world music production, which means about 50+ clips, 20+ tracks, 30+ transformations, and 60+ links."
        user_prompt = f"{form_structure_and_section_prompt(section_pattern)}\n{user_prompt}"
        gen_config = {
            "id": demo_id,
            "genre": "Pop",
            "user_prompt": user_prompt,
            "seed": 2014,
        }
    config.FIX_SEED = gen_config["seed"]
    llm = TOMILLMRequest(section_pattern=section_pattern)
    require_gen_song_blocks = True
    song_blocks = None
    if os.path.exists(os.path.join(dir_path, "song_blocks.json")):
        print("Loading existing song blocks")
        song_blocks = json.load(open(os.path.join(dir_path, "song_blocks.json"), "r"))
        require_gen_song_blocks = False
    while True:
        if require_gen_song_blocks:
            song_blocks = llm.generate_song_blocks(SongGenre.get_object_by_name(gen_config['genre']), user_prompt=gen_config['user_prompt'], output_save_path=dir_path, save_file_name="song_blocks.json")
            require_gen_song_blocks = False
        llm_gen = TOMISong(song_blocks=song_blocks, key=Key(key))
        llm_gen.gen(stream_output=False, open_editor=False)
        confirm = input(f"[{section_pattern}, {key}, {demo_id}] Shuffle clips again? (a(increase seed)/d(decrease seed)/n(next)/r(regenerate))")
        with open(os.path.join(dir_path, f"{song_name}_config.json"), "w") as f:
            json.dump(gen_config, f)
        if confirm == "a":
            gen_config["seed"] += 1
            config.FIX_SEED = gen_config["seed"]
        elif confirm == "d":
            gen_config["seed"] -= 1
            config.FIX_SEED = gen_config["seed"]
        elif confirm == 'r':
            require_gen_song_blocks = True
        else:
            break

def generate_standalone_llm_song(section_pattern: str, key: str, demo_id: int):
    song_name = f"llm_{section_pattern}_{key}_{demo_id}"
    dir_path = f"./demos/StandaloneLLM/{section_pattern}/{song_name}/"
    os.makedirs(dir_path, exist_ok=True)
    random.seed(f"{datetime.datetime.now()}")
    if os.path.exists(os.path.join(dir_path, f'{song_name}_config.json')):
        print("Loading existing config")
        gen_config = json.load(open(os.path.join(dir_path, f'{song_name}_config.json'), "r"))
    else:
        user_prompt = "Please make a {genre} instrumental song. Feel free to choose any instruments you like on your own. The tempo is about 120, mood is happy. Your generation should be completely provided, and should be close to real world music production, which means your result should contain about 20+ tracks, 50+ clips."
        user_prompt = f"{form_section_prompt(section_pattern)} {user_prompt}"
        gen_config = {
            "id": demo_id,
            "genre": "Pop",
            "user_prompt": user_prompt,
            "seed": random.randint(0, 2**32 - 1),
        }
    config.FIX_SEED = gen_config["seed"]
    llm = StandaloneLLMRequest(section_pattern=section_pattern)
    if os.path.exists(os.path.join(dir_path, "song_blocks.json")):
        print("Loading existing song blocks")
        song_blocks = json.load(open(os.path.join(dir_path, "song_blocks.json"), "r"))
    else:
        song_blocks = llm.generate_song_blocks(SongGenre.get_object_by_name(gen_config['genre']), user_prompt=gen_config['user_prompt'], output_save_path=dir_path, save_file_name="song_blocks.json")
    while True:
        llm_gen = StandaloneLLMSong(song_blocks=song_blocks, key=Key(key))
        llm_gen.gen(stream_output=False, open_editor=False)
        confirm = input(f"[{section_pattern}, {key}, {demo_id}] Shuffle clips again? (y/n)")
        with open(os.path.join(dir_path, f'{song_name}_config.json'), "w") as f:
            json.dump(gen_config, f)
        if confirm == "y":
            gen_config["seed"] = random.randint(0, 2**32 - 1)
            config.FIX_SEED = gen_config["seed"]
            continue
        else:
            break

def generate_random_song(section_pattern: str, key: str, demo_id: int):
    song_name = f"random_{section_pattern}_{key}_{demo_id}"
    dir_path = f"./demos/Random/{section_pattern}/{song_name}/"
    os.makedirs(dir_path, exist_ok=True)
    random.seed(f"{datetime.datetime.now()}")
    if os.path.exists(os.path.join(dir_path, f'{song_name}_config.json')):
        print("Loading existing config")
        gen_config = json.load(open(os.path.join(dir_path, f'{song_name}_config.json'), "r"))
    else:
        gen_config = {
            "id": demo_id,
            "seed": random.randint(0, 2**32 - 1),
        }
    print("Using seed:", gen_config['seed'])
    config.FIX_SEED = gen_config["seed"]
    while True:
        rb_gen = RandomSong(seed=gen_config['seed'], key=Key(key), section_pattern=section_pattern)
        rb_gen.gen(stream_output=False, open_editor=False, same_chord_in_section=True)
        confirm = input("Shuffle clips again? (y/n)")
        with open(os.path.join(dir_path, f"{song_name}_config.json"), "w") as f:
            json.dump(gen_config, f)
        if confirm == "y":
            gen_config["seed"] = random.randint(0, 2**32 - 1)
            print("Using seed:", gen_config['seed'])
            config.FIX_SEED = gen_config["seed"]
            continue
        else:
            break


if __name__ == "__main__":
    for sp in ('pattern1', 'pattern2', 'pattern3', 'pattern4'):
        for k in ('C', 'F', 'G', 'A#'):
            for i in range(2):
                generate_tomi_song(sp, k, i+1)

    for sp in ('pattern1', 'pattern2', 'pattern3', 'pattern4'):
        for k in ('C', 'F', 'G', 'A#'):
            for i in range(2):
                generate_standalone_llm_song(sp, k, i + 1)


    for sp in ('pattern1', 'pattern2', 'pattern3', 'pattern4'):
        for k in ('C', 'F', 'G', 'A#'):
            for i in range(2):
                generate_random_song(sp, k, i+1)

