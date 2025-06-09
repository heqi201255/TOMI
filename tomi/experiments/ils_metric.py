import os
import numpy as np
from numpy.linalg import norm
import torch
import pandas as pd
from pydub import AudioSegment
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import math
from collections import defaultdict


# If you want to use this ILS implementation on your own music audios, first get the song structure and change the following
# section_patterns dict, the key should be a name of the structure, the values should be a list of tuples, where each tuple
# consists of 3 elements: section name (any string), section role, and the bar length of the section. See the bottom of this
# module for more instructions.
section_patterns = {
    'pattern1': [
        ("Intro", "Intro", 8),
        ("Verse1", "Verse", 16),
        ("PreChorus", "PreChorus", 8),
        ("Chorus1", "Chorus", 16),
        ("Verse2", "Verse", 16),
        ("PreChorus", "PreChorus", 8),
        ("Chorus2", "Chorus", 16),
        ("Bridge", "Bridge", 8),
        ("Chorus3", "Chorus", 16),
        ("Outro", "Outro", 8)
    ],
    'pattern2': [
        ("Intro", "Intro", 8),
        ("Verse1", "Verse", 16),
        ("Chorus1", "Chorus", 8),
        ("Verse2", "Verse", 16),
        ("Chorus2", "Chorus", 8),
        ("Bridge", "Bridge", 8),
        ("Chorus3", "Chorus", 8),
        ("Outro", "Outro", 8)
    ],
    'pattern3': [
        ("Intro", "Intro", 8),
        ("Verse1", "Verse", 16),
        ("PreChorus1", "PreChorus", 8),
        ("Chorus1", "Chorus", 16),
        ("Verse2", "Verse", 16),
        ("PreChorus2", "PreChorus", 8),
        ("Chorus2", "Chorus", 16),
        ("Bridge", "Bridge", 8),
        ("Chorus3", "Chorus", 16),
        ("Outro", "Outro", 8)
    ],
    'pattern4': [
        ("Intro", "Intro", 8),
        ("Chorus", "Chorus", 8),
        ("Verse", "Verse", 16),
        ("PreChorus", "PreChorus", 4),
        ("Chorus", "Chorus", 8),
        ("Verse", "Verse", 16),
        ("PreChorus", "PreChorus", 4),
        ("Chorus", "Chorus", 8),
        ("Outro", "Outro", 8)
    ]
}


def convert_to_tensor(audio):
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(audio.sample_width)
    raw_data = np.frombuffer(audio.raw_data, dtype=dtype)
    max_val = np.iinfo(dtype).max
    normalized_data = raw_data.astype(np.float32) / max_val
    return torch.from_numpy(normalized_data).float()


data_root = "./song_dataset_wav"
SAMPLE_RATE = None
frechet = None
model = None
processor = None
mel_spectrogram = None
model_name = "MERT"


def init_module(m_name: str):
    global model_name, SAMPLE_RATE, frechet, processor, mel_spectrogram, model
    model_name = m_name
    SAMPLE_RATE = None
    frechet = None
    model = None
    processor = None
    mel_spectrogram = None
    if model_name == 'MERT':
        SAMPLE_RATE = 24000
        model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    elif model_name == 'MelSpectrogram':
        SAMPLE_RATE = 32000
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            win_length=None,
            hop_length=512,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm='slaney',
            onesided=True,
            n_mels=256,
            mel_scale="htk",
        )
    elif model_name == 'Waveform':
        SAMPLE_RATE = 32000


def get_embeddings(audio_list: list):
    global model_name, frechet, processor, mel_spectrogram, SAMPLE_RATE, model
    print(model_name)
    if model_name == 'VGGish':
        return [frechet.get_embeddings([x], SAMPLE_RATE) for x in audio_list]
    elif model_name == 'MERT':
        embeddings = []
        for audio in audio_list:
            audio_data = convert_to_tensor(audio)
            inputs = processor(audio_data, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            embeddings.append(outputs.hidden_states[6][0].numpy())
        return embeddings
    elif model_name == 'MelSpectrogram':
        return [mel_spectrogram(convert_to_tensor(audio)).T.numpy() for audio in audio_list]
    elif model_name == 'Waveform':
        return [a.reshape(1, -1) for a in audio_list]
    return None


def bars2sec(bars, bpm = 120, ppq: int = 96) -> float:
    total_ticks = bars * 16 * 24
    return total_ticks * (60 / (bpm * ppq))


def process_all_songs():
    stats = defaultdict(dict)
    for model_n in os.listdir(data_root):
        model_music_path = os.path.join(data_root, model_n)
        if not os.path.isdir(model_music_path):
            continue
        for pattern in os.listdir(model_music_path):
            pattern_path = os.path.join(model_music_path, pattern)
            if not os.path.isdir(pattern_path):
                continue
            for audio_file in os.listdir(pattern_path):
                audio_file_path = os.path.join(pattern_path, audio_file)
                if not audio_file.endswith(".wav"):
                    continue
                print("Processing", audio_file)
                score = process_audio(pattern, audio_file_path, gen_model=model_n, plot=True)[0]
                stats[model_n][audio_file] = score
    os.makedirs("./results", exist_ok=True)
    for model_n, result in stats.items():
        x = [j for j in result.items()]
        s = {"file": [n[0] for n in x], "ils": [n[1] for n in x]}
        pd.DataFrame.from_dict(s).to_csv(f"./results/ILS_{model_name}_{model_n}.csv", index=None)


def ils_results():
    results = {}
    root = f"./results/"
    for file in os.listdir(root):
        if not file.endswith(".csv"):
            continue
        _, ils_model, gen_model = file.rstrip(".csv").split("_")
        if ils_model not in results:
            results[ils_model] = {}
        results[ils_model][gen_model] = pd.read_csv(os.path.join(root, file))['ils'].values.mean()
    gen_models = list(list(results.values())[0].keys())
    for imodel, v in results.items():
        results[imodel] = [v[x] for x in gen_models]
    df = pd.DataFrame.from_dict(results)
    df.index = gen_models
    df.to_csv(f"./ILS_results.csv")


def ils_statistics():
    results = {}
    root = f"./results/"
    for file in os.listdir(root):
        if not file.endswith(".csv") or file == "ILS_results.csv":
            continue
        _, ils_model, gen_model = file.rstrip(".csv").split("_")
        file_path = os.path.join(root, file)
        scores = pd.read_csv(file_path)['ils'].values
        # scores = scores * 2 - 1
        mean, std = scores.mean(), scores.std()
        if ils_model not in results:
            results[ils_model] = {}
        results[ils_model][gen_model] = [mean, std]
    r2 = {}
    gen_models = list(list(results.values())[0].keys())
    for imodel, v in results.items():
        r2[imodel+"_mean"] = [v[x][0] for x in gen_models]
        r2[imodel+"_std"] = [v[x][1] for x in gen_models]
    df = pd.DataFrame.from_dict(r2)
    df.index = gen_models
    df.to_csv(f"./ILS_stats.csv")


def cosine_similarity(tensor1: np.ndarray, tensor2: np.ndarray) -> float:
    if tensor1.shape != tensor2.shape:
        l = min(tensor1.shape[0], tensor2.shape[0])
        t1 = tensor1[:l]
        t2 = tensor2[:l]
    else:
        t1 = tensor1
        t2 = tensor2
    if len(tensor1.shape) == 2:
        cs = np.sum(t1*t2, axis=1)/(norm(t1, axis=1)*norm(t2, axis=1))
    else:
        cs = np.dot(t1,t2)/(norm(t1)*norm(t2))
    return cs.mean() if isinstance(cs, np.ndarray) else cs


def cosine_similarity_matrix(arr):
    if isinstance(arr, list):
        similarity_matrix = [[cosine_similarity(a, a2) for a2 in arr] for a in arr]
        similarity_matrix = np.array(similarity_matrix)
    else:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        normalized_arr = arr / norms
        similarity_matrix = np.dot(normalized_arr, normalized_arr.T)
    return similarity_matrix


def slice_audio(audio: AudioSegment, pattern: str, by_second: bool = True, convert_to_numpy: bool = True):
    audio_sec = []
    if by_second:
        total_seconds = bars2sec(sum(x[2] for x in section_patterns[pattern]))
        if convert_to_numpy:
            audio_sec = [np.array(audio[int(i*1000):int((i+1)*1000)].get_array_of_samples()) for i in range(int(total_seconds))]
        else:
            audio_sec = [audio[int(i*1000):int((i+1)*1000)] for i in range(int(total_seconds))]
    else:
        start = 0
        for sec in section_patterns[pattern]:
            time = bars2sec(sec[2])
            end = start + time
            audio_sec.append(np.array(audio[int(start*1000):int(end*1000)].get_array_of_samples()))
            start = end
    return audio_sec


def process_audio(pattern: str, file: str, gen_model: str, plot: bool = False, output_plot_params: bool = False):
    global model_name, SAMPLE_RATE
    audio = AudioSegment.from_wav(file).set_channels(1).set_frame_rate(SAMPLE_RATE)
    convert_to_numpy = model_name == "VGGish" or model_name == "Waveform"
    audio_sec = slice_audio(audio, pattern, convert_to_numpy=convert_to_numpy)
    embeds = get_embeddings(audio_sec)
    freq = len(embeds[0])
    print(f"Frequency: {freq}")
    embeds = np.vstack(embeds)
    cosine_sim = cosine_similarity_matrix(embeds)
    ils_score, ticks, labels, rects = calc_ils_score(pattern, cosine_sim, freq)
    print(f"ILS Score: {ils_score}")
    mn = "MS" if model_name == "MelSpectrogram" else "WF" if model_name == "Waveform" else model_name
    title = r'$\text{ILS}_\mathrm{'+gen_model+r'}^\mathrm{'+mn+r'}$'+f" = {ils_score:.3f}"
    if plot:
        plot_sim_matrix(cosine_sim, title=title, ticks=ticks, labels=labels, file=f"{mn}_{file.split('/')[-1].rstrip('.wav')}")
    if output_plot_params:
        return ils_score, cosine_sim, title, ticks, labels, rects
    return ils_score, cosine_sim, title


def get_section_block_segment(matrix: np.ndarray, pattern: str, section_x: int, section_y: int, freq: int):
    sp = section_patterns[pattern]
    xstart = bars2sec(sum(x[2] for x in sp[:section_x]))
    xend = xstart + bars2sec(sp[section_x][2])
    ystart = bars2sec(sum(x[2] for x in sp[:section_y]))
    yend = ystart + bars2sec(sp[section_y][2])
    xstart = int(xstart*freq)
    xend = int(xend*freq)
    ystart = int(ystart*freq)
    yend = int(yend*freq)
    return matrix[xstart:xend, ystart:yend]


def calc_ils_score(pattern: str, sim_matrix: np.ndarray, freq_of_one_second: int):
    def get_section_block(section_x: int, section_y: int):
        sp = section_patterns[pattern]
        xstart = bars2sec(sum(x[2] for x in sp[:section_x]))
        xend = xstart + bars2sec(sp[section_x][2])
        ystart = bars2sec(sum(x[2] for x in sp[:section_y]))
        yend = ystart + bars2sec(sp[section_y][2])
        xstart = int(xstart*freq_of_one_second)
        xend = int(xend*freq_of_one_second)
        ystart = int(ystart*freq_of_one_second)
        yend = int(yend*freq_of_one_second)
        block = normalized_sim_matrix[xstart:xend, ystart:yend]
        x_indices, y_indices = np.ogrid[xstart:xend, ystart:yend]
        block = block[x_indices != y_indices]
        return block, (xstart, ystart, xend-xstart, yend-ystart)
    normalized_sim_matrix = sim_matrix
    exclude_vals = []
    vals = []
    ticks = [0]
    labels = []
    rects = []
    for i, (sec_name, sec_type, sec_bars) in enumerate(section_patterns[pattern]):
        labels.append(sec_name)
        ticks.append(int(bars2sec(sec_bars)*freq_of_one_second)+ticks[-1])
        for i2, (sec_name2, sec_type2, sec_bars2) in enumerate(section_patterns[pattern]):
            block, block_rect = get_section_block(i, i2)
            nan_mask = np.isnan(block)
            if sec_type == sec_type2:
                vals.extend(block[~nan_mask].flatten())
                rects.append(block_rect)
            else:
                exclude_vals.extend(block[~nan_mask].flatten())
    n1 = len(vals)
    n2 = len(exclude_vals)
    vals = np.array(vals)
    exclude_vals = np.array(exclude_vals)
    mean1 = vals.mean()
    mean2 = exclude_vals.mean()
    var1 = vals.var(ddof=1)
    var2 = exclude_vals.var(ddof=1)
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    # Cohen's d
    d = (mean1 - mean2) / pooled_std
    return d, ticks, labels, rects


def convert_labels(labels: list[str]):
    result = []
    for label in labels:
        i = None
        if label[-1].isnumeric():
            label, i = label[:-1], label[-1]
        label = label[0].lower()
        result.append((label + i)  if i is not None else label)
    return result


def plot_sim_matrix(sim_matrix: np.ndarray, title: str = None, ticks: list = None, labels: list = None, ax_index: int = 0, vmin = -1, vmax = 1, ax = None, fig: plt.Figure = None, rects: list = None, file: str = None, fig_h = 16):
    single_plot = False
    if ax is None:
        single_plot = True
        plt.rcParams["text.usetex"] = True
        plt.rcParams['font.family'] = 'Times New Roman'
        fig, ax = plt.subplots(figsize=(20, 16))
        vmin = sim_matrix.min()
        vmax = sim_matrix.max()
    # norm = mcolors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(sim_matrix, cmap='Greys', norm=norm)
    matrix_size = sim_matrix.shape[0]
    rect = patches.Rectangle((0, 0), matrix_size, matrix_size, linewidth=6/16 * fig_h, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    ax.plot([0, matrix_size - 1], [0, matrix_size - 1], color='tab:red', linewidth=4/16 * fig_h)
    ax.set_xlim([0, matrix_size - 1])
    ax.set_ylim([matrix_size - 1, 0])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xticks(ticks if ticks is not None else [])
    ax.set_yticks(ticks if ticks is not None else [])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    labels = convert_labels(labels) if labels else []
    ax.grid(which='major', color='black', linestyle='--', linewidth=0.6)
    if rects is not None:
        for rect in rects:
            xstart, ystart, w, h = rect
            rect = patches.Rectangle((xstart, ystart), w, h, linewidth=4/16 * fig_h, edgecolor='yellow', facecolor='none', zorder=3, alpha=0.6)
            ax.add_patch(rect)
    ax.set_title(title, y=-0.15, fontsize=56/16 * fig_h) if title else None
    box = ax.get_position()
    ax.set_position([box.x0 - ax_index * 0.01, box.y0, box.width, box.height])
    box = ax.get_position()
    for i in range(len(ticks) - 1):
        x_pos = (ticks[i] + ticks[i + 1]) / 2
        y_pos = (ticks[i] + ticks[i + 1]) / 2
        fig_x, fig_y = ax.transData.transform((x_pos, y_pos))  # 转换到像素坐标
        fig_x, fig_y = fig.transFigure.inverted().transform((fig_x, fig_y))  # 转换到 Figure 归一化坐标
        ax.text(fig_x, 1 - box.y0 + 0.015, labels[i], ha='center', va='center', fontsize=45/16 * fig_h, transform=fig.transFigure)  # x轴标签
        ax.text(box.x0 - 0.008, fig_y, labels[i], ha='center', va='center', fontsize=45/16 * fig_h, rotation=90, transform=fig.transFigure)  # y轴标签
    ax.tick_params(axis='both',
                   length=15/16 * fig_h,    # 刻度线长度
                   width=4/16 * fig_h,      # 刻度线粗细
                   direction='out')  # 刻度线方向，可选 'in', 'out', 'inout'
    if single_plot:
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # 颜色条的位置
        # norm = plt.Normalize(vmin=vmin, vmax=vmax)
        # sm = plt.cm.ScalarMappable(cmap='Greys', norm=norm)
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Similarity', fontsize=48)
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        if file is not None:
            os.makedirs("./plots", exist_ok=True)
            plt.savefig(f"./plots/{file}.png")
        plt.show()

def plot_multiple_sim_matrices(sim_matrices, titles, ticks, labels, rects, figsize: tuple = (64, 16)):
    plt.rcParams['font.family'] = 'Times New Roman'
    sim_matrices = np.array(sim_matrices)
    min_v = sim_matrices.min()
    max_v = sim_matrices.max()
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    text_quantize = figsize[1] // 4
    for i, ax in enumerate(axes):
        plot_sim_matrix(sim_matrices[i], title=titles[i], ticks=ticks, labels=labels, ax_index=i, vmin=min_v, vmax=max_v, ax=ax, fig=fig, rects=rects, fig_h=figsize[1])
    # 添加一个共享 colorbar
    cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.7])  # 颜色条的位置
    norm = plt.Normalize(vmin=min_v, vmax=max_v)
    sm = plt.cm.ScalarMappable(cmap='Greys', norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax, ticks=[min_v, max_v])
    # cbar.ax.tick_params(labelsize=48)
    cbar.ax.tick_params(labelsize=12 * text_quantize)
    # cbar.set_label('Similarity', fontsize=48, labelpad=10)
    cbar.set_label('Similarity', fontsize=12 * text_quantize, labelpad=10/16 * figsize[1])
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    os.makedirs("./plots", exist_ok=True)
    plt.savefig('./plots/temp_plot.png', bbox_inches='tight', pad_inches=0.5)
    plt.show()


if __name__ == '__main__':
    # First, initialize the module with "MERT", "MelSpectrogram", or "Waveform".
    init_module("Waveform")
    # To get the final ILS results as tables:
    process_all_songs()
    ils_results()
    ils_statistics()
    # Example plot
    _, cosine_sim1, title1, ticks, labels, rects = process_audio("pattern3", "./song_dataset_wav/TOMI/pattern3/tomi_pattern3_C_1.wav", gen_model="TOMI", plot=False, output_plot_params=True)
    _, cosine_sim2, title2 = process_audio("pattern3", "./song_dataset_wav/MusicGen/pattern3/musicgen_pattern3_C_2.wav", gen_model="MusicGen", plot=False, output_plot_params=False)
    _, cosine_sim3, title3 = process_audio("pattern3", "./song_dataset_wav/StandaloneLLM/pattern3/standalonellm_pattern3_F_2.wav", gen_model="StandaloneLLM", plot=False, output_plot_params=False)
    _, cosine_sim4, title4 = process_audio("pattern3", "./song_dataset_wav/Random/pattern3/random_pattern3_F_2.wav", gen_model="Random", plot=False, output_plot_params=False)
    plot_multiple_sim_matrices([cosine_sim1, cosine_sim2, cosine_sim3, cosine_sim4], [title1, title2, title3, title4], ticks, labels, rects)