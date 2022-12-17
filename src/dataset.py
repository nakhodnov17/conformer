import os
import glob
import json
import regex
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import tqdm

import librosa

import pandas as pd

import torch
import torchaudio
from torch.utils.data import Dataset, BatchSampler, SequentialSampler, RandomSampler, Sampler


def _strip_text(text):
    """Normalize text:
        1. Replace diacritic symbols
        2. Replace all non-cyrillic letter with space
        3. Strip excess spaces
        :param str text:
        :return str:
    """
    s = ""
    ### YOUR CODE HERE
    text = text.lower()
    for i in range(len(text)):
        if (ord(text[i]) >= ord("а") and ord(text[i]) <= ord("я")):
            s += text[i]
        else:
            s += " "
    while text.count("  ") > 0:
        text = text.replace("  ", " ")
    text = text.strip()
    text = text.replace("ё", "е")
    return text


def _get_manifest_dataset(base_path, manifest_path):
    """Load data from .jsonl manifest file
        :param str base_path: suffix for each audio_path in manifest
        :param str manifest_path: path to manifest file
        :return pandas.DataFrame:
    """

    texts = []
    wav_paths = []
    durations = []
    # Read manifest file. Parse each line as json and save needed values
    ### YOUR CODE HERE

    real_path = manifest_path
    manifest = open(real_path, "r")
    a = manifest.readlines()
    for j in a:
        stroka = json.loads(j)
        wav_paths.append(os.path.join(base_path, stroka["audio_filepath"]))
        durations.append(stroka["duration"])
        text = _strip_text(stroka["text"])
        texts.append(stroka["text"])

    # Apply text preprocessing
    ### YOUR CODE HERE
    ...

    return pd.DataFrame.from_dict({
        'audio_path': wav_paths,
        'text': texts,
        'duration': durations
    })


def get_libri_speech_dataset(base_path, split='train'):
    assert split in {'dev', 'test', 'train'}

    base_path = os.path.join(base_path, split)
    manifest_path = os.path.join(base_path, 'manifest.json')

    return _get_manifest_dataset(base_path, manifest_path)


def get_golos_dataset(base_path, split='train'):
    assert split in {'train', 'test/crowd', 'test/farfield'}

    if split in {'test/crowd', 'test/farfield'}:
        base_path = os.path.join(base_path, split)
        manifest_path = os.path.join(base_path, 'manifest.jsonl')

        return _get_manifest_dataset(base_path, manifest_path)
    else:
        base_path = os.path.join(base_path, 'train')
        manifest_path = os.path.join(base_path, 'manifest.jsonl')

        return _get_manifest_dataset(base_path, manifest_path)


def open_audio(audio_path, desired_sample_rate):
    """ Open and resample audio, average across channels
        :param str audio_path: path to audio
        :param in desired_sample_rate: the sampling rate to which we would like to convert the audio
        :return Tuple[torch.Tensor, int]: (audio, audio_len):
            audion: 1D tensor with shape (num_timesteps)
            audio_len: int, len of audio
    """

    # Load audio. Use torchaudio
    ### YOUR CODE HERE

    audio_data, orig_sample_rate = torchaudio.load(audio_path)

    # Resample audio. Use torchaudio.transforms
    ### YOUR CODE HERE
    audio_data = torchaudio.transforms.Resample(orig_sample_rate, desired_sample_rate)

    ...

    # Average out audio channels
    ### YOUR CODE HERE
    audio_data = audio_data.mean(0)
    ...

    return audio_data, audio_data.shape[0]


class AudioDataset(Dataset):
    def __init__(self, data, tokenizer, sample_rate=16000, min_duration=None, max_duration=None):
        """
            :param pandas.DataFrame data:
            :param sentencepiece.SentencePieceProcessor tokenizer:
            :param float sample_rate:
            :param Optional[float] min_duration:
            :param Optional[float] max_duration:
        """
        super().__init__()

        # Filter out all entities that are longer then max_duration or shorter min_duration
        ### YOUR CODE HERE
        if (max_duration == None):
            max_duration = max(data["duration"])
        if (min_duration == None):
            min_duration = 0

        a = data
        a.clear()
        for i in data:
            if (i["duration"] > max_duration or i["duration"] < min_duration):
                continue
            a.append(i)



        # Sort data w.r.t. duration
        ### YOUR CODE HERE
        ...
        self.data = a.sort_values("duration")

        self.tokenizer = tokenizer

        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration

        # Tokenize all texts
        ### YOUR CODE HERE
        self.data['tokens'] = sp_tokenizer.encode_as_ids(["text"])
        
    def __getitem__(self, idx):
        """
            :param int idx:
            :return Tuple[str, torch.FloatTensor, int, str, torch.LongTensor, int]: (audio_path, audio, audio_len, text, tokens, tokens_len)
        """
        # Load audio with desired sample rate
        ### YOUR CODE HERE
        audio, audio_len = ...

        return ...

    def __len__(self):
        ### YOUR CODE HERE
        ...


def collate_fn(batch):
    """
        :param: List[Tuple[str, torch.FloatTensor, int, str, torch.LongTensor, int]] batch: list of elements with length=batch_size
        :return dict:
    """
    # Pad and concatenate audios. Use torch.nn.utils.rnn.pad_sequence
    ### YOUR CODE HERE
    batch_audio = ...
    # Pad and concatenate tokens. Use torch.nn.utils.rnn.pad_sequence
    ### YOUR CODE HERE
    batch_tokens = ...

    # Convert ints to torch.LongTensors
    ### YOUR CODE HERE
    batch_audio_len = ...
    ### YOUR CODE HERE
    batch_tokens_len = ...

    return {
        'audio_path': batch_audio_path,
        'audio': batch_audio,
        'audio_len': batch_audio_len,
        'text': batch_text,
        'tokens': batch_tokens,
        'tokens_len': batch_tokens_len
    }
