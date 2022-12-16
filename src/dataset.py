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
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, BatchSampler, SequentialSampler, RandomSampler, Sampler



def _strip_text(text):
    """Normalize text:
        1. Replace diacritic symbols
        2. Replace all non-cyrillic letter with space
        3. Strip excess spaces 
        :param str text:
        :return str:
    """
    ### YOUR CODE HERE
    
    text = text.lower()
    text = text.strip()
    res_text = ''
    for i in text:
      # If char is cirillic letter or whitespace we append it
      if 1072 <= ord(i) <= 1103 or (ord(i) == 32 and res_text[-1] != ' '):
        res_text += i

    res_text = res_text.strip()

    return res_text


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
    full_path = manifest_path
    with open(full_path, "r", encoding="utf-8") as file:
      for raw_json in file.readlines():
        parsed_json = json.loads(raw_json)
        wav_paths.append(os.path.join(base_path, parsed_json["audio_filepath"]))
        durations.append(parsed_json["duration"])
        texts.append(parsed_json["text"])
    

    # Apply text preprocessing
    ### YOUR CODE HERE
    for i in range(len(texts)):
      texts[i] = _strip_text(texts[i])
    

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
    resemple = torchaudio.transforms.Resample(orig_sample_rate, desired_sample_rate)
    audio_data = resemple(audio_data)

    # Average out audio channels
    ### YOUR CODE HERE
    audio_data = audio_data.mean(0)

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
        if not max_duration:
          max_duration = data["duration"].max()
        if not min_duration:
          min_duration = 0
        
        mask_max = data["duration"] <= max_duration
        mask_min = data["duration"] >= min_duration
        data = data[mask_min * mask_max]

        # Sort data w.r.t. duration
        ### YOUR CODE HERE
        
        self.data = data.sort_values("duration")
        
        self.tokenizer = tokenizer

        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration

        # Tokenize all texts
        ### YOUR CODE HERE
        
        self.data['tokens'] = tokenizer.encode_as_ids(list(self.data["text"]))

    def __getitem__(self, idx):
        """
            :param int idx: 
            :return Tuple[str, torch.FloatTensor, int, str, torch.LongTensor, int]: (audio_path, audio, audio_len, text, tokens, tokens_len)
        """
        # Load audio with desired sample rate
        ### YOUR CODE HERE
        obj = self.data.iloc[idx]
        audio, audio_len = open_audio(obj["audio_path"], 16000)

        return (obj["audio_path"], audio, audio_len, obj["text"], torch.tensor(obj["tokens"], dtype = torch.long), len(obj["tokens"]))

    def __len__(self):
        ### YOUR CODE HERE
        return self.data.shape[0]


def collate_fn(batch):
    """
        :param: List[Tuple[str, torch.FloatTensor, int, str, torch.LongTensor, int]] batch: list of elements with length=batch_size
        :return dict:
    """
    # Pad and concatenate audios. Use torch.nn.utils.rnn.pad_sequence
    ### YOUR CODE HERE
    batch_size = len(batch)

    batch_audio = pad_sequence([batch[i][1] for i in range(batch_size)])
    # Pad and concatenate tokens. Use torch.nn.utils.rnn.pad_sequence
    ### YOUR CODE HERE
    batch_tokens = pad_sequence([batch[i][4] for i in range(batch_size)])
    
    # Convert ints to torch.LongTensors
    ### YOUR CODE HERE
    batch_audio_len = torch.tensor([batch[i][2] for i in range(batch_size)], dtype=torch.long)
    ### YOUR CODE HERE
    batch_tokens_len = torch.tensor([batch[i][5] for i in range(batch_size)], dtype=torch.long)

    return {
        'audio_path': [batch[i][0] for i in range(batch_size)],
        'audio': batch_audio,
        'audio_len': batch_audio_len,
        'text': [batch[i][3] for i in range(batch_size)],
        'tokens': batch_tokens,
        'tokens_len': batch_tokens_len
    }
