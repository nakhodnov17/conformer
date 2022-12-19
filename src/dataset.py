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
    ### YOUR CODE HERE

    text = text.lower()
    text = text.replace('ё', 'е')
    text0 = []
    for s in text:
        if s < 'а' or 'я' < s:
            s = ' '
        text0.append(s)
    text = ''.join(text0)
    text = text.strip()
    text0 = []
    pred = 'a'
    for s in text:
        if pred == ' ' and s == ' ':
            continue
        text0.append(s)
        pred = s
    text = ''.join(text0)

    ...
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
    ...
    
    with open(manifest_path, 'r', encoding = 'utf-8') as file:
        lines = file.readlines()
        for l in lines:
            s = json.loads(l)
            texts.append(_strip_text(s['text']))
            path = os.path.join(base_path, s['audio_filepath'])
            wav_paths.append(path)
            durations.append(s['duration']) 
        
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
    audio_data = torchaudio.transforms.Resample(orig_sample_rate, desired_sample_rate)(audio_data)

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
        
        
        if (max_duration == None):
            max_duration = data["duration"].max()
        if (min_duration == None):
            min_duration = 0

            
            
        a = data["duration"] >= min_duration
        b = data["duration"] <= max_duration
        
        data = data[a * b]
        
        
        # Sort data w.r.t. duration
        ### YOUR CODE HERE
        
        self.data = data.sort_values("duration")

        self.tokenizer = tokenizer

        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration

        # Tokenize all texts
        ### YOUR CODE HERE
#         print(list(self.data['text']))
        self.data['tokens'] = self.tokenizer.encode_as_ids(list(self.data['text']))

    def __getitem__(self, idx):
        """
            :param int idx: 
            :return Tuple[str, torch.FloatTensor, int, str, torch.LongTensor, int]: (audio_path, audio, audio_len, text, tokens, tokens_len)
        """
        # Load audio with desired sample rate
#         open_audio(self.data.iloc[idx]['audio_path'], self.sample_rate)
        ### YOUR CODE HERE
        audio, audio_len = open_audio(self.data.iloc[idx]['audio_path'], self.sample_rate)
        
        
        return (
            self.data.iloc[idx]['audio_path'], audio, audio_len, 
            self.data.iloc[idx]['text'], torch.tensor(self.data.iloc[idx]['tokens'], dtype = torch.long), len(self.data.iloc[idx]['tokens'])
        )

    def __len__(self):
        ### YOUR CODE HERE
        return len(self.data)
        ...


def collate_fn(batch):
    """
        :param: List[Tuple[str, torch.FloatTensor, int, str, torch.LongTensor, int]] batch: list of elements with length=batch_size
        :return dict:
    """
    # Pad and concatenate audios. Use torch.nn.utils.rnn.pad_sequence
    ### YOUR CODE HERE
    batch_audio = torch.nn.utils.rnn.pad_sequence([bath_el[1] for bath_el in batch], batch_first = True)
    
    # Pad and concatenate tokens. Use torch.nn.utils.rnn.pad_sequence
    ### YOUR CODE HERE
    batch_tokens = torch.nn.utils.rnn.pad_sequence([bath_el[4] for bath_el in batch], batch_first = True)
    
    # Convert ints to torch.LongTensors
    ### YOUR CODE HERE
    s = torch.tensor([bath_el[2] for bath_el in batch], dtype = torch.long)
    batch_audio_len = s
    
    ### YOUR CODE HERE
    s = torch.tensor([bath_el[5] for bath_el in batch], dtype = torch.long)
    batch_tokens_len = s
    
    batch_audio_path = batch[0]
    
    batch_text = batch[3]

    return {
        'audio_path': batch_audio_path,
        'audio': batch_audio,
        'audio_len': batch_audio_len,
        'text': batch_text,
        'tokens': batch_tokens,
        'tokens_len': batch_tokens_len
    }


