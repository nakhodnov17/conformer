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
    text = text.replace('ё', 'е')
    norm_text = ''
    for sym in text:
        if sym < 'а' or sym > 'я':
            norm_text += ' '
        else:
            norm_text += sym
    last_sym = 'a'
    text = ''
    for sym in norm_text:
        if sym != ' ' or last_sym != ' ':
            text += sym
        last_sym = sym
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
    with open(manifest_path, 'r', encoding='utf-8') as manifest_file:
        for string in manifest_file:
            manifest_data = json.loads(string)
            texts.append(_strip_text(manifest_data["text"]))
            wav_paths.append(os.path.join(base_path, manifest_data["audio_filepath"]))
            durations.append(manifest_data["duration"])

    # Apply text preprocessing
    ### YOUR CODE HERE
    

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
    
    transform = torchaudio.transforms.Resample(orig_sample_rate, desired_sample_rate)
    audio_data = transform(audio_data)

    # Average out audio channels
    ### YOUR CODE HERE
    
    audio_data = torch.mean(audio_data, 0)

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
        
        mask1 = data['duration'] >= min_duration
        mask2 = data['duration'] <= max_duration
        mask = mask1 & mask2
        data = data[mask]
        
        # Sort data w.r.t. duration
        ### YOUR CODE HERE
        data = data.sort_values(by=['duration'])
        self.data = data
        # print(self.data.shape)
        
        self.tokenizer = tokenizer

        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration

        # Tokenize all texts
        ### YOUR CODE HERE
        self.data['tokens'] = self.tokenizer.encode_as_ids(list(data['text']))
        
    def __getitem__(self, idx):
        """
            :param int idx: 
            :return Tuple[str, torch.FloatTensor, int, str, torch.LongTensor, int]: (audio_path, audio, audio_len, text, tokens, tokens_len)
        """
        # Load audio with desired sample rate
        ### YOUR CODE HERE
        audio, audio_len = open_audio(self.data['audio_path'][idx], self.sample_rate)

        return (self.data['audio_path'][idx],
                audio, audio_len,
                self.data['text'][idx],
                torch.LongTensor(self.data['tokens'][idx]),
                len(self.data['tokens'][idx]))

    def __len__(self):
        ### YOUR CODE HERE
        return len(self.data["duration"])


def collate_fn(batch):
    """
        (audio_path, audio, audio_len, text, tokens, tokens_len)
        :param: List[Tuple[str, torch.FloatTensor, int, str, torch.LongTensor, int]] batch: list of elements with length=batch_size
        :return dict:
    """
    # Pad and concatenate audios. Use torch.nn.utils.rnn.pad_sequence
    ### YOUR CODE HERE
    batch_audio = torch.nn.utils.rnn.pad_sequence([x[1] for x in batch])
    # Pad and concatenate tokens. Use torch.nn.utils.rnn.pad_sequence
    ### YOUR CODE HERE
    batch_tokens = torch.nn.utils.rnn.pad_sequence([x[4] for x in batch])
    
    # Convert ints to torch.LongTensors
    ### YOUR CODE HERE
    batch_audio_len = torch.tensor([x[2] for x in batch], dtype=torch.long)
    ### YOUR CODE HERE
    batch_tokens_len = torch.tensor([x[5] for x in batch], dtype=torch.long)
    batch_audio_path = [x[0] for x in batch]
    batch_text = [x[3] for x in batch]
    
    return {
        'audio_path': batch_audio_path,
        'audio': batch_audio,
        'audio_len': batch_audio_len,
        'text': batch_text,
        'tokens': batch_tokens,
        'tokens_len': batch_tokens_len
    }
