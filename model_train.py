import torch
import os
import math
import sentencepiece
import wandb

import torch.distributed as distributed
import torch.multiprocessing as mp

from src.train import train
from src.conformer import Conformer
from src.scheduler import NoamAnnealing
from src.dataset import get_libri_speech_dataset, get_golos_dataset, AudioDataset, collate_fn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


def get_dataloaders(rank, world_size, batch_size=128, num_workers=8, base_path="./datasets", verbose=False):
    
    libri_speech_base_path = os.path.join(base_path, 'LibriSpeech_ds')

    libri_speech_dev = get_libri_speech_dataset(libri_speech_base_path, split='dev')
    libri_speech_train = get_libri_speech_dataset(libri_speech_base_path, split='train')
    libri_speech_test = get_libri_speech_dataset(libri_speech_base_path, split='test')
    
    if verbose:
        print('Loaded {0:d} objects'.format(len(libri_speech_dev['audio_path'])))
        print('Loaded {0:d} objects'.format(len(libri_speech_train['audio_path'])))
        print('Loaded {0:d} objects'.format(len(libri_speech_test['audio_path'])))

    # Load tokenizer model
    tokenizer = sentencepiece.SentencePieceProcessor(model_file='tokenizer.model')
    
    # creating training dataset
    train_dataset = AudioDataset(libri_speech_train, tokenizer, min_duration=1.36, max_duration=10.96)
    
    # creating validation datasets
    libri_speech_dev_ds = AudioDataset(libri_speech_dev, tokenizer, min_duration=1.36, max_duration=10.96)
    libri_speech_test_ds = AudioDataset(libri_speech_test, tokenizer, min_duration=1.36, max_duration=10.96)
    
    validation_datasets = {
        "libri_speech_dev": libri_speech_dev_ds,
        "libri_speech_test": libri_speech_test_ds
    }

    # creating train dataloader
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False, collate_fn=collate_fn,
        sampler=DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    )
    
    # creating validation dataloaders
    validate_dataloaders = {
        name: DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=False, collate_fn=collate_fn,
                         sampler=DistributedSampler(dataset, num_replicas=world_size,
                                                    rank=rank, shuffle=True, drop_last=False))
        for name, dataset in validation_datasets.items()
    }
    return tokenizer, train_dataloader, validate_dataloaders


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "42069"
    
    # creating process group 
    distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    distibuted.destroy_process_group()
    

def main(rank, world_size, device_type, base_path="./datasets", run=None):
    # staring processes
    setup(rank, world_size)
    
    device = torch.device(device_type, rank)
    
    batch_size, num_workers = 64, 8
    total_batch_size = 2000
    grad_scale = None
    epoch_num = 50
    accumulate_grad_batches = math.ceil(total_batch_size / (batch_size * world_size))  # calculating accumulation length
    verbose = (rank == 0)  # wether to output or not
    
    if verbose:
        print(f"Batch size: {batch_size}; Accumulating gradient for {accumulate_grad_batches}, " + 
              "recieving ~{total_batch_size} total batch_size\n Number of worhers: {num_workers}")
    
    # creating all dataloaders
    tokenizer, train_dataloader, validation_dataloader_dict = get_dataloaders(rank, world_size, batch_size, num_workers, base_path, verbose)
    
    # creating model
    model = Conformer()
    model.to(device)
    
    # replacing batch normalization with synchronized batch normalization
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    ddp_model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=2, weight_decay=1e-3)
    scheduler = NoamAnnealing(optimizer, d_model=model.d_model, warmup_steps=5000)
    
    train(model, tokenizer, grad_scale, optimizer,
          scheduler, epoch_num, train_dataloader, validation_dataloader_dict,
          device, accumulate_grad_batches=accumulate_grad_batches, model_dir="model_train", verbose=verbose, run=run)
    
    # ending processes
    cleanup()


if __name__ == "__main__":
    base_path = '/home/jupyter/mnt/datasets'
    device_type = "cuda"
    run = wandb.init(project="Conformer Model")
    world_size = torch.cuda.device_count()
    
    mp.spawn(main, args=(world_size, device_type, base_path, run), nprocs=world_size, join=True)