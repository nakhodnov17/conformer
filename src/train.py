import torch
import os

from tqdm.notebook import tqdm 
from src.metrics import WERMetric, CTCLossMetric
from src.scheduler import NoamAnnealing


@torch.no_grad() 
def evaluate(model, tokenizer, dataloader, device, compute_frequency=10, verbose=True, leave_pbar=True):
    model.eval()
    
    wermetric = WERMetric(len(tokenizer), tokenizer)
    ctclossmetric = CTCLossMetric()
    
    wermetric.to(device)
    ctclossmetric.to(device)
    
    for batch in (eval_pbar := tqdm(dataloader, leave=leave_pbar, total=len(dataloader))):
        batch["audio"] = batch["audio"].to(device)
        batch["audio_len"] = batch["audio_len"].to(device)
        batch["tokens"] = batch["tokens"].to(device)
        batch["tokens_len"] = batch["tokens_len"].to(device)
        
        # inference through model
        log_probs, encoding_lengths, greedy_predictions = model(batch["audio"], batch["audio_len"])
        
        # Calculate CTCLoss (via conformer.loss) -> tensor
        loss = model.loss(torch.transpose(log_probs, 0, 1), batch["tokens"], encoding_lengths, batch["tokens_len"])
        
        # update CTC and WER metrics (from Loss use sum)
        wermetric.update(log_probs, encoding_lengths, batch["text"])
        ctclossmetric.update(loss.sum(), len(loss))
        
    # metrics.compute --> return
    return wermetric.compute(), ctclossmetric.compute()


def train( 
        model, tokenizer, grad_scaler, optimizer, scheduler, 
        num_epochs, train_dataloader, val_dataloaders, device, 
        accumulate_grad_batches=1, compute_frequency=10, model_dir=None, verbose=True 
        ):
    wermetric = WERMetric(len(tokenizer), tokenizer)
    ctclossmetric = CTCLossMetric()
    
    wermetric.to(device)
    ctclossmetric.to(device)
    model.to(device)
    
    for epoch in (epoch_pbar := tqdm(range(num_epochs), total=num_epochs)):
        epoch_pbar.set_description("Processing epoch num %i" % epoch)
        
        model.train()
        optimizer.zero_grad()
        
        for idx, batch in (batch_pbar := tqdm(enumerate(train_dataloader), leave=False, total=len(train_dataloader))):
            
            # setting appropriate device for batch
            batch["audio"] = batch["audio"].to(device)
            batch["audio_len"] = batch["audio_len"].to(device)
            batch["tokens"] = batch["tokens"].to(device)
            batch["tokens_len"] = batch["tokens_len"].to(device)
            
            # forward pass through model
            log_probs, encoding_lengths, greedy = model(batch["audio"], batch["audio_len"])
            
            # calculating loss
            loss = model.loss(torch.transpose(log_probs, 0, 1), batch["tokens"], encoding_lengths, batch["tokens_len"])  # tensor (batch) -> need .sum() for ctcloss and .mean() for gradients
            
            # updating metrics
            wermetric.update(log_probs, encoding_lengths, batch["text"])
            ctclossmetric.update(loss.sum(), len(loss))
            
            
            # calculating gradients
            (loss.mean()).backward()
            
            # making gradient descent step w.r.t. accumulation
            if (idx + 1) % accumulate_grad_batches == 0:

                # making gradient descent step
                optimizer.step()

                # updating learning rate
                scheduler.step()
                
                # resetting gradinets
                optimizer.zero_grad()
            
            if (idx + 1) % compute_frequency == 0:
                batch_pbar.set_postfix(wer=wermetric.compute(), ctcloss=ctclossmetric.compute())
        
        
        # evaluating model to see its progress
        for key, dataloader in val_dataloaders.items():
            wer_res, ctc_res = evaluate(model, tokenizer, dataloader, device, leave_pbar=False)
            epoch_pbar.set_postfix(validation=key, wer=wer_res, ctc_loss=ctc_res)
        
        # saving weights to 
        current_weights = model.state_dict()
        file_name = os.path.join(model_dir, "model-" + str(epoch) + "pt")
        torch.save(current_weights, file_name)
        
        # deletting old weights
        if epoch > 4:
            os.remove(os.path.join(model_dir, "model-" + str(epoch - 5) + "pt"))
        
        