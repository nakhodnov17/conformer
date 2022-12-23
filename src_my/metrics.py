import torch
import torchmetrics

import editdistance

import numpy as np
import math
from collections import defaultdict
from ctcdecode import CTCBeamDecoder

@torch.no_grad()
def ctc_greedy_decoding(logits, logits_len, blank_id, tokenizer):
    """Decode text from logits using greedy strategy. 
    Collapse all repeated tokens and then remove all blanks.
        :param torch.FloatTensor logits: (batch, time, vocabulary)
        :param torch.LongTensor logits_len: (batch)
        :param int blank_id:
        :param sentencepiece.SentencePieceProcessor tokenizer:
        :returns: List[str]
    """

    hypotheses = []
    ### YOUR CODE HERE
    logits = torch.argmax(logits, dim=2)
    for tokens_tensor, tokens_len in zip(logits, logits_len):
        tokens_tensor = tokens_tensor[:tokens_len]
        mask1 = tokens_tensor[1:] != tokens_tensor[:-1]
        mask2 = tokens_tensor != blank_id
        mask2[1:] &= mask1
        tokens_tensor = tokens_tensor[mask2]
        tokens_list = tokens_tensor.tolist()
        hypotheses.append(tokenizer.decode(tokens_list))

    return hypotheses

def fast_beam_search_decoding(logits, blank_id, tokenizer, beam_size=10, alpha=0, beta=0):
    """Decode text from logits using beam search. 
    Collapse all repeated tokens and then remove all blanks.
        :param torch.FloatTensor logits: (batch, time, vocabulary)
        :param int blank_id:
        :param sentencepiece.SentencePieceProcessor tokenizer:
        :param int beam_size:
        :returns: List[List[Tuple[str, float]]]
    """
    
    list1 = []
    for indx in range(128):
        token = tokenizer.id_to_piece(indx)
        token = token.replace('▁', ' ')
        list1.append(token)
    list1.append('blank')
    print(list1, '1!')
    decoder = CTCBeamDecoder(list1, model_path='/home/jupyter/work/resources/lm_50x50.binary', alpha=alpha, beta=beta
                             beam_width=beam_size, blank_id=blank_id, log_probs_input=True, is_token_based=True)
    beam_results, beam_scores, timesteps, out_lens = decoder.decode(logits)
    hypotheses = []
    for batch, lens, scores in zip(beam_results, out_lens, beam_scores):
        list1 = []
        for tokens, tokens_len, score in zip(batch, lens, scores):
            print(tokens[:tokens_len].tolist())
            list1.append((tokenizer.decode(tokens[:tokens_len].tolist()), score))
        hypotheses.append(list1)
    return hypotheses

def beam_search_decoding(logits, logits_len, blank_id, tokenizer, beam_size=10):
    """Decode text from logits using beam search. 
    Collapse all repeated tokens and then remove all blanks.
        :param torch.FloatTensor logits: (batch, time, vocabulary)
        :param torch.LongTensor logits_len: (batch)
        :param int blank_id:
        :param sentencepiece.SentencePieceProcessor tokenizer:
        :param int beam_size:
        :returns: List[List[Tuple[str, float]]]
    """
    
    hypotheses = []
    
    for time_tokens_tensor, tokens_len in zip(logits.cpu(), logits_len.cpu()):
        hypos = {(blank_id,)}
        prob_blank = defaultdict(lambda: float(-1e9))
        prob_blank[(blank_id,)] = 0
        prob_non_blank = defaultdict(lambda: float(-1e9))
        prob_non_blank[(blank_id,)] = -1e9
        
        new_hypos = set()
        new_prob_blank = defaultdict(lambda: float(-1e9))
        new_prob_non_blank = defaultdict(lambda: float(-1e9))
        
        for tokens_tensor in time_tokens_tensor[:tokens_len]:
            for string in hypos:
                for token in range(len(tokens_tensor)):
                    p_token = tokens_tensor[token].item()
                    if token == blank_id:
                        new_hypos.add(string)
                                                
                        new_prob_blank[string] = np.logaddexp(new_prob_blank[string], p_token + np.logaddexp(
                            prob_blank[string], prob_non_blank[string]
                        ))
                        
                    else:
                        if string[-1] == token:
                            new_hypos.add(string)
                            new_prob_non_blank[string] = np.logaddexp(new_prob_non_blank[string],
                                p_token + prob_non_blank[string]
                            )
                            
                            new_hypos.add((*string, token))
                            new_prob_non_blank[(*string, token)] = np.logaddexp(new_prob_non_blank[(*string, token)],
                                prob_blank[string] + p_token
                            )
                        else:
                            new_hypos.add((*string, token))
                            new_prob_non_blank[(*string, token)] = np.logaddexp(new_prob_non_blank[(*string, token)],
                                p_token + np.logaddexp(prob_blank[string], prob_non_blank[string])
                            )
            list1 = []
            for key in new_hypos:
                list1.append((key, new_prob_blank[key], 0))
                list1.append((key, new_prob_non_blank[key], 1))
            list1.sort(key=lambda x: x[1], reverse=True)
            
            hypos.clear()
            prob_blank.clear()
            prob_non_blank.clear()
            new_hypos.clear()
            new_prob_blank.clear()
            new_prob_non_blank.clear()
            
            for i in range(min(beam_size, len(list1))):
                hypos.add(list1[i][0])
                if list1[i][2] == 0:
                    prob_blank[list1[i][0]] = list1[i][1]
                else:
                    prob_non_blank[list1[i][0]] = list1[i][1]
        list1 = []
        for key in hypos:
            list1.append((tokenizer.decode(list(key[1:])), np.logaddexp(prob_blank[key], prob_non_blank[key])))
        hypotheses.append(list1)
    
    return hypotheses

@torch.no_grad()
def decode(model, signals, lengths, tokenizer, is_eval=True):
    """Perform model inference given raw audio
        :param torch.FloatTensor signals: (batch, time)
        :param torch.LongTensor lengths: (batch)
        :param sentencepiece.SentencePieceProcessor tokenizer:
        :param bool is_eval: whether to use eval mode for inference.
            Restore model state after the computations
        :return List[str]: list of predicted hypotheses
    """
    # Save initial model state
    model_state = model.training
    # Set model state w.r.t. `is_eval``
    ### YOUR CODE HERE
    ...

    # Perform forward pass and ctc decoding
    ### YOUR CODE HERE
    ...

    # Restore model state
    model.train(model_state)

    return hypotheses


def word_error_rate(hypotheses, references):
    """Compute macro averaged word-level Levenshtein distance. Use editdistance library
        :param List[str] hypotheses:
        :param List[str] references:
        :return Tuple[float, int, int]: (wer, words, scores):
            float wer:
            int words: total number of words in references. I.e. wer denominator
            int scores: sum of distances between hypotheses and references. I.e. wer numerator
    """
    ### YOUR CODE HERE
    ...

    return wer, words, scores


class WERMetric(torchmetrics.Metric):
    is_differentiable = False
    full_state_update = False
    higher_is_better = False

    def __init__(self, blank_id, tokenizer):
        """Wrapper for WER aggregation
            :param int blank_id:
            :param sentencepiece.SentencePieceProcessor tokenizer:
        """
        super().__init__()

        self.blank_id = blank_id
        self.tokenizer = tokenizer

        self.add_state('words', default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)
        self.add_state('scores', default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)

    def update(self, logits, logits_len, references):
        """
            :param torch.FloatTensor logits: (batch, time, vocabulary)
            :param torch.LongTensor logits_len: (batch)
            :param List[str] references: list of target texts
        """
        # Compute hypotheses for given logits
        ### YOUR CODE HERE
        ...

        # Calculate WER statistics
        ### YOUR CODE HERE
        ...

        # Update statistics
        ### YOUR CODE HERE
        ...

    def compute(self):
        """Compute aggregated statistics
            :return Tuple[float, int, int]: (wer, words, scores):
                float wer:
                int words: total number of words in references. I.e. wer denominator
                int scores: sum of distances between hypotheses and references. I.e. wer numerator
        """

        ### YOUR CODE HERE
        ...

        return wer, words, scores


class CTCLossMetric(torchmetrics.Metric):
    is_differentiable = False
    full_state_update = False
    higher_is_better = False

    def __init__(self):
        """Wrapper for CTCLoss aggregation
        """
        super().__init__()

        self.add_state('loss', default=torch.tensor(0.0), dist_reduce_fx='sum', persistent=False)
        self.add_state('num_objects', default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)

    def update(self, loss, num_objects):
        """
            :param float loss:
            :param int num_objects:
        """

        # Update statistics
        ### YOUR CODE HERE
        ...

    def compute(self):
        """Compute aggregated statistics
            :return Tuple[float, int, int]: (mean_loss, loss, num_objects):
                float mean_loss:
                float loss: aggregated loss. I.e. mean_loss numerator
                int num_objects: total number of objects. I.e. mean_loss denominator
        """
        
        # Update statistics
        ### YOUR CODE HERE
        ...

        return mean_loss, loss, num_objects
