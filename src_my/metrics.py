import torch
import torchmetrics

import editdistance

import math
from collections import defaultdict

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
        tokens_list = token_tensor.tolist()
        hypotheses.append(tokenizer.decode(tokens_list))

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
    
    for time_tokens_tensor, tokens_len in zip(logits, logits_len):
        hypos = set('')
        prob_blank = defaultdict(lambda: [float(-1e9), blank_id])
        prob_blank[''] = [0, blank_id]
        prod_non_blank = defaultdict(lambda: [float(-1e9), blank_id])
        prob_non_blank[''] = [-1e9, blank_id]
        
        new_hypos = set()
        new_prob_blank = defaultdict(lambda: [float(-1e9), blank_id])
        new_prod_non_blank = defaultdict(lambda: [float(-1e9), blank_id])
        
        for i in tokens_len:
            for string in hypos:
                for token in range(time_tokens_tensor[i]):
                    if token == blank_id:
                        new_hypos.add(string)
                        
                        last_token = prob_blank[string][1]
                        
                        new_prob_blank[string] = [
                            time_tokens_tensor[i][token] + math.log(
                                math.exp(prob_blank[string][0]) + 
                                math.exp(prob_non_blank[string][0])
                            ), last_token
                        ]
                    else:
                        token_str = tokenizer.decode(token)
                        if prob_non_blank[string][1] == token:
                            new_hypos.add(string)
                            new_prob_non_blank[string] = [prob_non_blank[string][0] + time_tokens_tensor[i][token], token]
                            new_hypos.add(string + token_str)
                            new_prob_non_blank[string + token_str] = [prob_blank[string][0] + time_tokens_tensor[i][token], token]
                        else:
                            new_hypos.add(string + token_str)
                            new_prob_non_blank[string + token_str] = [
                                time_tokens_tensor[i][token] + math.log(
                                    math.exp(prob_blank[string][0]) + 
                                    math.exp(prob_non_blank[string][0])
                                ), token
                            ]
            list1 = []
            for key in new_hypos:
                list1.append((key, new_prob_blank[key][0], new_prob_blank[key][1], 0))
                list1.append((key, new_prob_non_blank[key][0], new_prob_non_blank[key][1], 1))
            list1.sort(key=lambda: x[1], reverse=True)
            
            hypos.clear()
            prob_blank.clear()
            prob_non_blank.clear()
            new_hypos.clear()
            new_prob_blank.clear()
            new_prob_non_blank.clear()
            
            for i in range(min(beam_size, len(list1))):
                hypos.add(list1[i][0])
                if list1[i][3] == 0:
                    prob_blank[list1[i][0]] = [list1[i][1], list1[i][2]]
                else:
                    prob_non_blank[list1[i][0]] = [list1[i][1], list1[i][2]]
        list1 = []
        for key in hypos:
            list1.append((key, math.log(math.exp(prob_blank[key][0]) + math.exp(prob_non_blank[key][0]))))
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
