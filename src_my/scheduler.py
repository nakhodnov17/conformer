import warnings

import torch


class NoamAnnealing(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self, optimizer, *, d_model, warmup_steps, min_lr=0.0, last_epoch=-1
    ):
        """
            :param torch.optim.Optimizer optimizer:
            :param int d_model:
            :param int warmup_steps:
            :param float min_lr: lower bound for learning rate after warmup
            :param int last_epoch:
        """
        assert warmup_steps
        
        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by an inner class.
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self._normalize = d_model ** (-0.5)

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning
            )

        step = max(1, self.last_epoch)
        new_lrs = [
            self._noam_annealing(initial_lr=initial_lr, step=step) 
            for initial_lr in self.base_lrs
        ]
        return new_lrs

    def _noam_annealing(self, initial_lr, step):
        """Compute noam annealing learning rate 
            as described in https://arxiv.org/abs/1706.03762 Section 5.3.
            After warmup_steps learning rate should be always greater than min_lr

            :param float initial_lr: additional multiplicative factor for learning rate
            :param int step: current optimization step
            :return float: 
        """

        ### YOUR CODE HERE
        out_lr = min(max(initial_lr * self._normalize * step**(-0.5), self.min_lr), initial_lr * self._normalize * step * self.warmup_steps**(-1.5))

        return out_lr
