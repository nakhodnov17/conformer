import torch


class ConformerFeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        """
            :param int d_model: input feature size
            :param int d_ff: hidden feature size
            :param float dropout:
        """
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        ### YOUR CODE HERE
        self.layer_norm = ...
        self.linear1 = ...
        self.activation = ...
        self.dropout_1 = ...
        self.linear2 = ...
        self.dropout_2 = ...

    def forward(self, x):
        """
            :param torch.Tensor x: (batch, time, d_model)
            :return torch.Tensor: (batch, time, d_model)
        """
        ### YOUR CODE HERE
        ...
        
        return x
