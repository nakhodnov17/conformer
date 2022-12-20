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
        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.linear_1 = torch.nn.Linear(d_model, d_ff)
        self.activation = torch.nn.SiLU()
        self.dropout_1 = torch.nn.Dropout(p=dropout)
        self.linear_2 = torch.nn.Linear(d_ff, d_model) 
        self.dropout_2 = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        """
            :param torch.Tensor x: (batch, time, d_model)
            :return torch.Tensor: (batch, time, d_model)
        """
        ### YOUR CODE HERE
        x = self.layer_norm(x)
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout_1(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)
        
        return x
