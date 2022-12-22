import torch

from src.feedforward import ConformerFeedForward
from src.convolution import ConformerConvolution
from src.attention import RelPositionMultiHeadAttention


class ConformerLayer(torch.nn.Module):
    def __init__(self, d_model, d_ff, n_heads, conv_kernel_size, dropout, dropout_att):
        """
            :param int d_model:
            :param int d_ff:
            :param int n_heads:
            :param int conv_kernel_size:
            :param float dropout:
            :param float dropout_att:
        """
        super().__init__()

        self.fc_factor = 0.5

        ### YOUR CODE HERE
        self.feed_forward1 = ...

        self.norm_self_att = ...
        self.self_attn = ...
        self.dropout = ...

        self.conv = ...

        self.feed_forward2 = ...

        self.norm_out = ...

    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None):
        """
            :param torch.Tensor x: input signals (batch, time, d_model)
            :param torch.Tensor att_mask: attention masks (batch, time, time)
            :param torch.Tensor pos_emb: relative positional embeddings (1, 2*time-1, d_model)
            :param torch.Tensor pad_mask: padding mask
            
            :return torch.Tensor: (batch, time, d_model)
        """
        ### YOUR CODE HERE
        ...
        return x
