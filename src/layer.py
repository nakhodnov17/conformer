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
            :param float dropout: for feedforward, convolutional layers
            :param float dropout_att: for multihead attention
        """
        super().__init__()

        self.fc_factor = 0.5

        ### YOUR CODE HERE
        self.feed_forward1 = ConformerFeedForward(d_model, d_ff, dropout)

        self.norm_self_att = torch.nn.LayerNorm(d_model)
        self.self_attn = RelPositionMultiHeadAttention(n_heads, d_model, dropout_att)
        self.dropout = torch.nn.Dropout(p=dropout)

        self.conv = ConformerConvolution(d_model, conv_kernel_size, dropout)

        self.feed_forward2 = ConformerFeedForward(d_model, d_ff, dropout)

        self.norm_out = torch.nn.LayerNorm(d_model)

    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None):
        """
            :param torch.Tensor x: input signals (batch, time, d_model)
            :param torch.Tensor att_mask: attention masks (batch, time, time)
            :param torch.Tensor pos_emb: relative positional embeddings (1, 2*time-1, d_model)
            :param torch.Tensor pad_mask: padding mask
            
            :return torch.Tensor: (batch, time, d_model)
        """
        ### YOUR CODE HERE
        
        res = self.feed_forward1(x)
        x = x + self.fc_factor * res  # x + 1/2 * Feed Forward Module  <-- skip-connection
        
        #  Here goes Multi-Head Attention block
        res = self.norm_self_att(x)  # Layer normalization
        res = self.self_attn(res, res, res, att_mask, pos_emb)  # Multi-Head Attention with Relative Positional Embedding
        res = self.dropout(res)  # dropout
        x = x + res  # x + Multi-Head Attention  <-- skip-connection
        
        res = self.conv(x, pad_mask)
        x = x + res  # x + Convolution Module  <-- skip-connection
        
        res = self.feed_forward2(x)
        x = x + self.fc_factor * res  # x + 1/2 * Feed Forward Module  <-- skip-connection
        
        x = self.norm_out(x)
        
        return x
