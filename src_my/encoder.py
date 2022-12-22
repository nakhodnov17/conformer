import math

import torch

from src.layer import ConformerLayer
from src.preprocessor import ConvSubsampling
from src.encoding import RelPositionalEncoding


class ConformerEncoder(torch.nn.Module):
    def __init__(
        self,
        feat_in,
        d_model,
        n_layers,
        pos_emb_max_len=5000,

        sampling_num=2, sampling_conv_channels=512,

        ff_expansion_factor=4,
        n_heads=8, conv_kernel_size=31,
        dropout=0.1, dropout_pre_encoder=0.1, dropout_emb=0.0, dropout_att=0.1
    ):
        super().__init__()

        # Create subsampling layer
        ### YOUR CODE HERE
        self.pre_encode = ...

        # Create layer for relative positional encoding
        ### YOUR CODE HERE
        self.pos_enc = ...
        # Compute embeddings for given maximum sequence length
        self.pos_enc.extend_pe(length=pos_emb_max_len, device=torch.device('cpu'))

        layers = []
        # Create and append Conformer layers
        ### YOUR CODE HERE
        ...

        self.layers = torch.nn.ModuleList(layers)

    @staticmethod
    def _create_masks(max_length, lengths, device):
        """
            :param int max_length: size of time dimension of the batch
            :param torch.LongTensor lengths: (batch) length of sequences in batch
            :param torch.Device device:
            :return Tuple[torch.BoolTensor, torch.BoolTensor]: (pad_mask, att_mask):
                torch.BoolTensor pad_mask: (batch, max_length)
                torch.BoolTensor att_mask: (1, max_length, max_length)
        """
        # pad_mask is the masking to be used to ignore paddings
        ### YOUR CODE HERE
        ...

        # att_mask is the masking to be used in self-attention
        ### YOUR CODE HERE
        ...

        return pad_mask, att_mask


    def forward(self, features, feature_lengths):
        """
            :param torch.Tensor features: (batch, d, time)
            :param torch.Tensor feature_lengths: (batch)
            :return Tuple[torch.Tensor, torch.Tensor]: (features, feature_lengths)
                torch.Tensor features: (batch, d, time)
                torch.Tensor feature_lengths: (batch)
        """
        features = torch.transpose(features, 1, 2)
        
        # Apply pre encoding subsampling
        ### YOUR CODE HERE
        ...

        # Apply positional encoding
        ### YOUR CODE HERE
        ...

        # Create the self-attention and padding masks
        ### YOUR CODE HERE
        ...

        # Apply Conformer layers
        ### YOUR CODE HERE
        ...

        return features, feature_lengths
