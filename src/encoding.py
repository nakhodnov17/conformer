import math

import torch


class RelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding for TransformerXL's layers
    See : Appendix B in https://arxiv.org/abs/1901.02860
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, xscale=False, dropout_rate_emb=0.0):
        """Construct an RelPositionalEncoding object.
            :param int d_model: embedding dim
            :param float dropout_rate: dropout rate
            :param int max_len: maximum input length
            :param bool xscale: whether to scale the input by sqrt(d_model)
            :param float dropout_rate_emb: dropout rate for the positional embeddings
        """
        super().__init__()
        self.d_model = d_model
        self.xscale = xscale
        self.max_len = max_len

        # Create Dropout layer for embeddings 
        ### YOUR CODE HERE
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        # Create Dropout layer for positional embeddings 
        if dropout_rate_emb > 0:
            ### YOUR CODE HERE
            self.dropout_emb = torch.nn.Dropout(p=dropout_rate_emb)
        else:
            self.dropout_emb = None
        

    def create_pe(self, positions):
        """Compute positional encoding for given indices
            :attr torch.Tensor pe: (1, pos_length, d_model)
            :param torch.Tensor positions: (pos_length)
        """
        pos_length = positions.size(0)

        # Compute positional encoding
        # as described in https://arxiv.org/abs/1706.03762 Section 3.5
        pe = torch.zeros(pos_length, self.d_model, device=positions.device, dtype=positions.dtype)
        ### YOUR CODE HERE
        positinos = positions.view(-1, 1)
        pe[:] = torch.arange(self.d_model, device=positions.device, dtype=positions.dtype)
        pe[:, ::2] = torch.sin(positions / torch.exp((pe[:, ::2] / self.d_model) * math.log(10000)))
        pe[:, 1::2] = torch.cos(positions / torch.exp((pe[:, 1::2] / self.d_model) * math.log(10000)))
        pe.unsqueeze_(0)
        

        # Save precomputed positional embeddings
        if hasattr(self, 'pe'):
            self.pe = pe
        else:
            self.register_buffer('pe', pe, persistent=False)

    def extend_pe(self, length, device):
        """Reset and extend the positional encodings if needed."""
        needed_size = 2 * length - 1
        self.center_pos = length  # saving central position for forward
        if hasattr(self, 'pe') and self.pe.size(1) >= needed_size:
            return
        # Positions would be from negative numbers to positive
        # Positive positions would be used for left positions and negative for right positions
        positions = torch.arange(length - 1, -length, -1, dtype=torch.float32, device=device).unsqueeze(1)
        self.create_pe(positions=positions)

    def forward(self, x):
        """Compute positional encoding.
            :param torch.Tensor x: Input of size(batch, time, feature_size)

            :return Tuple[torch.Tensor, torch.Tensor]: (x, pos_emb):
                torch.Tensor x: (batch, time, feature_size)
                torch.Tensor pos_emb: (1, 2*time-1, feature_size)
        """

        # Rescale input
        if self.xscale:
            ### YOUR CODE HERE
            x = x * math.sqrt(self.d_model)
        # Apply embeddings dropout
        ### YOUR CODE HERE
        x = self.dropout(x)

        # Center_pos would be the index of position 0
        # Negative positions would be used for right and positive for left tokens
        # for input of length L, 2*L-1 positions are needed, positions from (L-1) to -(L-1)
        ### YOUR CODE HERE
        length = x.size(1)
        start_pos: int = self.center_pos - length
        end_pos: int = self.center_pos + length - 1
        pos_emb = self.pe[:, start_pos:end_pos]

        # Apply positional embeddings dropout
        ### YOUR CODE HERE
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        
        return x, pos_emb
