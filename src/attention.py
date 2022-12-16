import math
import torch


class RelPositionMultiHeadAttention(torch.nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate):
        super().__init__()

        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.h = n_head

        # Linear transformations for queries, keys and values
        ### YOUR CODE HERE
        self.linear_q = ...
        self.linear_k = ...
        self.linear_v = ...
        self.linear_out = ...
        # Dropout layer for attention probabilities
        ### YOUR CODE HERE
        self.dropout = ...

        # Linear transformation for positional encoding
        ### YOUR CODE HERE
        self.linear_pos = ...

        # These two learnable biases are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        ### YOUR CODE HERE
        self.pos_bias_u = ...
        self.pos_bias_v = ...

    def forward_qkv(self, query, key, value):
        """Transforms query, key and value.
            :param torch.Tensor query: (batch, time1, n_feat)
            :param torch.Tensor key: (batch, time2, n_feat)
            :param torch.Tensor value: (batch, time2, n_feat)
            
            :return Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (q, k, v):
                torch.Tensor q: (batch, head, time1, d_k)
                torch.Tensor k: (batch, head, time2, d_k)
                torch.Tensor v: (batch, head, time2, d_k)
        """
        # Apply linear transformation for queries, keys and values
        ### YOUR CODE HERE
        ...

        return q, k, v

    @staticmethod
    def rel_shift(x):
        """Compute relative positional encoding.
            :param torch.Tensor x: (batch, head, time1, time2)
            :return torch.Tensor: (batch, head, time1, time2)
        """
        # Need to add a column of zeros on the left side of last dimension to perform the relative shifting
        ### YOUR CODE HERE
        x = ...  # (batch, head, time1, time2 + 1)

        # Reshape matrix
        ### YOUR CODE HERE
        x = ...  # (batch, head, time2 + 1, time1)

        # Need to drop the first row and reshape matrix back
        ### YOUR CODE HERE
        x = ...  # (batch, head, time1, time2)
        
        return x

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.
            :param torch.Tensor value: (batch, head, time2, d_k)
            :param torch.Tensor scores: (batch, head, time1, time2)
            :param torch.Tensor mask: (batch, time1, time2)

            :return torch.Tensor: transformed `value` (batch, time1, n_feat) weighted by the attention scores
        """
        if mask is not None:
            # Mask scores so that the won't be used in attention probabilities
            ### YOUR CODE HERE
            ...

            # Calculate attention probabilities
            ### YOUR CODE HERE
            attn = ... # (batch, head, time1, time2)
        else:
            # Calculate attention probabilities
            ### YOUR CODE HERE
            attn = ... # (batch, head, time1, time2)

        # Apply attention dropout
        ### YOUR CODE HERE
        ...

        # Reweigh value w.r.t. attention probabilities
        x = ... # (batch, time1, d_model)
        # Apply output linear transformation
        ### YOUR CODE HERE
        ...

        return x

    def forward(self, query, key, value, mask, pos_emb):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
            :param torch.Tensor query: (batch, time1, n_feat)
            :param torch.Tensor key: (batch, time2, n_feat)
            :param torch.Tensor value: (batch, time2, n_feat)
            :param torch.Tensor mask: (batch, time1, time2)
            :param torch.Tensor pos_emb: (batch, 2*time2-1, n_feat)

            :return torch.Tensor: transformed `value` (batch, time1, n_feat) weighted by the query dot key attention
        """
        # Apply linear transformation for positional embeddings
        ### YOUR CODE HERE
        p = ...
        p = ...  # (batch, head, 2*time1-1, d_k)

        # Apply linear transformation for queries, keys and values
        ### YOUR CODE HERE
        q, k, v = ...
        q = ...  # (batch, time1, head, d_k)

        # Sum q with biases
        ### YOUR CODE HERE
        q_with_bias_u = ... # (batch, head, time1, d_k)
        q_with_bias_v = ... # (batch, head, time1, d_k)

        # Compute attention score
        # First compute matrix a + matrix c
        #   as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        ### YOUR CODE HERE
        matrix_ac = ... # (batch, head, time1, time2)

        # Compute matrix b + matrix d
        ### YOUR CODE HERE
        matrix_bd = ... # (batch, head, time1, 2*time2-1)
        # Apply relative shift to b + d matrix
        ### YOUR CODE HERE
        matrix_bd = ... # (batch, head, time1, 2*time2-1)
        # Drops extra elements in the matrix_bd to match the matrix_ac's size
        ### YOUR CODE HERE
        matrix_bd = ... # (batch, head, time1, time2)

        scores = (matrix_ac + matrix_bd) / self.s_d_k  # (batch, head, time1, time2)
        # Compute reweighed values using scores and mask 
        ### YOUR CODE HERE
        ...

        return out
