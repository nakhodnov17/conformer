import math
import torch


class RelPositionMultiHeadAttention(torch.nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate):
        super().__init__()

        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.head = n_head

        # Linear transformations for queries, keys and values
        ### YOUR CODE HERE
        self.linear_q = torch.nn.Linear(n_feat, n_feat)
        self.linear_k = torch.nn.Linear(n_feat, n_feat)
        self.linear_v = torch.nn.Linear(n_feat, n_feat)
        self.linear_out = torch.nn.Linear(n_feat, n_feat)
        # Dropout layer for attention probabilities
        ### YOUR CODE HERE
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # Linear transformation for positional encoding
        ### YOUR CODE HERE
        self.linear_pos = torch.nn.Linear(n_feat, n_feat, bias=False)

        # These two learnable biases are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        ### YOUR CODE HERE
        self.pos_bias_u = torch.nn.Parameter(torch.randn(n_head, self.d_k))
        self.pos_bias_v = torch.nn.Parameter(torch.randn(n_head, self.d_k))

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
        
        q = self.linear_q(query)
        q = torch.transpose(q.reshape(q.shape[0], q.shape[1], self.head, -1), 1, 2)  # reshaping to make tensor of (batch, head, time, d_k)
        
        k = self.linear_k(key)
        k = torch.transpose(k.reshape(k.shape[0], k.shape[1], self.head, -1), 1, 2)  # -//-
        
        v = self.linear_v(value)
        v = torch.transpose(v.reshape(v.shape[0], v.shape[1], self.head, -1), 1, 2)  # -//-

        return q, k, v

    @staticmethod
    def rel_shift(x):
        """Compute relative positional encoding.
            :param torch.Tensor x: (batch, head, time1, time2)
            :return torch.Tensor: (batch, head, time1, time2)
        """
        # Need to add a column of zeros on the left side of last dimension to perform the relative shifting
        ### YOUR CODE HERE
        x = torch.nn.functional.pad(x, (1, 0), "constant", 0)  # (batch, head, time1, time2 + 1)

        # Reshape matrix
        ### YOUR CODE HERE
        x = x.reshape(x.shape[0], x.shape[1], x.shape[3], x.shape[2])  # (batch, head, time2 + 1, time1)

        # Need to drop the first row and reshape matrix back
        ### YOUR CODE HERE
        x = (x[:, :, 1:]).reshape(x.shape[0], x.shape[1], x.shape[3], -1)  # (batch, head, time1, time2)
        
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
            scores = scores.masked_fill(mask.unsqueeze(1), -1e9)

            ### YOUR CODE HERE
            attn = torch.nn.functional.softmax(scores, dim=-1).masked_fill(mask.unsqueeze(1), 0) # (batch, head, time1, time2)
        else:
            # Calculate attention probabilities
            ### YOUR CODE HERE
            attn = torch.nn.functional.softmax(scores, dim=-1) # (batch, head, time1, time2)

        # Apply attention dropout
        ### YOUR CODE HERE
        attn = self.dropout(attn)

        # Reweight value w.r.t. attention probabilities
        
        x = torch.matmul(attn, value)  # (batch, head, time1, d_k)
        x = torch.transpose(x, 1, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1) # (batch, time1, d_model)
        # Apply output linear transformation
        ### YOUR CODE HERE
        x = self.linear_out(x)

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
        p = self.linear_pos(pos_emb)
        p =  torch.transpose(p.reshape(p.shape[0], p.shape[1], self.head, -1), 1, 2) # (batch, head, 2*time1-1, d_k)

        # Apply linear transformation for queries, keys and values
        ### YOUR CODE HERE
        q, k, v = self.forward_qkv(query, key, value)
        q = torch.transpose(q, 1, 2)  # (batch, time1, head, d_k)

        # Sum q with biases
        ### YOUR CODE HERE
        q_with_bias_u = torch.transpose(self.pos_bias_u + q, 1, 2) # (batch, head, time1, d_k)
        q_with_bias_v = torch.transpose(self.pos_bias_v + q, 1, 2) # (batch, head, time1, d_k)
        
        
        # Compute attention score
        # First compute matrix a + matrix c
        #   as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        ### YOUR CODE HERE
        matrix_ac = torch.matmul(q_with_bias_u, torch.transpose(k, 2, 3)) # (batch, head, time1, time2)

        # Compute matrix b + matrix d
        ### YOUR CODE HERE
        matrix_bd = torch.matmul(q_with_bias_v, torch.transpose(p, 2, 3)) # (batch, head, time1, 2*time2-1)
        # Apply relative shift to b + d matrix
        ### YOUR CODE HERE
        matrix_bd = self.rel_shift(matrix_bd) # (batch, head, time1, 2*time2-1)
        # Drops extra elements in the matrix_bd to match the matrix_ac's size
        ### YOUR CODE HERE
        length = matrix_ac.shape[3]
        matrix_bd = matrix_bd[:, :, :, :length] # (batch, head, time1, time2)

        scores = (matrix_ac + matrix_bd) / self.s_d_k  # (batch, head, time1, time2)
        # Compute reweighed values using scores and mask 
        ### YOUR CODE HERE
        out = self.forward_attention(v, scores, mask)

        return out
