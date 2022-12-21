import torch


class ConformerConvolution(torch.nn.Module):
    def __init__(self, d_model, kernel_size, dropout):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size

        ### YOUR CODE HERE
        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.pointwise_conv1 = torch.nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise_conv = torch.nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)
        self.batch_norm = torch.nn.BatchNorm1d(d_model)
        self.glu_activation = torch.nn.GLU(dim=1)
        self.silu_activation = torch.nn.SiLU()
        self.pointwise_conv2 = torch.nn.Conv1d(d_model, d_model, 1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        """
            :param torch.Tensor x: (batch, time, d_model)
            :param torch.Tensor pad_mask: (batch, time)
            :return torch.Tensor: (batch, time, d_model)
        """
        # Apply layer norm
        # Apply the first pointwise convolution which expands number of channels
        # Apply GLU
        ### YOUR CODE HERE
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu_activation(x)

        if pad_mask is not None:
            # Fill elements correspond to padding with zeros
            ### YOUR CODE HERE
            x.masked_fill_(pad_mask.unsqueeze(1), 0)

        # Apply depthwise convolution
        # Apply batchnorm
        # Apply swish activation
        # Apply the second pointwise convolution
        # Apply dropout
        ### YOUR CODE HERE
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.silu_activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)

        return x