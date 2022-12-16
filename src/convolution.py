import torch


class ConformerConvolution(torch.nn.Module):
    def __init__(self, d_model, kernel_size, dropout):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size

        ### YOUR CODE HERE
        self.layer_norm = ...
        self.pointwise_conv1 = ...
        self.depthwise_conv = ...
        self.batch_norm = ...
        self.activation = ...
        self.pointwise_conv2 = ...
        self.dropout = ...

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
        ...

        if pad_mask is not None:
            # Fill elements correspond to padding with zeros
            ### YOUR CODE HERE
            ...

        # Apply depthwise convolution
        # Apply batchnorm
        # Apply activation
        # Apply the second pointwise convolution
        # Apply dropout
        ### YOUR CODE HERE
        ...

        return x
