import torch


class ConformerDecoder(torch.nn.Module):
    def __init__(self, feat_in, num_classes):
        """Simple CTC-Conformer decoder
            :param int feat_in: number of input features
            :param int num_classes: number of output classes without blank token
        """
        super().__init__()

        self._feat_in = feat_in
        # Add 1 for blank token
        self._num_classes = num_classes + 1

        self.decoder_layers = torch.nn.Sequential(
            # Create pointwise convolution to change number of channels
            ### YOUR CODE HERE
            torch.nn.Conv1d(self._feat_in, self._num_classes, 1)
        )

    def forward(self, encoder_output):
        """
            :param torch.Tensor encoder_output: (batch, d, time)
            :return torch.Tensor: (batch, time, num_classes)
                tokens log-probabilities in given position
        """

        # Apply pointwise convolution to change number of channels
        ### YOUR CODE HERE
        logits = self.decoder_layers(encoder_output) # (batch, num_classes, time)

        # Transpose logits and apply log_softmax
        ### YOUR CODE HERE
        logits = torch.nn.functional.log_softmax(torch.transpose(logits, 1, 2), dim=-1)

        return logits