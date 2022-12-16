import torch

from src.decoder import ConformerDecoder
from src.encoder import ConformerEncoder
from src.preprocessor import AudioToMelSpectrogramPreprocessor


class Conformer(torch.nn.Module):
    def __init__(
        self, d_model=176, n_layers=16, num_classes=128,

        sampling_num=2, sampling_conv_channels=176,

        ff_expansion_factor=4,
        n_heads=4, conv_kernel_size=31,
        dropout=0.1, dropout_att=0.1
    ):
        """
            :param int d_model:
            :param int n_layers:
            :param int num_classes:

            :param int sampling_num: number of subsamplings in pre encoder
            :param int sampling_conv_channels: number of channels in pre encoder

            :param int ff_expansion_factor: feed forward layer hidden size expansion factor

            :param int n_heads:

            :param int conv_kernel_size: convolution layer kernel size

            :param float dropout: feed forward and convolution layers dropout
            :param float dropout_att: attention dropout
        """
        super().__init__()

        self.d_model = d_model

        # Create audio to spectrogram preprocessor
        ### YOUR CODE HERE
        self.preprocessor = ...
        # Create Conformer encoder
        ### YOUR CODE HERE
        self.encoder = ...
        # Create Conformer decoder
        ### YOUR CODE HERE
        self.decoder = ...
        # Create spectrogram augmentation module
        ### YOUR CODE HERE
        self.spec_augmentation = ...

        self.loss = torch.nn.CTCLoss(blank=num_classes)

    def forward(self, signals=None, lengths=None):
        """
            :param torch.Tensor signals: (batch, time)
            :param torch.Tensor lengths: (batch)
            :return Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (log_probs, encoded_len, greedy_predictions):
                log_probs: (batch, time, vocabulary)
                encoded_len: (batch)
                greedy_predictions: (batch, time)
        """
        # Transform raw audio to spectrogram features
        features, feature_lengths = ...

        # Apply spectrogram augmentation
        if self.spec_augmentation is not None and self.training:
            features = ...

        # Apply Conformer encoder
        ### YOUR CODE HERE
        encoded, encoded_len = ...

        # Apply Conformer decoder
        ### YOUR CODE HERE
        log_probs = ...

        # Make greedy predictions for each position
        ### YOUR CODE HERE
        greedy_predictions = ...

        return (
            log_probs,
            encoded_len,
            greedy_predictions,
        )
