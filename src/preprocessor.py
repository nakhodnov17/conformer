import torch
import torchaudio


def make_seq_mask(features, lengths):
    """
    Create mask with True value for each non-padding position
        :param torch.Tensor features: (batch, d, time)
        :param torch.Tensor lengths: (batch)
        :return torch.Tensor: (batch, 1, time)
    """
    ### YOUR CODE HERE
    mask = torch.arange(features.shape[2], device=lengths.device, dtype=lengths.dtype)
    mask = mask < lengths.unsqueeze(1)
    mask = mask.unsqueeze(1)

    return mask


class AudioToMelSpectrogramPreprocessor(torch.nn.Module):
    def __init__(
        self,

        sample_rate=16000,
        window_size=0.025,
        window_stride=0.01,

        nfilt=80, n_fft=512,
        window_fn=torch.hann_window,

        dither=1e-5, preemph=0.97,
    ):
        super().__init__()

        self._sample_rate = sample_rate

        self._n_window_size = int(window_size * self._sample_rate)
        self._n_window_stride = int(window_stride * self._sample_rate)

        self._dither_value = dither
        self._preemphasis_value = preemph

        self._num_fft = n_fft
        self._num_filters = nfilt

        self._mel_spec_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=self._sample_rate,
            win_length=self._n_window_size,
            hop_length=self._n_window_stride,

            mel_scale="slaney", norm="slaney",
            window_fn=window_fn, wkwargs={"periodic": False},
            n_mels=self._num_filters, n_fft=self._num_fft, f_min=0.0, f_max=None,
        )

    def _apply_dithering(self, signals):
        """Apply dithering to input signal
            :param torch.Tensor signals: (batch, time)
            :return torch.Tensor: (batch, time)
        """
        if self.training and self._dither_value > 0.0:
            # Sample random noise with defined magnitude and add it to the signal
            ### YOUR CODE HERE
            noise = torch.rand(signals.shape) * self._dither_value
            signals += noise
        return signals

    def _apply_preemphasis(self, signals):
        """Apply preemphasis filter to input signal
            :param torch.Tensor signals: (batch, time)
            :return torch.Tensor: (batch, time)
        """
        if self._preemphasis_value is not None:
            # Transform signal as follows: s_{i + 1} -> s_{i + 1} - pv * s_i
            ### YOUR CODE HERE
            signals[:, 1:] -= signals[:, :-1] * self._preemphasis_value
        return signals

    def _apply_normalization(self, features, lengths, eps=1e-5):
        """Normalize spectrogram
            :param torch.Tensor features: (batch, d, time)
            :param torch.Tensor lengths: (batch)
        """
        mask = ~make_seq_mask(features, lengths)

        # Compute statistics for each object and each feature separately
        # Do not count masked elements that corresponds to padding
        ### YOUR CODE HERE
        features.masked_fill_(mask, 0) # (batch, d, time)
        means = torch.sum(features, dim=-1, keepdims=True) / lengths[:, None, None] # (batch, d, 1)
        vrs = torch.sum(torch.pow(features - means, 2).masked_fill(mask, 0), dim=-1, keepdims=True) / (lengths[:, None, None] - 1)
        vrs.clamp_(float(2.0 ** -24))
        stds = vrs.sqrt() # (batch, d, 1)

        # Normalize non-masked elements. Use eps to prevent by-zero-division
        ### YOUR CODE HERE
        features = (features - means) / (stds + eps)
        
        # Set masked elements to zero
        ### YOUR CODE HERE
        features.masked_fill_(mask, 0) # (batch, d, time)

        return features

    def forward(self, signals, lengths, eps=2.0 ** -24):
        """
            :param torch.Tensor signals: (batch, time)
            :param torch.Tensor lengths: (batch)
            :return Tuple[torch.Tensor, torch.Tensor]: (features, feature_lengths):
                torch.Tensor features: (batch, d, time)
                torch.Tensor feature_lengths: (batch)
        """
        # Apply dithering
        ### YOUR CODE HERE
        signals = self._apply_dithering(signals)
        # Apply preemphasis
        ### YOUR CODE HERE
        signals = self._apply_preemphasis(signals)
        # Compute mel spectrogram features
        ### YOUR CODE HERE
        features = self._mel_spec_extractor(signals)
        
        # Compute log mel spectrogram features. 
        # Use eps to prevent underflow in log
        ### YOUR CODE HERE
        features = torch.log(features + eps)

        # Compute features lengths
        ### YOUR CODE HERE
        feature_lengths = lengths // self._n_window_stride + 1
        
        # Apply features normalization
        ### YOUR CODE HERE
        features = self._apply_normalization(features, feature_lengths)

        return features, feature_lengths


def calc_length(length, conv_params):
    """
        :param torch.Tensor length: (1) or (batch) 
        :param dict conv_params: dictionary with parameters of convolution layer
        :return torch.LongTensor: (1) or (batch). Spatial size after convolution with `conv_params` parameters
    """
    stride = conv_params.get('stride', 1)
    padding = conv_params.get('padding', 0)
    kernel_size = conv_params.get('kernel_size')
    dillation = conv_params.get('dillation', 1)

    ### YOUR CODE HERE
    return ((length + 2 * padding - (kernel_size - 1) * dillation - 1) // stride) + 1


class ConvSubsampling(torch.nn.Module):
    def __init__(
        self, feat_in, feat_out, sampling_num, conv_channels,
        activation=torch.nn.ReLU()
    ):
        """
            :param int feat_in: 
            :param int feat_out:
            :param int sampling_num: number of consecutive convolution subsamples
            :param int conv_channels: number of intermediate channels
        """
        super().__init__()

        self._feat_in = feat_in
        self._feat_out = feat_out
        self._sampling_num = sampling_num
        self._conv_channels = conv_channels

        self.conv_params = {
            'kernel_size': 3,
            'stride': 2,
            'padding': 1
        }
        in_channels = 1

        conv_layers = []
        out_features = self._feat_in
        # Create and append subsampling convolution layers to the list
        # Compute number of output features after this layers. Use calc_length
        ### YOUR CODE HERE
        for i in range(self._sampling_num):
            conv_layers.append(torch.nn.Conv2d((1 if i == 0 else self._conv_channels),
                                             self._conv_channels,
                                             self.conv_params["kernel_size"],
                                             stride=self.conv_params["stride"],
                                             padding=self.conv_params["padding"]))
            conv_layers.append(activation)
            out_features = calc_length(out_features, self.conv_params)

        self.conv = torch.nn.Sequential(*conv_layers)

        # Create linear projection layer
        ### YOUR CODE HERE

        self.out = torch.nn.Linear(out_features * self._conv_channels, feat_out)

    def forward(self, x, lengths):
        """
            :param torch.Tensor x: (batch, time, feat_in)
            :param torch.Tensor lengths: (batch)
            :return Tuple[torch.Tensor]: (batch, time, feat_out), (batch)
        """
        # Compute output sequence length. Use calc_length
        ### YOUR CODE HERE
        for i in range(self._sampling_num):
            lengths = calc_length(lengths, self.conv_params)

        # Apply convolution layers
        ### YOUR CODE HERE
        
        x.unsqueeze_(1)
        x = self.conv.forward(x)
        x = torch.transpose(x, 1, 2).reshape(x.shape[0], x.shape[2], -1)  # Making 4D tensor 3D by combining channels and features
        
        # Apply feed forward layer
        ### YOUR CODE HERE
        x = self.out(x)

        return x, lengths
