import random

import torch


class SpecAugment(torch.nn.Module):
    def __init__(
            self, freq_masks=0, time_masks=0, freq_width=10, time_width=10
    ):
        """
        Zeroes out(cuts) random continuous horisontal or
        vertical segments of the spectrogram as described in
        SpecAugment (https://arxiv.org/abs/1904.08779).

        :param int freq_masks: how many frequency segments should be cut
        :param int time_masks: how many time segments should be cut
        :param int freq_width: maximum number of frequencies to be cut in one segment
        :param Union[int, float] time_width: maximum number of time steps to be cut in one segment.
            Can be a positive integer or a float value in the range [0, 1].
            If positive integer value, defines maximum number of time steps
            to be cut in one segment.
            If a float value, defines maximum percentage of timesteps that
            are cut adaptively.
        """
        super().__init__()

        self._rng = random.Random()

        self.freq_masks = freq_masks
        self.time_masks = time_masks

        self.freq_width = freq_width
        self.time_width = time_width

    @torch.no_grad()
    def forward(self, input_spec, length):
        """
            :param torch.Tensor input_spec: (batch, d, time)
            :param torch.Tensor length: (batch)
            :return torch.Tensor: (batch, d, time)
        """
        batch, d, _ = input_spec.shape

        for idx in range(batch):
            for i in range(self.freq_masks):
                # Sample range in frequency domain and zero in out
                ### YOUR CODE HERE
                rng = self._rng.randint(0, self.freq_width)
                pos = self._rng.randint(0, d)
                input_spec[idx, pos:(pos + rng)] = 0

            for i in range(self.time_masks):
                # Determine maximum cutout width
                ### YOUR CODE HERE
                if isinstance(self.time_width, float):
                    time_width = int(length[idx] * self.time_width)
                else:
                    time_width = self.time_width

                # Sample rectangle in time domain and zero in out
                ### YOUR CODE HERE
                rng = self._rng.randint(0, time_width)
                pos = self._rng.randint(0, length[idx])
                input_spec[idx, :, pos:(pos + rng)] = 0

        return input_spec


class SpecCutout(torch.nn.Module):
    def __init__(self, rect_masks=0, rect_time=5, rect_freq=20):
        """Zeroes out(cuts) random rectangles in the spectrogram
            as described in (https://arxiv.org/abs/1708.04552).
            :params int rect_masks: how many rectangular masks should be cut
            :params int rect_freq: maximum size of cut rectangles along the frequency dimension
            :params int rect_time: maximum size of cut rectangles along the time dimension
        """
        super().__init__()

        self._rng = random.Random()

        self.rect_masks = rect_masks
        self.rect_time = rect_time
        self.rect_freq = rect_freq

    @torch.no_grad()
    def forward(self, input_spec):
        """
            :param torch.Tensor input_spec: (batch, d, time)
            :return torch.Tensor: (batch, d, time)
        """
        batch, d, time = input_spec.shape  # we'r using time, not length???? what about padding????

        for idx in range(batch):
            for i in range(self.rect_masks):
                # Sample rectangle in frequency-time domain and zero in out
                ### YOUR CODE HERE
                freq_rng = self._rng.randint(0, self.rect_freq)
                time_rng = self._rng.randint(0, self.rect_time)
                freq_pos = self._rng.randint(0, d - freq_rng)
                time_pos = self._rng.randint(0, time - time_rng)

                input_spec[idx, freq_pos:(freq_pos + freq_rng), time_pos:(time_pos + time_rng)] = 0

        return input_spec


class SpectrogramAugmentation(torch.nn.Module):
    def __init__(
            self,
            freq_masks=0, time_masks=0, freq_width=10, time_width=10,
            rect_masks=0, rect_time=5, rect_freq=20
    ):
        """
        Performs time and freq cuts in one of two ways.
        SpecAugment zeroes out vertical and horizontal sections as described in
        SpecAugment (https://arxiv.org/abs/1904.08779). Arguments for use with
        SpecAugment are `freq_masks`, `time_masks`, `freq_width`, and `time_width`.
        SpecCutout zeroes out rectangulars as described in Cutout
        (https://arxiv.org/abs/1708.04552). Arguments for use with Cutout are
        `rect_masks`, `rect_freq`, and `rect_time`.
            :param int freq_masks: how many frequency segments should be cut
            :param int time_masks: how many time segments should be cut
            :param int freq_width: maximum number of frequencies to be cut in one segment
            :param Union[int, float] time_width: maximum number of time steps to be cut in one segment
            :param int rect_masks: how many rectangular masks should be cut
            :param int rect_freq: maximum size of cut rectangles along the frequency dimension
            :param int rect_time: maximum size of cut rectangles along the time dimension
        """
        super().__init__()

        self.spec_cutout = SpecCutout(rect_masks=rect_masks, rect_time=rect_time, rect_freq=rect_freq)
        self.spec_augment = SpecAugment(
            freq_masks=freq_masks, time_masks=time_masks,
            freq_width=freq_width, time_width=time_width
        )

    def forward(self, features, feature_lengths):
        """
            :param torch.Tensor features: (batch, d, time)
            :param torch.Tensor feature_lengths: (batch)
            :return torch.Tensor: (batch, d, time)
        """
        augmented_spec = self.spec_cutout(input_spec=features)
        augmented_spec = self.spec_augment(input_spec=augmented_spec, length=feature_lengths)

        return augmented_spec