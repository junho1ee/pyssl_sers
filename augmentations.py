import random

import numpy as np
import torch
from torchvision import transforms


def instantiate_from_name(str, **kwargs):
    if str == "powerline_noise":
        return PowerlineNoise(**kwargs)
    elif str == "emg_noise":
        return EMGNoise(**kwargs)
    elif str == "baseline_shift":
        return BaselineShift(**kwargs)
    elif str == "baseline_wander":
        return BaselineWander(**kwargs)
    elif str == "random_resized_crop":
        return RandomResizedCrop(**kwargs)
    elif str == "freqout":
        return FreqOut(**kwargs)
    else:
        raise ValueError(f"inappropriate perturbation choices: {str}")


def get_transformation(perturbation_mode=None, p=None):
    transform_list = []
    transform_list.append(ToFloatTensor())

    if perturbation_mode is not None:
        if hasattr(p, "__len__") and len(p) == 1:
            p = list(p) * len(perturbation_mode)
        elif isinstance(p, float):
            p = [p] * len(perturbation_mode)

        for aug, prob in zip(perturbation_mode, p):
            transform_list.append(instantiate_from_name(aug, p=prob))

    transform = transforms.Compose(transform_list)
    return transform


class ToFloatTensor(object):
    """
    Converts numpy arrays to float Variables in Pytorch.
    """

    def __call__(self, x):
        x = torch.from_numpy(x)
        return x


class PowerlineNoise(object):
    def __init__(
        self,
        max_amplitude=0.1,  # 0.5,
        min_amplitude=0,
        p=1.0,
        time=500,
    ):
        self.max_amplitude = max_amplitude
        self.min_amplitude = min_amplitude
        self.time = time
        self.p = p

    def __call__(self, x):
        x = x.clone()
        if self.p > random.random():
            csz, fsz = x.shape
            amp = np.random.uniform(self.min_amplitude, self.max_amplitude, size=(1, 1))
            t = 50 if np.random.uniform(0, 1) > 0.5 else 60
            noise = self._apply_powerline_noise(fsz, t)
            x = x + noise * amp

        return x

    def _apply_powerline_noise(self, fsz, t):
        f = np.linspace(0, fsz - 1, fsz)
        phase = np.random.uniform(0, 2 * np.pi)
        noise = np.cos(2 * np.pi * t * (f / self.time) + phase)
        return noise


class EMGNoise(object):
    def __init__(
        self,
        max_amplitude=0.1,  # 0.5,
        min_amplitude=0,
        p=1.0,
        **kwargs,
    ):
        self.max_amplitude = max_amplitude
        self.min_amplitude = min_amplitude
        self.p = p

    def __call__(self, x):
        x = x.clone()
        if self.p > random.random():
            csz, fsz = x.shape
            amp = np.random.uniform(
                self.min_amplitude, self.max_amplitude, size=(csz, 1)
            )
            noise = np.random.normal(0, 1, [csz, fsz])
            x = x + noise * amp
        return x


class BaselineShift(object):
    def __init__(
        self,
        max_amplitude=0.1,  # 0.25,
        min_amplitude=0,
        shift_ratio=0.2,
        num_segment=1,
        time=500,
        p=1.0,
        n_channels=1,
    ):
        self.max_amplitude = max_amplitude
        self.min_amplitude = min_amplitude
        self.shift_ratio = shift_ratio
        self.num_segment = num_segment
        self.time = time
        self.p = p
        self.n_channels = n_channels

    def __call__(self, x):
        x = x.clone()
        if self.p > random.random():
            csz, fsz = x.shape
            shift_length = fsz * self.shift_ratio
            amp_channel = np.random.choice([1, -1], size=(csz, 1))
            amp_general = np.random.uniform(
                self.min_amplitude, self.max_amplitude, size=(1, 1)
            )
            amp = amp_channel - amp_general
            noise = np.zeros(shape=(csz, fsz))
            for i in range(self.num_segment):
                segment_len = np.random.normal(shift_length, shift_length * 0.2)
                f0 = int(np.random.uniform(0, fsz - segment_len))
                f = int(f0 + segment_len)
                c = np.array([i for i in range(self.n_channels)])
                noise[c, f0:f] = 1
            x = x + noise * amp
        return x


class BaselineWander(object):
    def __init__(
        self,
        max_amplitude=0.5,
        min_amplitude=0,
        p=1.0,
        max_time=0.2,
        min_time=0.01,
        k=3,
        time=500,
        n_channels=1,
    ):
        self.max_amplitude = max_amplitude
        self.min_amplitude = min_amplitude
        self.max_time = max_time
        self.min_time = min_time
        self.k = k
        self.time = time
        self.p = p
        self.n_channels = n_channels

    def __call__(self, x):
        x = x.clone()
        if self.p > random.random():
            csz, fsz = x.shape
            amp_channel = np.random.normal(1, 0.5, size=(csz, 1))
            c = np.array([i for i in range(self.n_channels)])
            amp_general = np.random.uniform(
                self.min_amplitude, self.max_amplitude, size=self.k
            )
            noise = np.zeros(shape=(1, fsz))
            for k in range(self.k):
                noise += self._apply_baseline_wander(fsz) * amp_general[k]
            noise = (noise * amp_channel).astype(np.float32)
            x[c, :] = x[c, :] + noise[c, :]
        return x

    def _apply_baseline_wander(self, fsz):
        t = np.random.uniform(self.min_time, self.max_time)
        f = np.linspace(0, fsz - 1, fsz)
        r = np.random.uniform(0, 2 * np.pi)
        noise = np.cos(2 * np.pi * t * (f / self.time) + r)
        return noise


class RandomCrop(object):
    """Crop randomly the image in a x."""

    def __init__(self, output_size, annotation=False):
        self.output_size = output_size
        self.annotation = annotation

    def __call__(self, x):
        x = x.clone()

        fsz = len(x)
        assert fsz >= self.output_size
        if fsz == self.output_size:
            start = 0
        else:
            start = random.randint(0, fsz - self.output_size - 1)
        x = x[start : start + self.output_size]
        return x


def interpolate(x, marker):
    fsz, csz = x.shape
    x = x.flatten(order="F")
    x[x == marker] = np.interp(
        np.where(x == marker)[0], np.where(x != marker)[0], x[x != marker]
    )
    x = x.reshape(fsz, csz, order="F")
    return x


class RandomResizedCrop(object):
    """Extract crop at random position and resize it to full size"""

    def __init__(self, crop_ratio_range=[0.5, 1.0], output_size=551, p=1.0):
        self.crop_ratio_range = crop_ratio_range
        self.output_size = output_size
        self.p = p

    def __call__(self, x):
        x = x.clone().T
        if self.p > random.random():
            fsz, csz = x.shape
            output = np.full((self.output_size, csz), np.inf)
            output_fsz, csz = output.shape
            crop_ratio = random.uniform(*self.crop_ratio_range)
            x = RandomCrop(int(crop_ratio * fsz))(x)  # apply random crop
            cropped_fsz = x.shape[0]
            if output_fsz >= cropped_fsz:
                indices = np.sort(
                    np.random.choice(
                        np.arange(output_fsz - 2) + 1,
                        size=cropped_fsz - 2,
                        replace=False,
                    )
                )
                indices = np.concatenate(
                    [np.array([0]), indices, np.array([output_fsz - 1])]
                )
                output[indices, :] = x
                output = interpolate(output, np.inf)
            else:
                indices = np.sort(
                    np.random.choice(
                        np.arange(cropped_fsz), size=output_fsz, replace=False
                    )
                )
                output = x[indices]
            return torch.from_numpy(output.T)
        else:
            return x.T


class FreqOut(object):
    """replace random crop by zeros"""

    def __init__(self, crop_ratio_range=[0.0, 0.5], p=1.0):
        self.crop_ratio_range = crop_ratio_range
        self.p = p

    def __call__(self, x):
        x = x.clone()
        if self.p > random.random():
            csz, fsz = x.shape
            crop_ratio = random.uniform(*self.crop_ratio_range)
            crop_fsz = int(crop_ratio * fsz)
            start_idx = random.randint(0, fsz - crop_fsz - 1)
            x[:, start_idx : start_idx + crop_fsz] = 0.0
        return x
