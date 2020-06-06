import pdb

import numpy as np

import torch
from metrics import rms as np_rms
from metrics_torch import rms
from torch import Tensor, mean, sum


class CalcErrors:
    def __init__(self, window_predict):
        # initialize local copy of parameters
        self.window_predict = window_predict

        self._weights = self.generate_weights()

        # prepare loss
        self._loss = 0
        self._rms_ = 0
        self._rms_per_frame = Tensor(np.zeros(self.window_predict))
        self._rms_per_sequence = []
        self._count = 0
        self._rms_latitude = Tensor(np.zeros(72))

        self._rms_global_mean = []

    def __call__(self, outputs, targets):
        with torch.no_grad():
            # compute the rms for each image
            rms_tec_images_latitude = rms(outputs - targets, dim=(-3, -1))
            rms_tec_images = rms(rms_tec_images_latitude, dim=-1)

            self._rms_latitude += sum(rms_tec_images_latitude, dim=(0, 1))

            self._rms_gm = sum((outputs - targets) * self._weights[None, None, None, :, :], dim=(-3, -2, -1))

            for i in range(self._rms_gm.shape[0]):
                self._rms_global_mean.append(self._rms_gm[i].cpu().detach().numpy())

            # update global rms
            self._rms_ += sum(rms_tec_images)

            # update rms per seq frame
            self._rms_per_frame += sum(rms_tec_images, dim=0)

            for seq_id in range(rms_tec_images.shape[0]):
                self._rms_per_sequence.append(mean(rms_tec_images, dim=1))

    def update_count(self, value):
        self._count += value

    def update_loss(self, value):
        self._loss += value

    def get_loss(self):
        return self._loss / self._count

    def get_rms(self):
        return float(self._rms_.cpu().detach().numpy()) / self._count

    def calc_errors(self):
        return global_calc_errors(
            self._rms_global_mean, self._loss, self._rms_, self._rms_latitude,
            self._rms_per_frame, self._rms_per_sequence, self.window_predict,
            self._count)

    def generate_weights(self):
        weights = np.arange(-36, 36).reshape((72, 1))
        weights = np.repeat(weights, 72, 1)
        weights = np.abs(weights)
        weights[:36] -= 1
        weights = np.cos((weights.astype(float) / 36) * np.pi / 2)
        return Tensor(weights / weights.sum())


def global_calc_errors(rms_global_mean, loss, rms_, rms_latitude,
                       rms_per_frame, rms_per_sequence, window_predict, count):
    rms_global_mean = np.array(rms_global_mean)

    print("RMS GLOBAL MEAN", rms_global_mean.shape,
          np_rms(rms_global_mean, axis=0).mean())

    loss = loss / count
    rms_ = rms_ / count
    rms_latitude = rms_latitude / count
    for frame_id in range(window_predict):
        rms_per_frame[frame_id] /= count / window_predict

    dict_loss = {
        'loss': loss,
        'rms_': rms_,
        'rms_per_frame': rms_per_frame,
        'rms_per_sequence': rms_per_sequence,
        'rms_latitude': rms_latitude
    }

    return dict_loss
