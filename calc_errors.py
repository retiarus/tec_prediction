import numpy as np

from metrics import rms


class CalcErrors:
    def __init__(self, window_predict, diff):
        # inicialize local copy of parameters
        self.window_predict = window_predict
        self.diff = diff

        # prepare weights
        weights = np.arange(-36, 36).reshape((72, 1))
        weights = np.repeat(weights, 72, 1)
        weights = np.abs(weights)
        weights[:36] -= 1
        weights = np.cos((weights.astype(float) / 36) * np.pi / 2)
        self._weights = weights / weights.sum()

        # prepare loss
        self._loss = 0
        self._rms_ = 0  # mean rms oversequence
        self._rms_periodic = 0  # mean rms over sequence
        self._rms_per_frame = [0 for i in range(self.window_predict)]
        self._rms_periodic_per_frame = [0 for i in range(self.window_predict)]
        self._rms_per_sequence = []
        self._rms_per_sequence_periodic = []
        self._count = 0
        self._rms_lattitude = np.zeros(72)

        self._rms_global_mean = []

    def __call__(self, np_outputs, np_periodic_blur, np_periodic, np_targets):
        if self.diff:
            # remove code associated with preprocess
            # outputs_complete = unprocess(outputs_np + np_periodic_blur)
            outputs_complete = np_outputs + np_periodic_blur
        else:
            # remove code associated with preprocess
            # outputs_complete = unprocess(outputs_np)
            outputs_complete = np_outputs
        # remove code associated with preprocess
        # periodic_complete = unprocess(periodic_np)
        # targets_complete = unprocess(targets_np)
        periodic_complete = np_periodic
        targets_complete = np_targets

        # compute the rms for each image
        rms_tec_images = rms(outputs_complete - targets_complete,
                             axis=(2, 3, 4))
        rms_tec_images_periodic = rms(periodic_complete - targets_complete,
                                      axis=(2, 3, 4))

        rms_tec_images_lattitude = rms(outputs_complete - targets_complete,
                                       axis=(2, 4))
        self._rms_lattitude += rms_tec_images_lattitude.sum(axis=(0, 1))

        # rms_gm = outputs_complete.mean(axis=(2,3,4))
        #                       - targets_complete.mean(axis=(2,3,4))
        self._rms_gm = (
            outputs_complete * self._weights[None, None, None, :, :]).sum(
                axis=(2, 3, 4)) - (targets_complete *
                                   self._weights[None, None, None, :, :]).sum(
                                       axis=(2, 3, 4))
        rms_gm = self._rms_gm.transpose(1, 0)
        for i in range(rms_gm.shape[0]):
            self._rms_global_mean.append(rms_gm[i])

        # update global rms
        self._rms_ += rms_tec_images.sum()
        self._rms_periodic += rms_tec_images_periodic.sum()

        # update rms per seq frame
        for frame_id in range(self.window_predict):
            self._rms_per_frame[frame_id] += rms_tec_images[frame_id].sum()
            self._rms_periodic_per_frame[frame_id] += rms_tec_images_periodic[
                frame_id].sum()

        for seq_id in range(rms_tec_images.shape[1]):
            self._rms_per_sequence.append(rms_tec_images[:, seq_id].mean())
            self._rms_per_sequence_periodic.append(
                rms_tec_images_periodic[:, seq_id].mean())

    def update_count(self, value):
        self._count += value

    def update_loss(self, value):
        self._loss += value

    def get_loss(self):
        return self._loss / self._count

    def get_rms(self):
        return self._rms_ / self._count

    def get_rms_periodic(self):
        return self._rms_periodic / self._count

    def calc_errors(self):
        return global_calc_errors(
            self._rms_global_mean, self._loss, self._rms_, self._rms_lattitude,
            self._rms_periodic, self._rms_per_frame, self._rms_periodic_per_frame,
            self._rms_per_sequence, self._rms_per_sequence_periodic,
            self.window_predict, self._count)


def global_calc_errors(rms_global_mean, loss, rms_, rms_lattitude,
                       rms_periodic, rms_per_frame, rms_periodic_per_frame,
                       rms_per_sequence, rms_per_sequence_periodic,
                       window_predict, count):
    rms_global_mean = np.array(rms_global_mean)

    print("RMS GLOBAL MEAN", rms_global_mean.shape,
          rms(rms_global_mean, axis=1).mean())

    loss = loss / count
    rms_ = rms_ / count
    rms_lattitude = rms_lattitude / count
    rms_periodic = rms_periodic / count
    for frame_id in range(window_predict):
        rms_per_frame[frame_id] /= count / window_predict
        rms_periodic_per_frame[frame_id] /= count / window_predict

    dict_loss = {
        'loss': loss,
        'rms_': rms_,
        'rms_periodic': rms_periodic,
        'rms_per_frame': rms_per_frame,
        'rms_periodic_per_frame': rms_periodic_per_frame,
        'rms_per_sequence': rms_per_sequence,
        'rms_per_sequence_periodic': rms_per_sequence_periodic,
        'rms_lattitude': rms_lattitude
    }

    return dict_loss
