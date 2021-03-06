import torch
from torch import Tensor
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import log_loss


class Unnormalize:
    """Converts an image tensor that was previously Normalize'd
    back to an image with pixels in the range [0, 1]."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        mean = torch.as_tensor(self.mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
        std = torch.as_tensor(self.std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
        return torch.clamp(tensor * std + mean, 0., 1.)


class Classification_Meter:
    """A meter to keep track of iou and dice scores throughout an epoch"""

    def __init__(self, writer, phase: str, epoch: int):
        self.pos_loss = []
        self.neg_loss = []
        self.loss = []
        self.acc = []
        self.pos_acc = []
        self.neg_acc = []
        self.confidence = []
        self.writer = writer
        self.epoch = epoch
        self.phase = phase

    def update(self, targets: Tensor, outputs: Tensor, loss: float):
        try:
            predicts = torch.sigmoid(outputs).numpy().clip(1e-6, 1 - 1e-6)
            targets = targets.numpy()
            acc = ((predicts > 0.5) == targets).sum().item() / len(predicts)
            pos_p, pos_l = predicts[targets == 0], targets[targets == 0]
            neg_p, neg_l = predicts[targets == 1], targets[targets == 1]
            pos_acc = ((pos_p > 0.5) == pos_l).sum().item() / len(pos_p)
            neg_acc = ((neg_p > 0.5) == neg_l).sum().item() / len(neg_p)
            neg_loss = log_loss(neg_l, neg_p, labels=[0, 1])
            pos_loss = log_loss(pos_l, pos_p, labels=[0, 1])
            confidence = np.min([predicts, 1 - predicts], 0).mean()

            self.loss.append(loss)
            self.acc.append(acc)
            self.neg_loss.append(neg_loss)
            self.pos_loss.append(pos_loss)
            self.confidence.append(confidence)
            self.pos_acc.append(pos_acc)
            self.neg_acc.append(neg_acc)
        except:
            pass

    def log_metric(self, write_scalar=True):
        loss = np.nanmean(self.loss)
        neg_loss = np.nanmean(self.neg_loss)
        pos_loss = np.nanmean(self.pos_loss)
        acc = np.nanmean(self.acc)
        pos_acc = np.nanmean(self.pos_acc)
        neg_acc = np.nanmean(self.neg_acc)
        confidence = 1 / np.nanmean(self.confidence)

        ret = {"acc": acc, "loss": loss, "neg_loss": neg_loss, "pos_loss": pos_loss, "confidence": confidence,
               "pos_acc": pos_acc, "neg_acc": neg_acc}

        if self.writer and write_scalar:
            for key, value in ret.items():
                self.writer.add_scalar(self.phase + "/" + key, value, self.epoch)

        return ret

    def get_metric(self):
        return self.log_metric(write_scalar=False)
