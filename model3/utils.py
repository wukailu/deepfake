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
        return torch.clamp(tensor*std + mean, 0., 1.)


class Classification_Meter:
    """A meter to keep track of iou and dice scores throughout an epoch"""

    def __init__(self, writer: SummaryWriter, phase: str, epoch: int):
        self.pos_loss = []
        self.neg_loss = []
        self.loss = []
        self.acc = []
        self.writer = writer
        self.epoch = epoch
        self.phase = phase

    def update(self, targets: Tensor, outputs: Tensor, loss: float):
        predicts = torch.sigmoid(outputs).numpy().clip(1e-6, 1 - 1e-6)
        targets = targets.numpy()
        acc = ((predicts > 0.5) == targets).sum().item()/len(predicts)
        neg_loss = log_loss(targets[targets == 1], predicts[targets == 1], labels=[0, 1])
        pos_loss = log_loss(targets[targets == 0], predicts[targets == 0], labels=[0, 1])

        self.loss.append(loss)
        self.acc.append(acc)
        self.neg_loss.append(neg_loss)
        self.pos_loss.append(pos_loss)

    def log_metric(self):
        loss = np.nanmean(self.loss)
        neg_loss = np.nanmean(self.neg_loss)
        pos_loss = np.nanmean(self.pos_loss)
        acc = np.nanmean(self.acc)

        self.writer.add_scalar(self.phase + "/loss", loss, self.epoch)
        self.writer.add_scalar(self.phase + "/neg_loss", neg_loss, self.epoch)
        self.writer.add_scalar(self.phase + "/pos_loss", pos_loss, self.epoch)
        self.writer.add_scalar(self.phase + "/acc", acc, self.epoch)

        return acc, loss
