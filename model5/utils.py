import torch
from torch import Tensor
import numpy as np
from sklearn.metrics import log_loss


def all_sum(tensor):
    import torch.distributed as dist
    rt = tensor.clone()  # The function operates in-place.
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


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


class DistributedClassificationMeter:
    """A meter to keep track of iou and dice scores throughout an epoch"""

    def __init__(self, writer, phase: str, epoch: int, workers, criterion):
        self.pos_loss = []
        self.neg_loss = []
        self.loss = []
        self.acc = []
        self.pos_acc = []
        self.neg_acc = []
        self.confidence = []
        self.bias = []
        self.mean = []
        self.writer = writer
        self.epoch = epoch
        self.phase = phase
        self.workers = workers
        self.criterion = criterion

    def update(self, targets: Tensor, outputs: Tensor, loss: Tensor):
        targets = targets.cuda()
        outputs = outputs.cuda()
        loss = loss.cuda()
        # try:
        predicts = torch.sigmoid(outputs).clamp(1e-6, 1 - 1e-6)
        mean = predicts.mean()
        acc = ((predicts > 0.5) == targets).sum().float() / len(predicts)
        bias = 1 - torch.logical_xor((predicts > 0.5)[:len(predicts)//2], (predicts > 0.5)[len(predicts)//2:]).float().mean()
        pos_p, pos_l = predicts[targets == 0], targets[targets == 0]
        neg_p, neg_l = predicts[targets == 1], targets[targets == 1]
        pos_acc = ((pos_p > 0.5) == pos_l).sum().float() / len(pos_p)
        neg_acc = ((neg_p > 0.5) == neg_l).sum().float() / len(neg_p)
        neg_loss = self.criterion(neg_p, neg_l)
        pos_loss = self.criterion(pos_p, pos_l)
        confidence = (torch.min(predicts, 1 - predicts)).mean()

        self.loss.append(all_sum(loss).item()/self.workers)
        self.acc.append(all_sum(acc).item()/self.workers)
        self.neg_loss.append(all_sum(neg_loss).item()/self.workers)
        self.pos_loss.append(all_sum(pos_loss).item()/self.workers)
        self.confidence.append(all_sum(confidence).item()/self.workers)
        self.pos_acc.append(all_sum(pos_acc).item()/self.workers)
        self.neg_acc.append(all_sum(neg_acc).item()/self.workers)
        self.bias.append(all_sum(bias).item()/self.workers)
        self.mean.append(all_sum(mean).item() / self.workers)
        # except:
        #     pass

    def log_metric(self, write_scalar=True):
        loss = np.nanmean(self.loss)
        neg_loss = np.nanmean(self.neg_loss)
        pos_loss = np.nanmean(self.pos_loss)
        acc = np.nanmean(self.acc)
        pos_acc = np.nanmean(self.pos_acc)
        neg_acc = np.nanmean(self.neg_acc)
        confidence = 1 / np.nanmean(self.confidence)
        bias = np.nanmean(self.bias)
        mean = np.nanmean(self.mean)

        ret = {"acc": acc, "loss": loss, "neg_loss": neg_loss, "pos_loss": pos_loss, "confidence": confidence,
               "pos_acc": pos_acc, "neg_acc": neg_acc, "bias": bias, 'mean': mean}

        if self.writer and write_scalar:
            for key, value in ret.items():
                self.writer.add_scalar(self.phase + "/" + key, value, self.epoch)

        return ret

    def get_metric(self):
        return self.log_metric(write_scalar=False)
