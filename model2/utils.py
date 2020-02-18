import torch
from torch import Tensor
import numpy as np

import matplotlib
from torch.utils.tensorboard import SummaryWriter

matplotlib.use('Agg')


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


class Meter:
    """A meter to keep track of iou and dice scores throughout an epoch"""

    def __init__(self, writer: SummaryWriter, phase: str, epoch: int):
        self.base_threshold = 0.5  # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []
        self.losses = []
        self.writer = writer
        self.epoch = epoch
        self.phase = phase

    @staticmethod
    def _predict(X, threshold):
        """X is sigmoid output of the model"""
        X_p = np.copy(X)
        preds = (X_p > threshold).astype('uint8')
        return preds

    @staticmethod
    def _metric(probability: Tensor, truth: Tensor, threshold=0.5):
        """Calculates dice of positive and negative images seperately"""
        '''probability and truth must be torch tensors'''
        batch_size = len(truth)
        with torch.no_grad():
            probability = probability.view(batch_size, -1)
            truth = truth.view(batch_size, -1)
            assert (probability.shape == truth.shape)

            p = (probability > threshold).float()
            t = (truth > 0.5).float()

            t_sum = t.sum(-1)
            p_sum = p.sum(-1)
            neg_index = torch.nonzero(t_sum == 0)
            pos_index = torch.nonzero(t_sum >= 1)

            dice_neg = (p_sum == 0).float()
            dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

            dice_neg = dice_neg[neg_index]
            dice_pos = dice_pos[pos_index]
            dice = torch.cat([dice_pos, dice_neg])

            num_neg = len(neg_index)
            num_pos = len(pos_index)

        return dice, dice_neg, dice_pos, num_neg, num_pos

    def _compute_iou_batch(self, outputs: Tensor, labels: Tensor, classes=None):
        """computes mean iou for a batch of ground truth masks and predicted masks"""
        ious = []
        preds = np.copy(outputs)  # copy is imp
        labels = np.array(labels)  # tensor to np
        for pred, label in zip(preds, labels):
            ious.append(np.nanmean(self._compute_ious(pred, label, classes)))
        iou = np.nanmean(ious)
        return iou

    @staticmethod
    def _compute_ious(pred, label, classes, ignore_index=255, only_present=True):
        """computes iou for one ground truth mask and predicted mask"""
        pred[label == ignore_index] = 0
        ious = []
        for c in classes:
            label_c = label == c
            if only_present and np.sum(label_c) == 0:
                ious.append(np.nan)
                continue
            pred_c = pred == c
            intersection = np.logical_and(pred_c, label_c).sum()
            union = np.logical_or(pred_c, label_c).sum()
            if union != 0:
                ious.append(intersection / union)
        return ious if ious else [1]

    def update(self, targets: Tensor, outputs: Tensor, loss: float):
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos, _, _ = self._metric(probs, targets, self.base_threshold)

        self.losses.append(loss)
        self.base_dice_scores.extend(dice.tolist())
        self.dice_pos_scores.extend(dice_pos.tolist())
        self.dice_neg_scores.extend(dice_neg.tolist())
        preds = self._predict(probs, self.base_threshold)
        iou = self._compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def log_metric(self):
        dice = np.nanmean(self.base_dice_scores)
        dice_neg = np.nanmean(self.dice_neg_scores)
        dice_pos = np.nanmean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)
        loss = np.nanmean(self.losses)

        self.writer.add_scalar(self.phase + "/dice", dice, self.epoch)
        self.writer.add_scalar(self.phase + "/dice_pos", dice_neg, self.epoch)
        self.writer.add_scalar(self.phase + "/dice", dice_pos, self.epoch)
        self.writer.add_scalar(self.phase + "/iou", iou, self.epoch)
        self.writer.add_scalar(self.phase + "/loss", loss, self.epoch)

        return dices, iou, loss
