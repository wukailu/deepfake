import torch
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def visualize_metrics(records, extra_metric, name):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 6))
    axes[0].plot(list(range(len(records.train_losses))), records.train_losses, label='train')
    axes[0].plot(list(range(len(records.train_losses_wo_dropout))), records.train_losses_wo_dropout,
                 label='train w/o dropout')
    axes[0].plot(list(range(len(records.val_losses))), records.val_losses, label='val')
    axes[0].set_title('loss')
    axes[0].legend()

    axes[1].plot(list(range(len(records.train_accs))), records.train_accs, label='train')
    axes[1].plot(list(range(len(records.train_accs_wo_dropout))), records.train_accs_wo_dropout,
                 label='train w/o dropout')
    axes[1].plot(list(range(len(records.val_accs))), records.val_accs, label='val')
    axes[1].axhline(y=0.5, color='g', ls='--')
    axes[1].axhline(y=0.667, color='r', ls='--')
    axes[1].set_title('acc')
    axes[1].legend()

    axes[2].plot(list(range(len(records.train_custom_metrics))), records.train_custom_metrics, label='train')
    axes[2].plot(list(range(len(records.train_custom_metrics_wo_dropout))), records.train_custom_metrics_wo_dropout,
                 label='train w/o dropout')
    axes[2].plot(list(range(len(records.val_custom_metrics))), records.val_custom_metrics, label='val')
    axes[2].axhline(y=0.5, color='g', ls='--')
    axes[2].axhline(y=0.5, color='r', ls='--')
    axes[2].set_title(f'{extra_metric.__name__}')
    axes[2].legend()

    axes[3].plot(list(range(len(records.lrs))), records.lrs)
    _ = axes[3].set_title('lr')
    plt.tight_layout()
    plt.savefig(name, format='png')


def display_predictions_on_image(model, data, name):
    # val
    model.eval()

    inputs, labels = get_input_with_label(data)
    img_files = data['real_file'] + data['fake_file']

    with torch.no_grad():
        outputs = model(inputs)
        predicted = torch.sigmoid(outputs.data)
        # TODO: calc probability
        outputs_predicbilty = torch.sigmoid(outputs.data)
    numbers = min(labels.size(0), 100)
    nrows = int(numbers ** 0.5)
    ncols = int(np.ceil(numbers / nrows))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 40))
    step = 0
    for i in range(nrows):
        for j in range(ncols):
            face_crop = np.load(img_files[step])
            axes[i, j].set_title(
                f'{outputs_predicbilty[step][0]:.2f},{outputs_predicbilty[step][1]:.2f}|{predicted[step]}|{labels[step]}')
            axes[i, j].imshow(face_crop)
            step += 1
            if step == numbers:
                break
    plt.title('predicted probability real, fake | prediction | label (0: real 1: fake)')
    plt.tight_layout()
    plt.savefig(name, format='png')
    plt.close(fig)


def get_input_with_label(data: dict):
    # TODO: change this
    batch_size = data['real'].shape[0]
    inputs = torch.cat((data['real'], data['fake'])).cuda()
    labels = torch.cat((torch.zeros(batch_size), torch.ones(batch_size))).long().cuda()
    return inputs, labels


class Meter:
    """A meter to keep track of iou and dice scores throughout an epoch"""

    def __init__(self, phase, epoch):
        self.base_threshold = 0.5  # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    @staticmethod
    def _predict(X, threshold):
        """X is sigmoid output of the model"""
        X_p = np.copy(X)
        preds = (X_p > threshold).astype('uint8')
        return preds

    @staticmethod
    def _metric(probability, truth, threshold=0.5):
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

    def _compute_iou_batch(self, outputs, labels, classes=None):
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

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos, _, _ = self._metric(probs, targets, self.base_threshold)
        self.base_dice_scores.extend(dice.tolist())
        self.dice_pos_scores.extend(dice_pos.tolist())
        self.dice_neg_scores.extend(dice_neg.tolist())
        preds = self._predict(probs, self.base_threshold)
        iou = self._compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.nanmean(self.base_dice_scores)
        dice_neg = np.nanmean(self.dice_neg_scores)
        dice_pos = np.nanmean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)
        return dices, iou
