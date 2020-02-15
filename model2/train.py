import torch
from tqdm import tqdm
from apex import amp
import numpy as np
import os

from model2.utils import visualize_metrics, display_predictions_on_image, get_input_with_label
from sklearn.metrics import roc_auc_score as extra_metric

import foundations
import settings


class Records:
    def __init__(self):
        self.train_losses, self.train_losses_wo_dropout, self.val_losses = [], [], []
        self.train_accs, self.train_accs_wo_dropout, self.val_accs = [], [], []
        self.train_custom_metrics, self.train_custom_metrics_wo_dropout, self.val_custom_metrics = [], [], []
        self.lrs = []

    def write_to_records(self, **kwargs):
        assert len(set(kwargs.keys()) - set(self.__dir__())) == 0, 'invalid arguments!'
        for k, v in kwargs.items():
            setattr(self, k, v)

    def return_attributes(self):
        attributes = [i for i in self.__dir__() if
                      not (i.startswith('__') and i.endswith('__') or i in ('write_to_records', 'return_attributes',
                                                                            'get_metrics'))]
        return attributes

    def get_useful_metrics(self):
        return ['train_losses', 'val_accs', 'val_custom_metrics', 'val_losses']


def train_one_epoch(epoch, model, train_dl, max_lr, optimizer, criterion, scheduler, records):
    model.train()
    train_loss = 0
    train_loss_eval = 0
    train_tk = tqdm(train_dl, total=int(len(train_dl)), desc='Train Epoch')

    optimizer.zero_grad()
    total = 0
    correct_count = 0
    correct_count_eval = 0

    for step, data in enumerate(train_tk):
        inputs, labels = get_input_with_label(data)

        model.train()
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            predicted = torch.sigmoid(outputs.data)

            total += labels.size(0)
            correct_count += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

        train_loss += loss.item()
        train_tk.set_postfix(loss=train_loss / (step + 1), acc=correct_count / total)

    if scheduler is not None:
        records.lrs += scheduler.get_lr()
        scheduler.step()
    else:
        records.lrs.append(max_lr)

    records.train_losses_wo_dropout.append(train_loss_eval / (step + 1))
    records.train_accs_wo_dropout.append(correct_count_eval / total)
    records.train_losses.append(train_loss / (step + 1))
    records.train_accs.append(correct_count / total)

    print(f'Epoch {epoch}: train loss={records.train_losses[-1]:.4f} | train acc={records.train_accs[-1]:.4f}')
    print(
        f'Epoch {epoch}: eval_ loss={records.train_losses_wo_dropout[-1]:.4f} | train acc={records.train_accs_wo_dropout[-1]:.4f}')


def validate(model, val_dl, criterion, records):
    # val
    model.eval()
    val_loss = 0
    correct_count = 0
    total = 0

    all_labels = []
    all_predictions = []

    for data in val_dl:
        inputs, labels = get_input_with_label(data)

        with torch.no_grad():
            outputs = model(inputs)
            predicted = torch.sigmoid(outputs.data)

            total += labels.size(0)
            # TODO: calc metrics
            correct_count += (predicted == labels).sum().item()
            val_loss += criterion(outputs, labels)

        all_labels.append(labels.data.cpu().numpy())
        all_predictions.append(predicted.data.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    extra_score = extra_metric(all_labels, all_predictions)

    records.val_losses.append(val_loss / len(val_dl))
    records.val_accs.append(correct_count / total)
    records.val_custom_metrics.append(extra_score)
    print(f'\t val loss={records.val_losses[-1]:.4f} | val acc={records.val_accs[-1]:.4f} | '
          f'val {extra_metric.__name__}={records.val_custom_metrics[-1]:.4f}')


def train(train_dl, val_dl, test_dl, val_dl_iter, model, optimizer, n_epochs, max_lr, scheduler, criterion, val_rate):
    records = Records()
    best_metric = 1e9

    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(n_epochs):
        train_one_epoch(epoch, model, train_dl, max_lr, optimizer, criterion, scheduler, records)
        if epoch % val_rate == 0:
            validate(model, val_dl, criterion, records)
            # validate(model, test_dl, criterion, records)

            selection_metric = getattr(records, "val_losses")[-1]

            if selection_metric <= best_metric:
                best_metric = selection_metric
                print(
                    f'>>> Saving best model metric={selection_metric:.4f} compared to previous best {best_metric:.4f}')
                checkpoint = {'model': model,
                              'optimizer': optimizer.state_dict()}

                torch.save(checkpoint, 'checkpoints/best_model.pth')
                if settings.USE_FOUNDATIONS:
                    foundations.save_artifact('checkpoints/best_model.pth', key='best_model_checkpoint')

            # Save eyeball plot to Atlas GUI
            if settings.USE_FOUNDATIONS:
                display_filename = f'{epoch}_display.png'
                try:
                    data = next(val_dl_iter)
                except:
                    val_dl_iter = iter(val_dl)
                    data = next(val_dl_iter)
                display_predictions_on_image(model, data, name=display_filename)
                foundations.save_artifact(display_filename, key=f'{epoch}_display')

            # Save metrics plot
            visualize_metrics(records, extra_metric=extra_metric, name='metrics.png')

            # Save metrics plot to Atlas GUI
            if settings.USE_FOUNDATIONS:
                foundations.save_artifact('metrics.png', key='metrics_plot')

    # Log metrics to GUI
    max_index = np.argmin(getattr(records, 'val_losses'))

    useful_metrics = records.get_useful_metrics()
    for metric in useful_metrics:
        if settings.USE_FOUNDATIONS:
            foundations.log_metric(metric, float(getattr(records, metric)[max_index]))
        else:
            print(metric, float(getattr(records, metric)[max_index]))
