import torch
from tqdm import tqdm
from apex import amp
import numpy as np
import os
import random

from model4.utils import get_input_with_label
from model5.utils import DistributedClassificationMeter
from torch.utils.tensorboard import SummaryWriter

import foundations
import settings


def train_one_epoch(epoch, model, train_dl, max_lr, optimizer, criterion, scheduler, records, batch_repeat, rank, writer, params):
    print(np.random.randint(0, 2e9), random.randint(0, 2e9), torch.rand((1, 1)))

    model.train()
    train_loss = 0
    train_loss_eval = 0
    if rank == 0:
        train_tk = tqdm(train_dl, total=int(len(train_dl)), desc='Train Epoch')
    else:
        train_tk = train_dl

    optimizer.zero_grad()
    total = 0
    correct_count = 0
    cnt = 0

    optimizer.zero_grad()
    for step, (data, skip) in enumerate(train_tk):
        if skip:
            continue
        cnt = cnt + 1
        inputs, labels = get_input_with_label(data, smooth=params["smooth"])
        model.train()
        outputs = model(inputs)

        total += labels.size(0)
        loss = criterion(outputs, labels) / batch_repeat

        if cnt % batch_repeat == 0:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            with amp.scale_loss(loss, optimizer, delay_unscale=True) as scaled_loss:
                scaled_loss.backward()

        with torch.no_grad():
            records.update(labels.data, outputs.data, loss.data)

        train_loss += loss.item() * batch_repeat
        if rank == 0:
            train_tk.set_postfix(loss=train_loss / (step + 1), acc=correct_count / total)

    if rank == 0:
        if scheduler is not None:
            writer.add_scalar("lr", np.mean(scheduler.get_lr()), epoch)
            scheduler.step()
        else:
            writer.add_scalar("lr", max_lr, epoch)
        info = records.log_metric()
        print(f'Epoch {epoch}: train loss={info["loss"]:.4f} | train acc={info["acc"]:.4f}')


def validate(model, val_dl, criterion, records, rank):
    seeds = [np.random.randint(0, 2e9), random.randint(0, 2e9)]
    np.random.seed(2018011328)
    random.seed(2018011328)

    # val
    model.eval()

    for data, skip in val_dl:
        if skip:
            continue
        inputs, labels = get_input_with_label(data)

        with torch.no_grad():
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
            records.update(labels.data, outputs.data, val_loss.data)

    if rank == 0:
        info = records.log_metric()
        print(f'\t val loss={info["loss"]:.4f} | val acc={info["acc"]:.4f}| pos loss ={info["pos_loss"]:.4f}')

    np.random.seed(seeds[0])
    random.seed(seeds[1])


def train(train_dl, val_dl, test_dl, val_dl_iter, model, optimizer, scheduler, criterion, params, train_sampler,
          val_sampler, rank):
    n_epochs = params['n_epochs']
    max_lr = params['max_lr']
    val_rate = params['val_rate']
    batch_repeat = params['batch_repeat']
    history_best = {}
    best_metric = 0

    if rank == 0:
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('tensorboard', exist_ok=True)
        if settings.USE_FOUNDATIONS:
            foundations.set_tensorboard_logdir('tensorboard')
        writer = SummaryWriter("tensorboard")
    else:
        writer = None

    for epoch in range(n_epochs):
        train_records = DistributedClassificationMeter(writer=writer, phase="train", epoch=epoch, workers=2, criterion=criterion)
        if train_sampler:
            train_sampler.set_epoch(epoch)
        train_one_epoch(epoch, model, train_dl, max_lr, optimizer, criterion, scheduler, train_records, batch_repeat, rank, writer, params)
        if epoch % val_rate == 0:
            val_records = DistributedClassificationMeter(writer=writer, phase="validation", epoch=epoch, workers=2, criterion=criterion)
            if val_sampler:
                val_sampler.set_epoch(epoch)
            validate(model, val_dl, criterion, val_records, rank)

            # 改的时候记得改大于小于啊！！！
            # aaaa记得改初始值啊
            info = val_records.log_metric(write_scalar=False)
            selection_metric = info["acc"]

            if selection_metric >= best_metric and rank == 0 and info["confidence"] < 7:
                best_metric = selection_metric
                print(
                    f'>>> Saving best model metric={selection_metric:.4f} compared to previous best {best_metric:.4f}')
                checkpoint = {'model': model.module.state_dict(), 'params': params}
                history_best = {"train_" + key: value for key, value in train_records.get_metric().items()}
                for key, value in val_records.get_metric().items():
                    history_best["val_" + key] = value

                torch.save(checkpoint, 'checkpoints/best_model.pth')
                if settings.USE_FOUNDATIONS:
                    foundations.save_artifact('checkpoints/best_model.pth', key='best_model_checkpoint')

    # Log metrics to GUI
    if rank == 0:
        for metric, value in history_best.items():
            if settings.USE_FOUNDATIONS:
                foundations.log_metric(metric, float(value))
            else:
                print(metric, float(value))
