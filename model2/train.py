import torch
from tqdm import tqdm
from apex import amp
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

from model2.utils import Meter, Unnormalize
from model2.data_loader import training_mean, training_std

import foundations
import settings


class Trainer(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self, train_dl, val_dl, test_dl, model: torch.nn.Module, optimizer, scheduler, criterion, params):
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.visual_iter = iter(val_dl)
        self.unnorm = Unnormalize(training_mean, training_std)

        self.model = model
        self.optimizer = optimizer
        self.num_epochs = params["num_epochs"]
        self.lr = params["max_lr"]
        self.scheduler = scheduler
        self.criterion = criterion

        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('tensorboard', exist_ok=True)
        if settings.USE_FOUNDATIONS:
            foundations.set_tensorboard_logdir('tensorboard')
        self.writer = SummaryWriter("tensorboard")
        self.meter_train = Meter(self.writer, 'train',0)
        self.meter_val = Meter(self.writer, 'val',0)
        self.current_epoch = 0
        self.best_metric = 1e9
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.phase = 'train'
        self.train()

    def eval(self):
        self.phase = 'test'
        self.model.eval()
        self.meter_val = Meter(self.writer, self.phase, self.current_epoch)

    def train(self):
        self.phase = 'train'
        self.model.train()
        self.meter_train = Meter(self.writer, self.phase, self.current_epoch)

    def forward(self, images, targets) -> (torch.Tensor, torch.Tensor):
        images = images.to(self.device)
        masks = targets.to(self.device)

        if self.phase == 'train':
            with torch.set_grad_enabled(True):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
        else:
            with torch.no_grad():
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
        return loss, outputs

    def step(self):
        self.train()
        train_tk = tqdm(self.train_dl, total=int(len(self.train_dl)), desc='Train Epoch')

        for inputs, labels, data in train_tk:
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                loss, outputs = self.forward(inputs, labels)
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                self.optimizer.step()

            self.meter_train.update(labels, outputs.detach().cpu(), loss.item())

        if self.scheduler is not None:
            self.writer.add_scalar("lr", np.mean(self.scheduler.get_lr()))
            self.scheduler.step()
        else:
            self.writer.add_scalar("lr", self.lr)

        dices, iou, loss = self.meter_train.log_metric()
        print(f'Epoch {self.current_epoch}: train loss={loss:.4f} | train iou={iou:.4f}')


    def validate(self):
        self.eval()

        for inputs, labels, data in self.val_dl:
            loss, output = self.forward(inputs, labels)
            output = output.detach().cpu()
            self.meter_val.update(labels, output, loss.item())

        dices, iou, loss = self.meter_val.log_metric()
        selection_metric = loss

        if selection_metric <= self.best_metric:
            self.best_metric = selection_metric
            print(f'>>> Saving best model metric={selection_metric:.4f}')
            checkpoint = {'model': self.model}
            torch.save(checkpoint, 'checkpoints/best_model.pth')
            if settings.USE_FOUNDATIONS:
                foundations.save_artifact('checkpoints/best_model.pth', key='best_model_checkpoint')

                foundations.log_metric("train_loss", float(np.mean(self.meter_train.losses)))
                foundations.log_metric("val_loss", float(loss))
                foundations.log_metric("val_dice", float(dices[0]))
                foundations.log_metric("val_iou", float(iou))

        try:
            inputs, labels, data = next(self.visual_iter)
        except:
            self.visual_iter = iter(self.val_dl)
            inputs, labels, data = next(self.visual_iter)

        _, output = self.forward(inputs, labels)
        output = torch.sigmoid(output.detach().cpu())
        self.writer.add_images(f'validate/{self.current_epoch}_inputs.png', self.unnorm(inputs), self.current_epoch)
        self.writer.add_images(f'validate/{self.current_epoch}_mask.png', labels, self.current_epoch)
        self.writer.add_images(f'validate/{self.current_epoch}_predict.png',  output, self.current_epoch)
        print(f'Epoch {self.current_epoch}: val loss={loss:.4f} | val iou={iou:.4f}')

    def start(self):
        self.current_epoch = 0
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            self.step()
            self.validate()
