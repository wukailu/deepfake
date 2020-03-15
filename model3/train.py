import torch
from tqdm import tqdm
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

from model3.utils import Classification_Meter, Unnormalize

import foundations
import settings


class Trainer(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self, train_dl, val_dl, test_dl, model: torch.nn.Module, optimizer, scheduler, criterion, params):
        self.train_dl = train_dl
        # val_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.visual_iter = iter(val_dl)
        self.unnorm = Unnormalize(model.input_mean, model.input_std)

        self.model = torch.nn.DataParallel(model).cuda()
        self.optimizer = optimizer
        self.num_epochs = params["num_epochs"]
        self.lr = params["max_lr"]
        self.clip_gradient = params["clip_gradient"]
        self.scheduler = scheduler
        self.criterion = criterion
        self.batch_repeat = params["batch_repeat"]

        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('tensorboard', exist_ok=True)
        if settings.USE_FOUNDATIONS:
            foundations.set_tensorboard_logdir('tensorboard')
        self.writer = SummaryWriter("tensorboard")
        self.meter_train = Classification_Meter(self.writer, 'train', 0)
        self.meter_val = Classification_Meter(self.writer, 'val', 0)
        self.current_epoch = 0
        self.best_metric = 1e9
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.phase = 'train'
        self.train()

    def eval(self):
        self.phase = 'test'
        self.model.eval()
        self.meter_val = Classification_Meter(self.writer, self.phase, self.current_epoch)

    def train(self):
        self.phase = 'train'
        self.model.train()
        self.meter_train = Classification_Meter(self.writer, self.phase, self.current_epoch)

    def forward(self, images, targets) -> (torch.Tensor, torch.Tensor):
        images = images.cuda()
        masks = targets.cuda()

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
        print(np.random.randint(0, 2e9))
        self.train()
        train_tk = tqdm(self.train_dl, total=int(len(self.train_dl)), desc='Train Epoch')
        cnt = 0
        self.optimizer.zero_grad()
        for inputs, labels, data in train_tk:
            with torch.set_grad_enabled(True):
                cnt = cnt + 1
                loss, outputs = self.forward(inputs, labels)
                loss = loss/self.batch_repeat
                if cnt % self.batch_repeat == 0:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                    if self.clip_gradient > 0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.clip_gradient)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                else:
                    with amp.scale_loss(loss, self.optimizer, delay_unscale=True) as scaled_loss:
                        scaled_loss.backward()

            self.optimizer.zero_grad()
            self.meter_train.update(labels, outputs.detach().cpu(), loss.item() * self.batch_repeat)

        if self.scheduler is not None:
            self.writer.add_scalar("lr", np.mean(self.scheduler.get_lr()), self.current_epoch)
            self.scheduler.step()
        else:
            self.writer.add_scalar("lr", self.lr, self.current_epoch)

        acc, loss = self.meter_train.log_metric()
        print(f'Epoch {self.current_epoch}: train loss={loss:.4f} | train acc={acc:.4f}')

    def validate(self):
        self.eval()

        for inputs, labels, data in self.val_dl:
            loss, output = self.forward(inputs, labels)
            self.meter_val.update(labels, output.detach().cpu(), loss.item())

        acc, loss = self.meter_val.log_metric()
        selection_metric = loss

        if selection_metric <= self.best_metric:
            self.best_metric = selection_metric
            print(f'>>> Saving best model metric={selection_metric:.4f}')
            checkpoint = {'model': self.model, 'optimizer': self.optimizer.state_dict()}
            torch.save(checkpoint, 'checkpoints/best_model.pth')
            if settings.USE_FOUNDATIONS:
                foundations.save_artifact('checkpoints/best_model.pth', key='best_model_checkpoint')
                foundations.log_metric("train_loss", np.mean(self.meter_train.loss))
                foundations.log_metric("val_loss", loss)
                foundations.log_metric("val_acc", acc)

        try:
            inputs, labels, data = next(self.visual_iter)
        except:
            self.visual_iter = iter(self.val_dl)
            inputs, labels, data = next(self.visual_iter)

        _, output = self.forward(inputs, labels)
        output = torch.sigmoid(output.detach().cpu())
        inputs = inputs.view((-1, ) + inputs.shape[-3:])
        self.writer.add_images(f'validate/{self.current_epoch}_inputs.png', self.unnorm(inputs), self.current_epoch)
        print(labels.numpy().reshape(-1))
        print(output.numpy().reshape(-1))
        # self.writer.add_scalar(f"validate/{self.current_epoch}_label", labels[0, 0], self.current_epoch)
        # self.writer.add_scalar(f'validate/{self.current_epoch}_predict', output[0, 0], self.current_epoch)
        print(f'Epoch {self.current_epoch}: val loss={loss:.4f} | val acc={acc:.4f}')

    def start(self):
        self.current_epoch = 0
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            self.step()
            self.validate()
