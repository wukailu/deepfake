import torch
from tqdm import tqdm
from apex import amp
import random
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

from model5.utils import DistributedClassificationMeter, Unnormalize
import foundations
import settings


class Trainer(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self, train_dl, val_dl, test_dl, train_sampler, val_sampler, model: torch.nn.Module, optimizer, scheduler, criterion, params, rank):
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.rank = rank
        self.total_gpu = params["gpus"]
        self.visual_iter = iter(val_dl)
        self.unnorm = Unnormalize(model.input_mean, model.input_std)

        # self.model = torch.nn.DataParallel(convert_model(model)).cuda()
        # serious bugs due to DataParallel, may caused by BN and apex
        self.model = model
        self.optimizer = optimizer
        self.num_epochs = params["num_epochs"]
        self.lr = params["max_lr"]
        self.clip_gradient = params["clip_gradient"]
        self.scheduler = scheduler
        self.criterion = criterion
        self.batch_repeat = params["batch_repeat"]

        if rank == 0:
            os.makedirs('checkpoints', exist_ok=True)
            os.makedirs('tensorboard', exist_ok=True)
            if settings.USE_FOUNDATIONS:
                foundations.set_tensorboard_logdir('tensorboard')
            self.writer = SummaryWriter("tensorboard")
        else:
            self.writer = None
        self.meter_train = None
        self.meter_val = None
        self.current_epoch = 0
        self.best_metric = 1e9
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.phase = 'train'
        self.seeds = [np.random.randint(0, 2e9), random.randint(0, 2e9)]
        self.train()
        self.history_best = {}

    def eval(self):
        self.seeds = [np.random.randint(0, 2e9), random.randint(0, 2e9)]
        np.random.seed(2018011328)
        random.seed(2018011328)
        self.phase = 'test'
        self.model.eval()
        self.meter_val = DistributedClassificationMeter(self.writer, self.phase, self.current_epoch, self.total_gpu, self.criterion)

    def train(self):
        np.random.seed(self.seeds[0])
        random.seed(self.seeds[1])
        self.phase = 'train'
        self.model.train()
        self.meter_train = DistributedClassificationMeter(self.writer, self.phase, self.current_epoch, self.total_gpu, self.criterion)

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
        self.train()
        print(np.random.randint(0, 2e9))
        if self.rank == 0:
            train_tk = tqdm(self.train_dl, total=int(len(self.train_dl)), desc='Train Epoch')
        else:
            train_tk = self.train_dl
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
            with torch.no_grad():
                self.meter_train.update(labels, outputs.detach().cpu(), loss.detach().cpu() * self.batch_repeat)

        if self.scheduler is not None:
            self.writer.add_scalar("lr", np.mean(self.scheduler.get_lr()), self.current_epoch)
            self.scheduler.step()
        else:
            self.writer.add_scalar("lr", self.lr, self.current_epoch)

        info = self.meter_train.log_metric()
        if self.rank == 0:
            print(f'Epoch {self.current_epoch}: train loss={info["loss"]:.4f} | train acc={info["acc"]:.4f}')

    def validate(self):
        self.eval()

        for inputs, labels, data in self.val_dl:
            loss, output = self.forward(inputs, labels)
            self.meter_val.update(labels, output.detach().cpu(), loss)

        info = self.meter_val.log_metric()
        selection_metric = info["acc"]  # not using loss but pos loss

        if self.rank != 0:
            return

        if selection_metric <= self.best_metric:
            self.best_metric = selection_metric
            print(f'>>> Saving best model metric={selection_metric:.4f}')
            checkpoint = {'model': self.model}
            torch.save(checkpoint, 'checkpoints/best_model.pth')
            train_info = self.meter_train.log_metric(write_scalar=False)
            self.history_best = {"train_"+key: value for key, value in train_info.items()}
            for key, value in info.items():
                self.history_best["val_"+key] = value
            self.history_best["epoch"] = self.current_epoch
            if settings.USE_FOUNDATIONS:
                foundations.save_artifact('checkpoints/best_model.pth', key='best_model_checkpoint')

        try:
            inputs, labels, data = next(self.visual_iter)
        except:
            self.visual_iter = iter(self.val_dl)
            inputs, labels, data = next(self.visual_iter)

        _, output = self.forward(inputs, labels)
        output = torch.sigmoid(output.detach().cpu())
        inputs = inputs.view((-1, ) + inputs.shape[-3:])
        self.writer.add_images(f'validate/{self.current_epoch}_inputs.png', self.unnorm(inputs)[:8], self.current_epoch)
        print(f'Epoch {self.current_epoch}: val loss={info["loss"]:.4f} | val acc={info["acc"]:.4f}')

    def start(self):
        self.current_epoch = 0
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            self.train_sampler.set_epoch(epoch)
            self.step()
            self.val_sampler.set_epoch(epoch)
            self.validate()
        if settings.USE_FOUNDATIONS and self.rank == 0:
            for key, value in self.history_best.items():
                foundations.log_metric(key, float(value))
