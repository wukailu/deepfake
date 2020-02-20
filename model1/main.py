import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from apex import amp
import random

import sys
sys.path.append('/job/job_source/')
import settings
from model1.data_loader import create_dataloaders
from model1.model import get_trainable_params, create_model, print_model_params
from model1.train import train

# import pdb

if settings.USE_FOUNDATIONS:
    import foundations
    params = foundations.load_parameters()
    # Fix random seed
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    random.seed(params['seed'])
else:
    seed = np.random.randint(2e9)  # 2018011328
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    import hparams_search
    params = hparams_search.generate_params()
    params['seed'] = seed

print(params)
# params = {'batch_size': 64, 'n_epochs': 100, 'weight_decay': 0.0001, 'dropout': 0.7, 'augment_level': 3, 'max_lr': 0.0003, 'use_lr_scheduler': 0, 'scheduler_gamma': 0.95, 'use_hidden_layer': 0, 'backbone': 'resnet34', 'val_rate': 1, 'data_path': '/data1/data/deepfake/dfdc_train', 'metadata_path': '/data1/data/deepfake/dfdc_train/metadata_kailu.json', 'bbox_path': '/data1/data/deepfake/bbox_real.csv', 'cache_path': '/data1/data/deepfake/face/', 'seed': 1378744497}

params['metadata_path'] = settings.meta_data_path[params['metadata_path']]

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print('Creating loss function')
# Loss function
criterion = nn.BCEWithLogitsLoss()

print('Creating model')
# Create model, freeze layers and change last layer
model, params = create_model(bool(params['use_hidden_layer']), params['dropout'], params['backbone'], params)
_ = print_model_params(model)
params_to_update = get_trainable_params(model)

print('Creating optimizer')
# Create optimizer and learning rate schedules
optimizer = optim.Adam(params_to_update, lr=params['max_lr'], weight_decay=params['weight_decay'])
model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

# Learning rate scheme
if params['use_lr_scheduler'] == 0:
    scheduler = None
elif params['use_lr_scheduler'] == 1:
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=params['scheduler_gamma'])
elif params['use_lr_scheduler'] == 2:
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


print('Creating datasets')
# Get dataloaders
train_dl, val_dl, test_dl, val_dl_iter = create_dataloaders(params)
# pdb.set_trace()

if settings.USE_FOUNDATIONS:
    foundations.log_params(params)

print('Training start..')
# Train
train(train_dl, val_dl, test_dl, val_dl_iter, model, optimizer, params['n_epochs'], params['max_lr'], scheduler,
      criterion, params['val_rate'])
