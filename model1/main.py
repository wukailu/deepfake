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

params['metadata_path'] = settings.meta_data_path[params['metadata_path']]
# params['n_epochs'] *= params['batch_repeat']

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
if params['use_lr_scheduler'] == 3:
    params['max_lr'] = params['max_lr'] * 10

optimizer = optim.Adam(params_to_update, lr=params['max_lr'], weight_decay=params['weight_decay'])
model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
model = model.cuda()

# Learning rate scheme
if params['use_lr_scheduler'] == 0:
    scheduler = None
elif params['use_lr_scheduler'] == 1:
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=params['scheduler_gamma'])
elif params['use_lr_scheduler'] == 2:
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=7)
elif params['use_lr_scheduler'] == 3:
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
else:
    scheduler = None


print('Creating datasets')
# Get dataloaders
train_dl, val_dl, test_dl, val_dl_iter = create_dataloaders(params)
# pdb.set_trace()

if settings.USE_FOUNDATIONS:
    foundations.log_params(params)

print(params)
print('Training start..')
# Train
train(train_dl, val_dl, test_dl, val_dl_iter, model, optimizer, scheduler,
      criterion, params)
