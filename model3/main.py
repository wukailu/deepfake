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
from model3.data_loader import create_dataloaders
from model3.model import create_model, print_model_params
from model3.train import Trainer

if settings.USE_FOUNDATIONS:
    import foundations

    params = foundations.load_parameters()
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    random.seed(params['seed'])
else:
    seed = 2018011328  # np.random.randint(2e9)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    import model3.hparams_search as hparams_search

    params = hparams_search.generate_params()
    params['seed'] = seed
    print(params)

params['metadata_path'] = settings.meta_data_path[params['metadata_path']]
params['batch_size'] = params['total_batch_size'] // params['num_segments']

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print('Creating loss function')
# Loss function
criterion = nn.BCEWithLogitsLoss()

print('Creating model')
# Create model, freeze layers and change last layer
model, params = create_model(params)
_ = print_model_params(model)
params_to_update = model.get_optim_policies()
# params_to_update = get_trainable_params(model)

print('Creating optimizer')
# Create optimizer and learning rate schedules
if params['use_lr_scheduler'] == 1:
    params['max_lr'] = params['max_lr'] * 10
optimizer = optim.Adam(params_to_update, lr=params['max_lr'], weight_decay=params['weight_decay'])
# optimizer = optim.SGD(params_to_update, lr=params['max_lr'], weight_decay=params['weight_decay'])
model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

# Learning rate scheme
if bool(params['use_lr_scheduler']) == 1:
    scheduler = lr_scheduler.StepLR(optimizer, step_size=params["n_epochs"] // 3, gamma=0.1)
else:
    scheduler = None

print('Creating datasets')
# Get dataloaders
train_dl, val_dl, test_dl = create_dataloaders(params, mean=model.input_mean, std=model.input_std)

print('Training start..')

trainer = Trainer(train_dl, val_dl, test_dl, model, optimizer, scheduler, criterion, params)

trainer.start()
