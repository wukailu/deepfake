import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import apex
from apex import amp
import apex.optimizers as apex_optim
from apex.parallel import DistributedDataParallel as DDP
# from torch.nn.parallel import DistributedDataParallel as DDP

import random

import sys

sys.path.append('/job/job_source/')
import settings
from model4.data_loader import create_dataloaders
from model4.model import get_trainable_params, create_model, print_model_params
from model4.train import train

###
# python -m torch.distributed.launch --nproc_per_node=8 ../main.py
# nvidia-smi         发现内存泄露问题，即没有进程时，内存被占用
# kill $(ps aux | grep "main.py" | grep -v grep | awk '{print $2}')
###

if settings.USE_FOUNDATIONS:
    import foundations

    params = foundations.load_parameters()
    # Fix random seed
    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    random.seed(params['seed'])
else:
    seed = 2018011328  # np.random.randint(2e9)  #
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    import hparams_search
    params = hparams_search.generate_params()
    params['seed'] = seed

torch.distributed.init_process_group(backend='nccl', init_method='env://')
rank = torch.distributed.get_rank()
torch.cuda.set_device(rank)
device = torch.device("cuda", rank)

if rank == 0:
    print(params)

params['metadata_path'] = settings.meta_data_path[params['metadata_path']]
# params['n_epochs'] *= params['batch_repeat']

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if rank == 0:
    print('Creating loss function')

# Loss function
criterion = nn.BCEWithLogitsLoss()

if rank == 0:
    print('Creating model')

# Create model, freeze layers and change last layer
model, params = create_model(bool(params['use_hidden_layer']), params['dropout'], params['backbone'], params)
_ = print_model_params(model)

model = apex.parallel.convert_syncbn_model(model)
model.cuda()
params_to_update = get_trainable_params(model)

if rank == 0:
    print('Creating optimizer')
# Create optimizer and learning rate schedules
# if settings.USE_FOUNDATIONS:
#     optimizer = optim.Adam(params_to_update, lr=params['max_lr'], weight_decay=params['weight_decay'])
# else:
optimizer = apex_optim.FusedAdam(params_to_update, lr=params['max_lr'], weight_decay=params['weight_decay'])
model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

# if not settings.USE_FOUNDATIONS:
model = DDP(model, delay_allreduce=True)
# model = DDP(model)

# Learning rate scheme
if params['use_lr_scheduler'] == 0:
    scheduler = None
elif params['use_lr_scheduler'] == 1:
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=params['scheduler_gamma'])
elif params['use_lr_scheduler'] == 2:
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
elif params['use_lr_scheduler'] == 3:
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
else:
    scheduler = None

if rank == 0:
    print('Creating datasets')
# Get dataloaders
train_dl, val_dl, test_dl, val_dl_iter, train_sampler, val_sampler = create_dataloaders(params)
# pdb.set_trace()

if settings.USE_FOUNDATIONS and rank == 0:
    foundations.log_params(params)

if rank == 0:
    print(params)
    print('Training start..')
# Train
train(train_dl, val_dl, test_dl, val_dl_iter, model, optimizer, scheduler,
      criterion, params, train_sampler, val_sampler, rank)
