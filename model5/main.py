import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import apex
from apex import amp
import apex.optimizers as apex_optim
from apex.parallel import DistributedDataParallel as DDP

import random
import sys

sys.path.append('/job/job_source/')
import settings
from model5.data_loader import create_dataloaders
from model5.model import create_model, print_model_params
from model5.train import Trainer

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
params['batch_size'] = params['total_batch_size'] // params['num_segments']
params['num_epochs'] = params['num_epochs'] // params['num_segments']

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if rank == 0:
    print('Creating loss function')
# Loss function
criterion = nn.BCEWithLogitsLoss()

if rank == 0:
    print('Creating model')
# Create model, freeze layers and change last layer
model, params = create_model(params)

model = apex.parallel.convert_syncbn_model(model)
model.cuda()
params_to_update = model.get_optim_policies()

if rank == 0:
    print('Creating optimizer')
# Create optimizer and learning rate schedules
if params['use_lr_scheduler'] == 1:
    params['max_lr'] = params['max_lr'] * 10
optimizer = apex_optim.FusedAdam(params_to_update, lr=params['max_lr'], weight_decay=params['weight_decay'])
model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
model = DDP(model, delay_allreduce=True)

# Learning rate scheme
if bool(params['use_lr_scheduler']) == 1:
    scheduler = lr_scheduler.StepLR(optimizer, step_size=params["n_epochs"] // 3, gamma=0.1)
else:
    scheduler = None

if rank == 0:
    print('Creating datasets')
# Get dataloaders
train_dl, val_dl, test_dl, train_sampler, val_sampler = create_dataloaders(params, mean=model.input_mean, std=model.input_std)

if settings.USE_FOUNDATIONS and rank == 0:
    foundations.log_params(params)
if rank == 0:
    print(params)
    print('Training start..')

trainer = Trainer(train_dl, val_dl, test_dl, train_sampler, val_sampler, model, optimizer, scheduler, criterion, params, rank)
trainer.start()
