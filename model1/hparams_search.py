import foundations
import numpy as np
import settings

NUM_JOBS = 1


def generate_params():
    params = {'batch_size': int(np.random.choice([64])),
              'n_epochs': int(np.random.choice([50])),
              'weight_decay': float(np.random.choice([0.00001, 0.0001])),
              'dropout': float(np.random.choice([0.3, 0.5, 0.7])),
              'augment_level': int(np.random.choice([0, 1, 2, 3, 4, 5])),
              'max_lr': float(np.random.choice([0.001, 0.0003, 0.0001, 0.00003, 0.000001])),
              'use_lr_scheduler': int(np.random.choice([0, 1])),
              'scheduler_gamma': float(np.random.choice([0.95, 0.96, 0.94])),
              'use_hidden_layer': int(np.random.choice([0, 1])),
              'backbone': str(np.random.choice(["resnet18, resnet34, resnet50"])),
              'val_rate': 5,
              'data_path': settings.DATA_DIR,
              'metadata_path': settings.meta_data_path,
              'bbox_path': settings.bbox_path,
              'cache_path': settings.cache_path
              }

    return params


submitted_jobs = set()
for job_ in range(NUM_JOBS):
    print(f"packaging job {job_}")
    hyper_params = generate_params()
    while hyper_params in submitted_jobs:
        hyper_params = generate_params()
    submitted_jobs.add(hyper_params)
    print(hyper_params)
    foundations.submit(scheduler_config='scheduler', job_directory='.', command='main.py', params=hyper_params,
                       stream_job_logs=False, num_gpus=1)
