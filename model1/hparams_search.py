import foundations
import numpy as np
import settings
from numpy.random import choice

NUM_JOBS = 21


def generate_params():
    # return choice(good_params)
    # overfit return {'batch_size': 64, 'n_epochs': 40, 'weight_decay': 1e-05, 'dropout': 0.0, 'RandomAffine': 2, 'ColorJitter': 0, 'RandomPerspective': 1, 'RandomErasing': 0, 'RandomCrop': 0, 'freeze': 0, 'max_lr': 0.0003, 'use_lr_scheduler': 0, 'scheduler_gamma': 0.9, 'use_hidden_layer': 1, 'backbone': 7, 'same_transform': 1, 'val_rate': 1, 'data_path': '/data1/data/deepfake/dfdc_train', 'metadata_path': 0, 'bbox_path': '/data/deepfake/bbox_real.csv', 'cache_path': '/data/deepfake/faces/', 'seed': 254089471}

    params = {'batch_size': int(choice([32])),
              'batch_repeat': int(choice([3, 6, 9])),
              'n_epochs': int(choice([40])),
              'weight_decay': float(choice([0.00001])),  # 0, 0.00001, 0.0001
              'dropout': float(choice([0.75])),  # 0 0.3 [0.5] 0.7 0.9 (0)

              'RandomScale': int(choice([0, 1])),  # (2)
              'RandomRotate': int(choice([0, 1])),  # (2)
              'ColorJitter': int(choice([0])),  # (0)
              'RandomPerspective': int(choice([0, 1, 2])),  # (1)
              'RandomErasing': int(choice([1, 2, 3, 4])),  # (2)
              'RandomCrop': int(choice([0, 1, 2])),  # (0)
              'freeze': int(choice([0, 1])),

              'max_lr': float(choice([0.005, 0.002])),
              'use_lr_scheduler': int(choice([0])),  # 0, 1 (2)
              'scheduler_gamma': float(choice([0.9])),  # 0.96, 0.95, 0.94 (0.96)
              'use_hidden_layer': int(choice([0])),  # 0, (1)
              'backbone': int(choice([7])),  # 1, 2, 3, 4, 5, 6, 7, 8, 9
              'same_transform': int(choice([0, 1])),  # TODO: Why
              'val_rate': 1,
              'data_path': settings.DATA_DIR,
              'metadata_path': int(choice(list(range(len(settings.meta_data_path))))),  # 0, 1, 2
              'bbox_path': settings.bbox_path,
              'cache_path': settings.video_cache_path,
              'diff_path': settings.diff_dict_path,
              }
    return params


if __name__ == "__main__":
    submitted_jobs = set()
    for job_ in range(NUM_JOBS):
        print(f"packaging job {job_}")
        hyper_params = generate_params()
        # while frozenset(hyper_params.items()) in submitted_jobs:
        #     hyper_params = generate_params()
        # submitted_jobs.add(frozenset(hyper_params.items()))

        seed = np.random.randint(2e9)
        hyper_params['seed'] = int(seed)
        print(hyper_params)
        foundations.submit(scheduler_config='scheduler', job_directory='/home/kailu/deepfake',
                           command='model1/main.py', params=hyper_params, stream_job_logs=False, num_gpus=1)
