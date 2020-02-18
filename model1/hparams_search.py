import foundations
import numpy as np
import settings

NUM_JOBS = 50


def generate_params():
    params = {'batch_size': int(np.random.choice([128])),
              'n_epochs': int(np.random.choice([40])),
              'weight_decay': float(np.random.choice([0.00001])),  # 0, 0.00001, 0.0001
              'dropout': float(np.random.choice([0, 0.5])),  # 0.3 [0.5] 0.7 0.9
              
              # 'RandomAffine': int(np.random.choice([1])),
              # 'ColorJitter': int(np.random.choice([1])),
              # 'RandomPerspective': int(np.random.choice([0])),
              # 'RandomErasing': int(np.random.choice([2])),
              # 'RandomCrop': int(np.random.choice([0])),

              'RandomAffine': int(np.random.choice([1])),
              'ColorJitter': int(np.random.choice([0])),
              'RandomPerspective': int(np.random.choice([0])),
              'RandomErasing': int(np.random.choice([2])),
              'RandomCrop': int(np.random.choice([0])),

              'max_lr': float(np.random.choice([0.01, 0.003, 0.001, 0.0003, 0.0001])),  # 0.0003, [0.0001], 0.00003
              'use_lr_scheduler': int(np.random.choice([0, 1, 2])),  # 0, 1
              'scheduler_gamma': float(np.random.choice([0.96, 0.9, 0.8])),  # 0.96, 0.95, 0.94
              'use_hidden_layer': int(np.random.choice([0])),  # 0, 1
              'backbone': int(np.random.choice([7])),  # 1, 2, 3, 4, 5, 6, 7, 8
              'same_transform': int(np.random.choice([1])),
              'val_rate': 1,
              'data_path': settings.DATA_DIR,
              'metadata_path': int(np.random.choice(list(range(len(settings.meta_data_path))))),  # 0, 1, 2
              'bbox_path': settings.bbox_path,
              'cache_path': settings.face_cache_path,
              }

    return params


if __name__ == "__main__":
    submitted_jobs = set()
    for job_ in range(NUM_JOBS):
        print(f"packaging job {job_}")
        hyper_params = generate_params()
        while frozenset(hyper_params.items()) in submitted_jobs:
            hyper_params = generate_params()
        submitted_jobs.add(frozenset(hyper_params.items()))

        seed = np.random.randint(2e9)
        hyper_params['seed'] = int(seed)
        print(hyper_params)
        foundations.submit(scheduler_config='scheduler', job_directory='/home/kailu/deepfake',
                           command='model1/main.py', params=hyper_params, stream_job_logs=False, num_gpus=1)
