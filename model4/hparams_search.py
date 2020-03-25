import foundations
import numpy as np
import settings
from numpy.random import choice

NUM_JOBS = 40

good_params = [
    {'batch_size': 32, 'batch_repeat': 1, 'n_epochs': 50, 'weight_decay': 1e-05, 'dropout': 0.7, 'smooth': 0.05, 'RandomScale': 0, 'RandomRotate': 0, 'RandomErasing': 4, 'RandomCrop': 1, 'freeze': 1, 'max_lr': 0.0001, 'backbone': 7, 'use_hidden_layer': 0, 'same_transform': 0, 'img_diff': 0, 'val_rate': 1, 'data_path': '/data1/data/deepfake/dfdc_train', 'metadata_path': 0, 'bbox_path': '/data/deepfake/bbox_real.csv', 'cache_path': '/data/deepfake/video2/', 'diff_path': '/data/deepfake/diff.csv', 'seed': 1128054550},
]

def generate_params():
    # return choice(good_params)
    params = {'batch_size': int(choice([96])),
              'batch_repeat': int(choice([1])),
              'n_epochs': int(choice([100])),
              'weight_decay': float(choice([0.00001])),  # 0, 0.00001, 0.0001
              'dropout': float(choice([0.9, 0.95])),  # 0 0.3 [0.5] 0.7 0.9 (0)
              'smooth': float(choice([0.025, 0.05, 0.10, 0.125])),

              'RandomScale': int(choice([0, 1])),  # (2)
              'RandomRotate': int(choice([0])),  # (2)
              'RandomErasing': int(choice([3, 4])),  # (2)
              'RandomCrop': int(choice([0, 1])),  # (0)
              'freeze': int(choice([0, 1])),

              'max_lr': float(choice([0.001, 0.0006, 0.0003, 0.0002])),
              'backbone': int(choice([7])),  # 1, 2, 3, 4, 5, 6, 7, 8, 9
              'use_hidden_layer': int(np.random.choice([0])),
              'same_transform': int(choice([0])),
              'img_diff': int(choice([0])),
              'val_rate': 1,
              'data_path': settings.DATA_DIR,
              'metadata_path': int(choice(list(range(len(settings.meta_data_path))))),  # 0, 1, 2
              'bbox_path': settings.bbox_path,
              'cache_path': settings.video_cache_path2,
              'diff_path': settings.diff_dict_path,
              }
    return params


if __name__ == "__main__":
    submitted_jobs = set()
    for job_ in range(NUM_JOBS):
        print(f"packaging job {job_}")
        hyper_params = generate_params()

        seed = np.random.randint(2e9)
        hyper_params['seed'] = int(seed)
        print(hyper_params)
        foundations.submit(scheduler_config='scheduler', job_directory='/home/kailu/deepfake',
                           command='-m torch.distributed.launch --nproc_per_node=2 model4/main.py',
                           params=hyper_params, stream_job_logs=False, num_gpus=2)