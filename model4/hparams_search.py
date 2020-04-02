import foundations
import numpy as np
import settings
from numpy.random import choice
import insightface.app.face_analysis

NUM_JOBS = 10

good_params = [
    # {'fix_fake': 1, 'all_data': 0, 'batch_size': 96, 'gpus': 4, 'batch_repeat': 1, 'n_epochs': 100, 'weight_decay': 1e-05, 'dropout': 0.8, 'smooth': 0.01, 'data_dropout': 0, 'input_mix': 0, 'RandomScale': 1, 'RandomRotate': 0, 'RandomErasing': 4, 'RandomCrop': 1, 'freeze': 1, 'max_lr': 0.001, 'use_lr_scheduler': 0, 'backbone': 7, 'use_hidden_layer': 0, 'same_transform': 0, 'img_diff': 0, 'val_rate': 1, 'data_path': '/data1/data/deepfake/dfdc_train', 'metadata_path': 0, 'bbox_path': '/data/deepfake/bbox_real.csv', 'cache_path': '/data/deepfake/video2/', 'diff_path': '/data/deepfake/diff.csv', 'seed': 1151917352},
    {'fix_fake': 1, 'all_data': 0, 'batch_size': 32, 'gpus': 3, 'batch_repeat': 1, 'n_epochs': 100, 'weight_decay': 1e-05, 'dropout': 0.95, 'smooth': 0.05, 'data_dropout': 0, 'input_mix': 0, 'RandomScale': 1, 'RandomRotate': 0, 'RandomErasing': 3, 'RandomCrop': 1, 'freeze': 0, 'max_lr': 0.0003, 'use_lr_scheduler': 0, 'backbone': 7, 'use_hidden_layer': 0, 'same_transform': 0, 'img_diff': 0, 'val_rate': 1, 'data_path': '/data1/data/deepfake/dfdc_train', 'metadata_path': 0, 'bbox_path': '/data/deepfake/bbox_real.csv', 'cache_path': '/data/deepfake/video2/', 'diff_path': '/data/deepfake/diff.csv', 'seed': 885744885},
]

def generate_params():
    return choice(good_params)
    params = {'batch_size': int(choice([96])),
              'gpus': int(choice([7])),
              'batch_repeat': int(choice([1])),
              'n_epochs': int(choice([100])),
              'weight_decay': float(choice([0.00001])),  # 0, 0.00001, 0.0001
              'dropout': float(choice([0.75, 0.9])),  # 0 0.3 [0.5] 0.7 0.9 (0)
              'smooth': float(choice([0, 0.025, 0.01])),
              'data_dropout': float(choice([0])),  #
              'input_mix': float(choice([0])),  #
              'all_data': int(choice([2])),  #
              'fix_fake': int(choice([1])),  #

              'RandomScale': int(choice([0])),  # (2)
              'RandomRotate': int(choice([0])),  # (2)
              'RandomErasing': int(choice([2, 3])),  # (2)
              'RandomCrop': int(choice([1])),  # (0)
              'freeze': int(choice([0])),

              'max_lr': float(choice([0.003, 0.001, 0.0003])),
              'use_lr_scheduler': int(choice([0])),
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
                           command=f'-m torch.distributed.launch --nproc_per_node={hyper_params["gpus"]} model4/main.py',
                           params=hyper_params, stream_job_logs=False, num_gpus=hyper_params["gpus"])