import foundations
import numpy as np
import settings

good_params = [
    {'total_batch_size': 256, 'batch_repeat': 1, 'num_epochs': 160, 'weight_decay': 0.0001, 'max_lr': 0.0001, 'use_lr_scheduler': 0, 'clip_gradient': 0, 'same_transform': 0, 'label_smoothing': 0.15, 'backbone': 1, 'num_segments': 4, 'shift_div': 8, 'dropout': 0.5, 'RandomErasing': 0, 'RandomCrop': 0, 'data_path': '/data1/data/deepfake/dfdc_train', 'metadata_path': 0, 'cache_path': '/data/deepfake/video/', 'seed': 893248973},
    {'total_batch_size': 256, 'batch_repeat': 1, 'num_epochs': 160, 'weight_decay': 0.0005, 'max_lr': 1e-05, 'use_lr_scheduler': 0, 'clip_gradient': 10, 'same_transform': 0, 'label_smoothing': 0.05, 'backbone': 1, 'num_segments': 4, 'shift_div': 8, 'dropout': 0.3, 'RandomErasing': 4, 'RandomCrop': 2, 'data_path': '/data1/data/deepfake/dfdc_train', 'metadata_path': 0, 'cache_path': '/data/deepfake/video/', 'seed': 230503051},
]
NUM_JOBS = 7


def generate_params():
    # we should increase batchsize
    # return np.random.choice(good_params)
    # one run takes about 4 hour on V100
    params = {'total_batch_size': int(np.random.choice([256])),  # there's some bugs with DataParallel training
              'batch_repeat': int(np.random.choice([1])),  # Due to the problem with batch norm, batch_repeat>1 will be less efficient
              'num_epochs': int(np.random.choice([160])),
              'weight_decay': float(np.random.choice([0, 0.0001, 0.0005])),  # 0, 0.00001, 0.0001
              'max_lr': float(np.random.choice([0.0001, 1e-5, 1e-6])),  # 0.0003, 0.0001, 0.00003
              'use_lr_scheduler': int(np.random.choice([0])),  # 0, 1
              'clip_gradient': int(np.random.choice([0, 5, 10])),  # 20
              'same_transform': int(np.random.choice([0])),
              'label_smoothing': float(np.random.choice([0.05, 0.1])),

              'backbone': int(np.random.choice([1, 2])),
              'num_segments': int(np.random.choice([4])),   # 8 maybe better but need more time to converge
              'shift_div': int(np.random.choice([8])),  # shift 1/8 of the channel

              'dropout': float(np.random.choice([0.3, 0.5, 0.7])),  # 0.3 0.5 0.7 0.9

              'RandomErasing': int(np.random.choice([0, 1, 2, 3, 4])),
              'RandomCrop': int(np.random.choice([0])),

              'data_path': settings.DATA_DIR,
              'metadata_path': int(np.random.choice(list(range(len(settings.meta_data_path))))),
              'cache_path': settings.video_cache_path2,
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
        foundations.submit(scheduler_config='scheduler', job_directory='..', command='model3/main.py',
                           params=hyper_params, stream_job_logs=False, num_gpus=1)