import foundations
import numpy as np
import settings

NUM_JOBS = 15


def generate_params():
    # we should increase batchsize
    # one run takes about 4 hour on V100
    params = {'total_batch_size': int(np.random.choice([256])),  # there's some bugs with DataParallel training
              'batch_repeat': int(np.random.choice([1, 2, 4])),  # Due to the problem with batch norm, batch_repeat>1 will be less efficient
              'num_epochs': int(np.random.choice([160])),
              'weight_decay': float(np.random.choice([0.00001])),  # 0, 0.00001, 0.0001
              'max_lr': float(np.random.choice([0.0002, 0.0001, 0.0005])),  # 0.0003, 0.0001, 0.00003
              'use_lr_scheduler': int(np.random.choice([0])),  # 0, 1
              'clip_gradient': int(np.random.choice([0, 20])),  # 20
              'same_transform': int(np.random.choice([0])),

              'backbone': int(np.random.choice([1])),
              'num_segments': int(np.random.choice([2, 4, 8])),   # 8 frames
              'shift_div': int(np.random.choice([8])),  # shift 1/8 of the channel

              'dropout': float(np.random.choice([0.3, 0.5, 0.7])),  # 0.3 0.5 0.7 0.9

              'RandomErasing': int(np.random.choice([1, 2, 3])),
              'RandomCrop': int(np.random.choice([0, 1, 2])),

              'data_path': settings.DATA_DIR,
              'metadata_path': int(np.random.choice(list(range(len(settings.meta_data_path))))),
              'cache_path': settings.video_cache_path,
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