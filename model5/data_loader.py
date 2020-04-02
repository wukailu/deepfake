import pathlib
import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import random
import os
import albumentations as aug


def collate_fn(batch: list):
    ret = {}
    for key in batch[0].keys():
        ret[key] = [item[key] for item in batch if item[key] is not None]
    try:
        shape = len(ret['real'])
        ret['real'] = torch.stack(ret['real'], dim=0)
        ret['fake'] = torch.stack(ret['fake'], dim=0)
        smooth = ret['smooth'][0]
    except:
        print("empty batch")
        shape = 1
        ret['real'] = torch.zeros((8, 3, 224, 224))
        ret['fake'] = torch.zeros((8, 3, 224, 224))
        smooth = 0

    inputs = torch.cat([ret['real'], ret['fake']], 0)
    labels = torch.cat([torch.zeros((shape, 1)), torch.ones((shape, 1))], 0)
    if smooth != 0:
        mask = (torch.randint_like(labels, 0, int(1 / smooth) + 1) // int(1 / smooth)).bool()
        labels[mask] = -labels[mask] + 1

    return inputs, labels, ret


class DFDCVideoDataset(Dataset):
    def __init__(self, metadata: DataFrame, params: dict, transform, data_filter, frame_num=8):
        self.metadata_df = metadata
        self.real_filename = list(data_filter(metadata).index)
        self.transform = transform
        self.frame_num = frame_num
        self.same_transform = params['same_transform']
        self.smooth = params['label_smoothing']
        self.trans = aug.OneOf([aug.Downscale(0.5, 0.5, p=0.666),
                                aug.JpegCompression(quality_lower=20, quality_upper=20, p=0.666), aug.Flip(p=0)])

        self.video_path = pathlib.Path(params['data_path'])
        self.cached_path = pathlib.Path(params['cache_path'])
        self.cached_path.mkdir(exist_ok=True)
        self.data_dropout = params['data_dropout']
        self.input_mix = params['input_mix']

        np.random.shuffle(self.real_filename)
        self.real_filename = self.real_filename[:int(len(self.real_filename)*(1-self.data_dropout))]

        self.real2fakes = {fn: [] for fn in self.real_filename}
        filename_set = set(self.real_filename)
        for fn, row in metadata.iterrows():
            if row['label'] == 'FAKE' and row['original'] in filename_set:
                self.real2fakes[row['original']].append(fn)

    def __len__(self):
        return len(self.real_filename)

    def __get_transformed_face(self, filename: str, frame: int) -> (torch.Tensor, pathlib.Path):
        cached_dir = self.cached_path / filename.split('.')[0] / str(0)
        cached_file = cached_dir / (str(frame) + '.png')

        if cached_file.is_file():
            with open(cached_file, 'rb') as f:
                face = Image.open(f)
                face.load()
        else:
            raise IOError("cache not found")

        # Note That input is not resized!!!!!!!!!
        # assert face.size == (224, 224)

        img = np.array(face)
        face = Image.fromarray(self.trans(**{"image": img})["image"])
        image = self.transform(face)
        return image, cached_file

    def _read_frames(self, filename, frames: list, temp_seed=-1):
        if temp_seed == -1:
            temp_seed = np.random.randint(0, 2e9)
        # print(temp_seed)
        real_images, real_files = [], []
        for frame in frames:
            random.seed(temp_seed)
            real_image, real_file = self.__get_transformed_face(filename, frame)
            real_images.append(real_image)
            real_files.append(real_file)
        return torch.stack(real_images), real_files

    def _random_frames(self, all_frames):
        num_segments = self.frame_num
        new_length = 1
        average_duration = (len(all_frames) - new_length + 1) // num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(num_segments)), average_duration) + np.random.randint(average_duration,
                                                                                                   size=num_segments)
        else:
            raise IOError(f"No enough frames, expect at least {num_segments}, but only has {len(all_frames)}.")
        return np.array(all_frames)[offsets]

    def __getitem__(self, idx: int):
        real_fn = self.real_filename[idx]
        fake_fn = np.random.choice(self.real2fakes[real_fn])

        try:
            files = os.listdir(self.cached_path / real_fn.split(".")[0] / str(0))

            ids = sorted([int(file.split(".")[0]) for file in files])
            frames = self._random_frames(ids)

            real_image, real_file = self._read_frames(real_fn, frames)
            fake_image, fake_file = self._read_frames(fake_fn, frames)

            if np.random.rand() < self.input_mix:
                real_image = (real_image + fake_image)/2
                fake_image = real_image

            return {'real': real_image, 'fake': fake_image, 'real_file': real_file, 'fake_file': fake_file, 'smooth': self.smooth}
        except (IOError, KeyError) as e:
            # print(e)
            return {'real': None, 'fake': None, 'real_file': None, 'fake_file': None, "smooth": None}


def get_transforms(params, image_size, mean, std):
    tensor_transform = []

    train_transforms = [transforms.RandomHorizontalFlip()]
    if params['RandomErasing'] != 0:
        if params['RandomErasing'] == 1:
            tensor_transform.append(transforms.RandomErasing(scale=(0.1, 0.3), ratio=(0.2, 5)))
        elif params['RandomErasing'] == 2:
            tensor_transform.append(transforms.RandomErasing(scale=(0.3, 0.4), ratio=(0.2, 5)))
        elif params['RandomErasing'] == 3:
            tensor_transform.append(transforms.RandomErasing(scale=(0.4, 0.5), ratio=(0.2, 5)))
        elif params['RandomErasing'] == 4:
            tensor_transform.append(transforms.RandomErasing(scale=(0.5, 0.6), ratio=(0.2, 5)))
    elif params['RandomCrop'] != 0:
        if params['RandomCrop'] == 1:
            train_transforms.append(transforms.RandomResizedCrop(image_size, scale=(0.8, 1), ratio=(0.8, 1.25)))
        elif params['RandomCrop'] == 2:
            train_transforms.append(transforms.RandomResizedCrop(image_size, scale=(0.75, 1), ratio=(0.75, 1.33333)))
        elif params['RandomCrop'] == 3:
            train_transforms.append(transforms.RandomResizedCrop(image_size, scale=(0.66, 1), ratio=(0.666, 1.51515)))

    train_transforms = transforms.Compose(train_transforms)
    tensor_transform = transforms.Compose(tensor_transform)

    train_transforms = transforms.Compose([
        train_transforms,
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        tensor_transform,
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return train_transforms, val_transforms


def create_dataloaders(params: dict, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), image_size=224):
    def train_data_filter(metadata_df):
        return metadata_df[(metadata_df['label'] == 'REAL') & (metadata_df['person_num'] == 1)]

    def val_data_filter(metadata_df):
        return metadata_df[metadata_df['label'] == 'REAL']

    train_transforms, val_transforms = get_transforms(params, image_size, mean, std)
    metadata = pd.read_json(params['metadata_path']).T
    loader = 8
    # metadata[metadata['split_kailu'] == 'train']
    print("training on all the data")
    train_dl, sampler = _create_dataloader(metadata, params, train_transforms,
                                  val_data_filter, shuffle=True, num_workers=loader, batch_size=params['batch_size'])
    val_dl, val_sampler = _create_dataloader(metadata[metadata['split_kailu'] == 'validation'], params, val_transforms,
                                val_data_filter, shuffle=False, num_workers=loader, batch_size=params['batch_size']//3)
    test_dl, _ = _create_dataloader(metadata[metadata['split_kailu'] == 'test'], params, val_transforms,
                                 val_data_filter, shuffle=False, num_workers=loader, batch_size=params['batch_size'])

    return train_dl, val_dl, test_dl, sampler, val_sampler


def _create_dataloader(metadata: DataFrame, params: dict, transform, data_filter,
                       num_workers=4, shuffle=True, repeate=1, batch_size=32):
    assert len(metadata) != 0, f'metadata are empty'

    ds = DFDCVideoDataset(metadata=metadata, params=params, transform=transform, data_filter=data_filter,
                          frame_num=params["num_segments"])
    if repeate > 1:
        ds = torch.utils.data.ConcatDataset([ds] * repeate)

    sampler = DistributedSampler(ds)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                    collate_fn=collate_fn, drop_last=True)

    print(f"data: {len(ds)}")
    return dl, sampler
