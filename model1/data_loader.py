import pathlib
import numpy as np
import pandas as pd
from pandas import DataFrame

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import settings
import random


def collate_fn(batch: list):
    ret = {}
    for key in batch[0].keys():
        ret[key] = [item[key] for item in batch if item[key] is not None]
    # print(len(ret['real']))
    try:
        ret['real'] = torch.stack(ret['real'])
        ret['fake'] = torch.stack(ret['fake'])
    except:
        ret['real'] = torch.zeros((1, 3, 224, 224))
        ret['fake'] = torch.zeros((1, 3, 224, 224))
    return ret


def train_data_filter(metadata_df):
    return metadata_df[(metadata_df['label'] == 'REAL') & (metadata_df['person_num'] == 1)]


def val_data_filter(metadata_df):
    return metadata_df[metadata_df['label'] == 'REAL']


class DFDCDataset(Dataset):
    def __init__(self, metadata: DataFrame, bbox: DataFrame, params: dict, transform, data_filter):
        self.metadata_df = metadata
        self.real_filename = list(data_filter(metadata).index)
        self.bbox_df = bbox
        self.bbox_index_fn = set(bbox.index.get_level_values(0))
        self.transform = transform
        self.same_transform = params['same_transform']

        self.video_path = pathlib.Path(params['data_path'])
        self.cached_path = pathlib.Path(params['cache_path'])
        self.cached_path.mkdir(exist_ok=True)

        self.real2fakes = {fn: [] for fn in self.real_filename}
        filename_set = set(self.real_filename)
        for fn, row in metadata.iterrows():
            if row['label'] == 'FAKE' and row['original'] in filename_set:
                self.real2fakes[row['original']].append(fn)

    def __len__(self):
        return len(self.real_filename)

    def __get_transformed_face(self, filename: str, frame: int) -> (torch.Tensor, pathlib.Path):
        cached_dir = self.cached_path / filename.split('.')[0]
        cached_file = cached_dir / (str(frame) + '.png')

        if cached_file.is_file():
            with open(cached_file, 'rb') as f:
                face = Image.open(f)
                face.load()
        else:
            raise IOError("cache not found")

        assert face.size == (224, 224)
        image = self.transform(face)
        return image, cached_file

    def __getitem__(self, idx: int):
        real_fn = self.real_filename[idx]
        fake_fn = np.random.choice(self.real2fakes[real_fn])

        try:
            frames = self.bbox_df.xs(real_fn, level=0).index.get_level_values(0)
            frame = np.random.choice(frames)

            temp_seed = np.random.randint(0, 2e9)
            if self.same_transform:
                random.seed(temp_seed)
            real_image, real_file = self.__get_transformed_face(real_fn, frame)
            if self.same_transform:
                random.seed(temp_seed)
            fake_image, fake_file = self.__get_transformed_face(fake_fn, frame)

            return {'real': real_image, 'fake': fake_image, 'real_file': real_file, 'fake_file': fake_file}
        except (IOError, KeyError) as e:
            return {'real': None, 'fake': None, 'real_file': None, 'fake_file': None}


def get_transforms(params, image_size=224):
    pre_trained_mean, pre_trained_std = [0.439, 0.328, 0.304], [0.232, 0.206, 0.201]
    tensor_transform = []

    train_transforms = [transforms.RandomHorizontalFlip()]
    if params['RandomAffine'] != 0:
        if params['RandomAffine'] == 1:
            train_transforms.append(transforms.RandomAffine(degrees=20, scale=(.8, 1.2), shear=0))
        elif params['RandomAffine'] == 2:
            train_transforms.append(transforms.RandomAffine(degrees=40, scale=(.7, 1.3), shear=0))
    elif params['ColorJitter'] != 0:
        if params['ColorJitter'] == 1:
            train_transforms.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
        elif params['ColorJitter'] == 2:
            train_transforms.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5))
    elif params['RandomPerspective'] != 0:
        if params['RandomPerspective'] == 1:
            train_transforms.append(transforms.RandomPerspective(distortion_scale=0.1))
        elif params['RandomPerspective'] == 2:
            train_transforms.append(transforms.RandomPerspective(distortion_scale=0.2))
    elif params['RandomErasing'] != 0:
        if params['RandomErasing'] == 1:
            tensor_transform.append(transforms.RandomErasing(scale=(0.02, 0.15), ratio=(0.3, 1.6)))
        elif params['RandomErasing'] == 2:
            tensor_transform.append(transforms.RandomErasing(scale=(0.1, 0.5), ratio=(0.2, 5)))
    elif params['RandomCrop'] != 0:
        train_transforms.append(transforms.Resize(image_size + 32))
        train_transforms.append(transforms.RandomCrop(image_size, padding_mode='constant'))

    train_transforms = transforms.Compose(train_transforms)
    tensor_transform = transforms.Compose(tensor_transform)

    train_transforms = transforms.Compose([
        train_transforms,
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=pre_trained_mean, std=pre_trained_std),
        tensor_transform,
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=pre_trained_mean, std=pre_trained_std)
    ])
    return train_transforms, val_transforms


def create_dataloaders(params: dict):
    train_transforms, val_transforms = get_transforms(params, image_size=224)
    metadata = pd.read_json(params['metadata_path']).T
    bbox = pd.read_csv(params['bbox_path'], index_col=[0, 1, 2])
    loader = 64
    train_dl = _create_dataloader(metadata[metadata['split_kailu'] == 'train'], bbox, params, train_transforms,
                                  train_data_filter, shuffle=True, num_workers=loader)
    val_dl = _create_dataloader(metadata[metadata['split_kailu'] == 'validation'], bbox, params, val_transforms,
                                val_data_filter, shuffle=False, num_workers=loader)
    test_dl = _create_dataloader(metadata[metadata['split_kailu'] == 'test'], bbox, params, val_transforms,
                                 val_data_filter, shuffle=False, num_workers=loader)

    return train_dl, val_dl, test_dl, iter(val_dl)


def _create_dataloader(metadata: DataFrame, bbox: DataFrame, params: dict, transform, data_filter,
                       num_workers=4, shuffle=True):
    assert len(metadata) != 0, f'metadata are empty'

    ds = DFDCDataset(metadata=metadata, bbox=bbox, params=params, transform=transform, data_filter=data_filter)
    dl = DataLoader(ds, batch_size=params['batch_size'], num_workers=num_workers, shuffle=shuffle,
                    collate_fn=collate_fn, drop_last=True)

    print(f"data: {len(ds)}")
    return dl