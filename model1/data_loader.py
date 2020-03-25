import pathlib
import numpy as np
import pandas as pd
from pandas import DataFrame

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import random
import os


def collate_fn(batch: list):
    ret = {}
    for key in batch[0].keys():
        ret[key] = [item[key] for item in batch if item[key] is not None]
    # print(len(ret['real']))
    try:
        ret['real'] = torch.stack(ret['real'])
        ret['fake'] = torch.stack(ret['fake'])
        return ret, False
    except:
        ret['real'] = torch.zeros((1, 3, 224, 224))
        ret['fake'] = torch.zeros((1, 3, 224, 224))
        return ret, True


def train_data_filter(metadata_df):
    return metadata_df[(metadata_df['label'] == 'REAL') & (metadata_df['person_num'] == 1)]


def val_data_filter(metadata_df):
    return metadata_df[metadata_df['label'] == 'REAL']


class DFDCDataset(Dataset):
    def __init__(self, metadata: DataFrame, bbox: DataFrame, params: dict, transform, data_filter, diff):
        self.metadata_df = metadata
        self.real_filename = list(data_filter(metadata).index)
        self.bbox_df = bbox
        self.bbox_index_fn = set(bbox.index.get_level_values(0))
        self.transform = transform
        self.same_transform = params['same_transform']
        self.diff = diff

        self.video_path = pathlib.Path(params['data_path'])
        self.cached_path = pathlib.Path(params['cache_path'])
        self.cached_path.mkdir(exist_ok=True)

        self.real2fakes = {fn: [] for fn in self.real_filename}
        filename_set = set(self.real_filename)
        for fn, row in metadata.iterrows():
            if row['label'] == 'FAKE' and row['original'] in filename_set:
                self.real2fakes[row['original']].append(fn)

        import albumentations as aug
        self.trans = aug.OneOf([aug.Downscale(0.5, 0.5, p=0.66),
                                aug.JpegCompression(quality_lower=20, quality_upper=20, p=0.66),
                                aug.Flip(p=0)])

    def __len__(self):
        return len(self.real_filename)

    def __get_transformed_face(self, file_path) -> (torch.Tensor, pathlib.Path):
        if file_path.is_file():
            with open(file_path, 'rb') as f:
                face = Image.open(f)
                face.load()
        else:
            raise IOError("cache not found")

        img = np.array(face)
        face = Image.fromarray(self.trans(**{"image": img})["image"])

        image = self.transform(face)
        return image, file_path

    def _get_fake(self, real_fn):
        fake_fn = np.random.choice(self.real2fakes[real_fn])
        fold = np.random.choice(os.listdir(self.cached_path / fake_fn.split(".")[0]))
        file = np.random.choice(os.listdir(self.cached_path / fake_fn.split(".")[0] / fold))
        key = str((fake_fn.split(".")[0], int(fold), file))
        diff = self.diff.at[key, "diff"]
        if diff < 0.0:
            raise ValueError("it's not a fake")
        return fake_fn, fold, file

    def __getitem__(self, idx: int):
        real_fn = self.real_filename[idx]
        for _ in range(20):
            try:
                fake_fn, fold, file = self._get_fake(real_fn)
                break
            except (KeyError, ValueError, FileNotFoundError) as e:
                pass
        try:

            real_path = self.cached_path / real_fn.split('.')[0] / fold / file
            fake_path = self.cached_path / fake_fn.split('.')[0] / fold / file
            if self.same_transform:
                temp_seed = np.random.randint(0, 2e9)
                random.seed(temp_seed)
                real_image, real_file = self.__get_transformed_face(real_path)
                random.seed(temp_seed)
                fake_image, fake_file = self.__get_transformed_face(fake_path)
            else:
                real_image, real_file = self.__get_transformed_face(real_path)
                fake_image, fake_file = self.__get_transformed_face(fake_path)

            return {'real': real_image, 'fake': fake_image, 'real_file': real_file, 'fake_file': fake_file}
        except (IOError, KeyError, NameError) as e:
            return {'real': None, 'fake': None, 'real_file': None, 'fake_file': None}


def get_transforms(params, image_size=224):
    pre_trained_mean, pre_trained_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tensor_transform = []

    train_transforms = [transforms.RandomHorizontalFlip()]
    if params['RandomScale'] != 0:
        if params['RandomScale'] == 1:
            train_transforms.append(transforms.RandomAffine(degrees=0, scale=(.9, 1.1)))
        elif params['RandomScale'] == 2:
            train_transforms.append(transforms.RandomAffine(degrees=0, scale=(.8, 1.2)))
        elif params['RandomScale'] == 3:
            train_transforms.append(transforms.RandomAffine(degrees=0, scale=(.7, 1.3)))
    if params['RandomRotate'] != 0:
        if params['RandomRotate'] == 1:
            train_transforms.append(transforms.RandomAffine(degrees=10))
        elif params['RandomRotate'] == 2:
            train_transforms.append(transforms.RandomAffine(degrees=20))
        elif params['RandomRotate'] == 3:
            train_transforms.append(transforms.RandomAffine(degrees=30))
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
        elif params['RandomPerspective'] == 3:
            train_transforms.append(transforms.RandomPerspective(distortion_scale=0.3))
    elif params['RandomErasing'] != 0:
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
            train_transforms.append(transforms.Resize(image_size + 32))
            train_transforms.append(transforms.RandomCrop(image_size, padding_mode='constant'))
        elif params['RandomCrop'] == 2:
            train_transforms.append(transforms.Resize(image_size + 64))
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
    diff = pd.read_csv(params["diff_path"], index_col=[0])

    loader = 8
    # train_dl = _create_dataloader(metadata[metadata['split_kailu'] == 'validation'], bbox, params, train_transforms,
    #                             train_data_filter, shuffle=True, num_workers=loader, repeate=5)
    train_dl = _create_dataloader(metadata[metadata['split_kailu'] == 'train'], bbox, params, train_transforms,
                                  train_data_filter, diff, shuffle=True, num_workers=loader)
    val_dl = _create_dataloader(metadata[metadata['split_kailu'] == 'validation'], bbox, params, val_transforms,
                                train_data_filter, diff, shuffle=False, num_workers=loader, repeate=1)
    test_dl = _create_dataloader(metadata[metadata['split_kailu'] == 'test'], bbox, params, val_transforms,
                                 val_data_filter, diff, shuffle=False, num_workers=loader)

    return train_dl, val_dl, test_dl, iter(val_dl)


def _create_dataloader(metadata: DataFrame, bbox: DataFrame, params: dict, transform, data_filter, diff,
                       num_workers=4, shuffle=True, repeate=1):
    assert len(metadata) != 0, f'metadata are empty'

    ds = DFDCDataset(metadata=metadata, bbox=bbox, params=params, transform=transform, data_filter=data_filter, diff=diff)
    if repeate > 1:
        ds = torch.utils.data.ConcatDataset([ds]*repeate)
    dl = DataLoader(ds, batch_size=params['batch_size'], num_workers=num_workers, shuffle=shuffle,
                    collate_fn=collate_fn, drop_last=True)

    print(f"data: {len(ds)}")
    return dl