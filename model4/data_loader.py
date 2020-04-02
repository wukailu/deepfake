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
import settings
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
        self.use_diff = params["img_diff"]
        self.smooth = params["smooth"]
        self.fix_fake = params["fix_fake"]

        self.video_path = pathlib.Path(params['data_path'])
        self.cached_path = pathlib.Path(params['cache_path'])
        self.cached_path.mkdir(exist_ok=True)

        self.real2fakes = {fn: [] for fn in self.real_filename}
        filename_set = set(self.real_filename)
        for fn, row in metadata.iterrows():
            if row['label'] == 'FAKE' and row['original'] in filename_set:
                self.real2fakes[row['original']].append(fn)

        if self.fix_fake == 1:
            for key in self.real2fakes.keys():
                if len(self.real2fakes[key]) > 0:
                    self.real2fakes[key] = np.random.choice(self.real2fakes[key])

        import albumentations as aug
        self.trans1 = aug.Downscale(0.5, 0.5, p=1)
        self.trans2 = aug.JpegCompression(quality_lower=20, quality_upper=20, p=1)

    def __len__(self):
        return len(self.real_filename)

    def _get_png(self, path):
        if path.is_file():
            with open(path, 'rb') as f:
                face = Image.open(f)
                face.load()
        else:
            raise IOError("cache not found")
        return face

    def __get_transformed_face(self, file_path1, file_path2, augment=True) -> (torch.Tensor, pathlib.Path):
        face1 = self._get_png(file_path1)
        face2 = self._get_png(file_path2)

        if augment and np.random.rand() < 4 / 7:
            if np.random.rand() < 1 / 2:
                face1 = Image.fromarray(self.trans1(**{"image": np.array(face1)})["image"])
                face2 = Image.fromarray(self.trans1(**{"image": np.array(face2)})["image"])
            else:
                face1 = Image.fromarray(self.trans2(**{"image": np.array(face1)})["image"])
                face2 = Image.fromarray(self.trans2(**{"image": np.array(face2)})["image"])

        seed = random.randint(0, 2e9)
        random.seed(seed)
        image1 = self.transform(face1)
        random.seed(seed)
        image2 = self.transform(face2)
        if self.use_diff:
            return image1 - image2, (file_path1, file_path2)
        else:
            return image1, file_path1

    def _get_fake(self, real_fn, inter=1):
        if self.fix_fake == 1:
            fake_fn = self.real2fakes[real_fn]
        else:
            fake_fn = np.random.choice(self.real2fakes[real_fn])
        fold = np.random.choice(os.listdir(self.cached_path / fake_fn.split(".")[0]))
        ids = [int(x.split(".")[0]) for x in os.listdir(self.cached_path / fake_fn.split(".")[0] / fold)]
        ids = [i for i in ids if (i + inter) in ids]
        file = str(np.random.choice(ids)) + ".png"
        try:
            key = str((fake_fn.split(".")[0], int(fold), file))
            diff = self.diff.at[key, "diff"]
        except:
            diff = 100
        if diff < 0:
            raise ValueError("it's not a fake")
        return fake_fn, fold, int(file.split(".")[0])

    def __getitem__(self, idx: int):
        real_fn = self.real_filename[idx]
        inter = 1
        if np.random.rand() < 2 / 9:
            inter = 2
        for _ in range(20):
            try:
                fake_fn, fold, file = self._get_fake(real_fn, inter=inter)
                break
            except (KeyError, ValueError, FileNotFoundError) as e:
                pass
        try:
            real_path1 = self.cached_path / real_fn.split('.')[0] / fold / (str(file) + ".png")
            real_path2 = self.cached_path / real_fn.split('.')[0] / fold / (str(file + inter) + ".png")
            fake_path1 = self.cached_path / fake_fn.split('.')[0] / fold / (str(file) + ".png")
            fake_path2 = self.cached_path / fake_fn.split('.')[0] / fold / (str(file + inter) + ".png")
            if self.same_transform:
                temp_seed = np.random.randint(0, 2e9)
                random.seed(temp_seed)
                real_image, real_file = self.__get_transformed_face(real_path1, real_path2, augment=(inter == 1))
                random.seed(temp_seed)
                fake_image, fake_file = self.__get_transformed_face(fake_path1, fake_path2, augment=(inter == 1))
            else:
                real_image, real_file = self.__get_transformed_face(real_path1, real_path2, augment=(inter == 1))
                fake_image, fake_file = self.__get_transformed_face(fake_path1, fake_path2, augment=(inter == 1))

            return {'real': real_image, 'fake': fake_image, 'real_file': real_file, 'fake_file': fake_file,
                    'smooth': self.smooth}
        except (IOError, KeyError, NameError) as e:
            return {'real': None, 'fake': None, 'real_file': None, 'fake_file': None, 'smooth': 0}


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
    if params["all_data"] == 1:
        print("using val filter for train and val")
        print("using all data")
        val_data = metadata[metadata["label"] == 'REAL'].sample(100)
        val_index = val_data.index
        val_data = metadata[metadata.index.isin(val_index) | metadata["original"].isin(val_index)]
        train_data = metadata.drop(val_data.index)
    elif params["all_data"] == 0:
        train_data = metadata[metadata['split_kailu'] == 'train']
        val_data = metadata[metadata['split_kailu'] == 'validation']
    elif params["all_data"] == 2:
        print("using val filter for train and val")
        print("using all data except val")
        val_data = metadata[metadata['split_kailu'] == 'validation']
        train_data = metadata[metadata['split_kailu'] != 'validation']

    train_dl, train_sampler = _create_dataloader(train_data, bbox, params,
                                                 train_transforms, val_data_filter, diff, shuffle=True,
                                                 num_workers=loader, batch_size=params['batch_size'], drop_last=True)
    val_dl, val_sampler = _create_dataloader(val_data, bbox, params,
                                             val_transforms, val_data_filter, diff, shuffle=False, num_workers=loader,
                                             repeate=1, batch_size=params['batch_size'] // 3, drop_last=False)
    test_dl, test_sampler = _create_dataloader(metadata[metadata['split_kailu'] == 'test'], bbox, params,
                                               val_transforms, val_data_filter, diff, shuffle=False, num_workers=loader,
                                               batch_size=params['batch_size'], drop_last=False)
    return train_dl, val_dl, test_dl, iter(val_dl), train_sampler, val_sampler


def _create_dataloader(metadata: DataFrame, bbox: DataFrame, params: dict, transform, data_filter, diff,
                       num_workers=4, shuffle=True, repeate=1, batch_size=32, drop_last=True):
    assert len(metadata) != 0, f'metadata are empty'

    ds = DFDCDataset(metadata=metadata, bbox=bbox, params=params, transform=transform, data_filter=data_filter,
                     diff=diff)
    if repeate > 1:
        ds = torch.utils.data.ConcatDataset([ds] * repeate)

    sampler = DistributedSampler(ds)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                    collate_fn=collate_fn, drop_last=drop_last, sampler=sampler)
    print(f"data: {len(ds)}")
    return dl, sampler
