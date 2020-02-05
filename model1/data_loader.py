import pathlib
import numpy as np
import os
import pandas as pd
from pandas import DataFrame

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import settings
import time


def collate_fn(batch: list):
    ret = {}
    for key in batch[0].keys():
        ret[key] = [item[key] for item in batch if item[key] is not None]
    ret['real'] = torch.stack(ret['real'])
    ret['fake'] = torch.stack(ret['fake'])
    return ret


def train_data_filter(metadata_df):
    return metadata_df[(metadata_df['label'] == 'REAL') & (metadata_df['person_num'] == 1)]


def val_data_filter(metadata_df):
    return metadata_df[metadata_df['label'] == 'REAL']


class DFDCDataset(Dataset):
    def __init__(self, metadata: DataFrame, bbox: DataFrame, video_path: str, cached_path: str, transform,
                 data_filter):
        self.metadata_df = metadata
        self.real_filename = list(data_filter(metadata).index)
        self.bbox_df = bbox
        self.bbox_index_fn = set(bbox.index.get_level_values(0))
        self.transform = transform
        self.video_path = pathlib.Path(video_path)

        self.cached_path = pathlib.Path(cached_path)
        self.cached_path.mkdir(exist_ok=True)
        self.real2fakes = {fn: [] for fn in self.real_filename}
        filename_set = set(self.real_filename)
        for fn, row in metadata.iterrows():
            if row['label'] == 'FAKE' and row['original'] in filename_set:
                self.real2fakes[row['original']].append(fn)

    def __len__(self):
        return len(self.real_filename)

    def __get_transformed_face(self, filename: str, frame: int, bbox: np.ndarray = None) -> (
            torch.Tensor, pathlib.Path):
        cached_dir = self.cached_path / filename.split('.')[0]
        cached_file = cached_dir / (str(frame) + '.npy')

        if cached_file.is_file():
            face = np.load(cached_file)
        else:
            cap = cv2.VideoCapture(os.path.join(self.video_path, filename))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)  # 设置要获取的帧号
            success = cap.grab()
            _, frame_img = cap.retrieve()
            # _, frame_img = cap.read()
            if not _:
                settings.un_normal_list.append(filename)
                raise IOError(f"Reading {filename} Failed, frame = {frame}, path = {self.video_path}")
            cap.release()

            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            if bbox is None:
                left, top, right, bottom = np.fromstring(self.bbox_df.at[(filename, frame, 0), 'bbox'][1:-1],
                                                         sep=' ').astype(int)
            else:
                left, top, right, bottom = bbox
            face = frame_img[top:bottom, left:right]
            cached_dir.mkdir(exist_ok=True)

            np.save(cached_file, face)

        image = Image.fromarray(face)
        image = self.transform(image)
        return image, cached_file

    def __getitem__(self, idx: int):
        real_fn = self.real_filename[idx]
        fake_fn = np.random.choice(self.real2fakes[real_fn])

        try:
            if real_fn not in self.bbox_index_fn:
                settings.un_normal_list.append(real_fn)
                raise IOError("real_fn not found")
            frames = self.bbox_df.xs(real_fn, level=0).index.get_level_values(0)
            frame = np.random.choice(frames)

            real_image, real_file = self.__get_transformed_face(real_fn, frame)

            if fake_fn in self.bbox_index_fn and self.bbox_df.index.isin([(fake_fn, frame, 0)]).all():
                print(f"getting fake {real_fn}, {frame}")
                fake_image, fake_file = self.__get_transformed_face(fake_fn, frame)
            else:
                bbox = np.fromstring(self.bbox_df.at[(real_fn, frame, 0), 'bbox'][1:-1], sep=' ').astype(int)
                fake_image, fake_file = self.__get_transformed_face(fake_fn, frame, bbox=bbox)

            return {'real': real_image, 'fake': fake_image, 'real_file': real_file, 'fake_file': fake_file}

        except IOError as e:
            print(e)
            return {'real': None, 'fake': None, 'real_file': None, 'fake_file': None}


def get_transforms(level=0, image_size=224):
    # mean and std  RGB， 3 levels of augmentation
    # mean: [0.4386408842443215, 0.3283582082051558, 0.30372247414590847]
    # mean: [0.05387861529517991, 0.04259260701438289, 0.04022694313550189]
    pre_trained_mean, pre_trained_std = [0.439, 0.328, 0.304], [0.232, 0.206, 0.201]
    tensor_transform = transforms.Compose([])

    if level == 0:
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
        ])
    elif level == 1:
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=20, scale=(.9, 1.1), shear=0),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])
        tensor_transform = transforms.RandomErasing(scale=(0.02, 0.15), ratio=(0.3, 1.6))
    elif level == 2:
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=40, scale=(.9, 1.1), shear=0),
            transforms.RandomPerspective(distortion_scale=0.2),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        ])
        tensor_transform = transforms.RandomErasing(scale=(0.02, 0.3), ratio=(0.3, 3.3))
    elif level == 3:
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=40, scale=(.9, 1.1), shear=0),
            transforms.RandomPerspective(distortion_scale=0.2),
            transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6),
        ])
        tensor_transform = transforms.RandomErasing(scale=(0.1, 0.5), ratio=(0.2, 5))
    elif level == 4:
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.Resize(512),
            transforms.RandomCrop(256, padding_mode='constant'),
        ])
    elif level == 5:
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(384),
            transforms.RandomCrop(128, padding_mode='constant'),
        ])
    else:
        train_transforms = transforms.transforms.Compose([])

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
    train_transforms, val_transforms = get_transforms(image_size=224, level=params['augment_level'])
    metadata = pd.read_json(params['metadata_path']).T
    bbox = pd.read_csv(params['bbox_path'], index_col=[0, 1, 2])
    train_dl = _create_dataloader(metadata[metadata['split_kailu'] == 'train'], bbox, params, train_transforms,
                                  train_data_filter, batch_size=params['batch_size'], shuffle=True)
    val_dl = _create_dataloader(metadata[metadata['split_kailu'] == 'validation'], bbox, params, val_transforms,
                                val_data_filter, batch_size=params['batch_size'], shuffle=False)
    test_dl = _create_dataloader(metadata[metadata['split_kailu'] == 'test'], bbox, params, val_transforms,
                                 val_data_filter, batch_size=params['batch_size'], shuffle=False)

    return train_dl, val_dl, test_dl, iter(val_dl)


def _create_dataloader(metadata: DataFrame, bbox: DataFrame, params: dict, transform, data_filter,
                       batch_size=64, num_workers=0, shuffle=True):
    assert len(metadata) != 0, f'metadata are empty'

    ds = DFDCDataset(metadata=metadata, bbox=bbox, video_path=params['data_path'], cached_path=params['cache_path'],
                     transform=transform, data_filter=data_filter)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=collate_fn)

    print(f"data: {len(ds)}")
    return dl
