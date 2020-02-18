import pathlib
import numpy as np
import pandas as pd
from pandas import DataFrame

import cv2, os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import random
from torch import Tensor

training_mean, training_std = [0.465, 0.428, 0.415], [0.269, 0.275, 0.2755]


def blur(ret: Tensor):
    return (ret*9 + ret.roll(1, 1)*4 + ret.roll(-1, 1)*4 + ret.roll(1, 2)*4 + ret.roll(-1, 2)*4) / 25


def gen_mask(f1: Tensor, f2: Tensor, threhold=0.125):  # for batch
    assert f1.shape == f2.shape and len(f1.shape) == 4
    ret, _ = (f1 - f2).abs().max(dim=1)

    ret = blur(ret)
    ret = blur(ret)
    ret = blur(ret)

    ret[ret < threhold] = 0
    ret[ret > 0] = 1
    return ret.long()


def collate_fn(batch: list):
    ret = {}
    for key in batch[0].keys():
        ret[key] = [item[key] for item in batch if item[key] is not None]
    try:
        ret['real'] = torch.stack(ret['real'])
        ret['fake'] = torch.stack(ret['fake'])
    except:
        print(">>>>>>>>>>>>>>>>>>len(real) == 0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        ret['real'] = torch.zeros((1, 3, 224, 224))
        ret['fake'] = torch.zeros((1, 3, 224, 224))

    inputs = ret['fake']
    labels = gen_mask(ret['real'], ret['fake'])
    return inputs, labels, ret


class DFDCDataset(Dataset):
    def __init__(self, metadata: DataFrame, params: dict, transform):
        self.metadata_df = metadata
        self.real_filename = list(metadata.index)
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
            with open(cached_file, "rb") as fp:
                pil_img = Image.open(fp)
                pil_img.load()
        else:
            raise IOError("cache not found")
        if pil_img.size != (960, 540):
            raise IOError("shape not match")
        image = self.transform(pil_img)
        return image, cached_file

    def __getitem__(self, idx: int):
        real_fn = self.real_filename[idx]
        fake_fn = np.random.choice(self.real2fakes[real_fn])

        _, _, files = os.walk(self.cached_path / real_fn.split(".")[0])
        frames = [int(file.split(".")[0]) for file in files]
        try:
            frame = np.random.choice(frames)

            temp_seed = np.random.randint(0, 2e9)
            if self.same_transform:
                random.seed(temp_seed)
            real_image, real_file = self.__get_transformed_face(real_fn, frame)
            if self.same_transform:
                random.seed(temp_seed)
            fake_image, fake_file = self.__get_transformed_face(fake_fn, frame)

            return {'real': real_image, 'fake': fake_image, 'real_file': real_file, 'fake_file': fake_file}
        except IOError as e:
            return {'real': None, 'fake': None, 'real_file': None, 'fake_file': None}


def get_transforms(params, image_size=224):
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

    train_transforms = transforms.Compose(train_transforms)
    tensor_transform = transforms.Compose(tensor_transform)

    train_transforms = transforms.Compose([
        train_transforms,
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(training_mean, training_std),
        tensor_transform,
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(training_mean, training_std)
    ])
    return train_transforms, val_transforms


def create_dataloaders(params: dict):
    train_transforms, val_transforms = get_transforms(params, image_size=224)
    metadata = pd.read_json(params['metadata_path']).T
    loader = 8
    train_dl = _create_dataloader(metadata[metadata['split_kailu'] == 'train'], params, train_transforms,
                                  shuffle=True, num_workers=loader)
    val_dl = _create_dataloader(metadata[metadata['split_kailu'] == 'validation'], params, val_transforms,
                                shuffle=False, num_workers=loader)
    test_dl = _create_dataloader(metadata[metadata['split_kailu'] == 'test'], params, val_transforms,
                                 shuffle=False, num_workers=loader)

    return train_dl, val_dl, test_dl


def _create_dataloader(metadata: DataFrame, params: dict, transform, num_workers=4, shuffle=True):
    ds = DFDCDataset(metadata=metadata, params=params, transform=transform)
    dl = DataLoader(ds, batch_size=params['batch_size'], num_workers=num_workers, shuffle=shuffle,
                    collate_fn=collate_fn, drop_last=True)

    print(f"data: {len(ds)}")
    return dl
