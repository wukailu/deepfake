import pathlib
import numpy as np
import os
import pandas as pd
from pandas import DataFrame

import dlib
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

from utils import preprocess_image


def collate_fn(batch: list):
    ret = {}
    for key in batch[0].keys():
        ret[key] = [item[key] for item in batch]
    ret['real'] = torch.stack(ret['real'])
    ret['fake'] = torch.stack(ret['fake'])
    return ret


def train_data_filter(metadata_df):
    return metadata_df[metadata_df['label'] == 'REAL' and metadata_df['person_num'] == 1]


def val_data_filter(metadata_df):
    return metadata_df[metadata_df['label'] == 'REAL']


class DFDCDataset(Dataset):
    def __init__(self, metadata: DataFrame, bbox: DataFrame, video_path: str, cached_path: str, transform,
                 data_filter, output_image_size=224):
        self.metadata_df = metadata
        self.real_filename = list(data_filter(metadata).index)
        self.bbox_df = bbox
        self.transform = transform
        self.image_size = output_image_size
        self.video_path = pathlib.Path(video_path)

        self.cached_path = pathlib.Path(cached_path)
        self.cached_path.mkdir(exist_ok=True)
        self.real2fakes = {fn: [] for fn in self.real_filename}
        for fn, row in metadata.iterrows():
            if row['label'] == 'FAKE':
                self.real2fakes[row['original']].append(fn)

    def __len__(self):
        return len(self.real_filename)

    def __get_transformed_face(self, filename: str, frame: int) -> (torch.Tensor, pathlib.Path):
        cached_file = self.cached_path / filename / (str(frame) + '.npy')

        if cached_file.is_file():
            face = np.load(cached_file)
        else:
            cap = cv2.VideoCapture(os.path.join(self.video_path, filename))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)  # 设置要获取的帧号
            _, frame_img = cap.read()
            if not _:
                print("Reading " + filename + " Failed")
            cap.release()

            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            left, top, right, bottom = self.bbox_df.at[(filename, frame, 0), 'bbox'].astype(int)
            face = frame_img[top:bottom, left:right]
            np.save(cached_file, face)

        image = Image.fromarray(face)
        image = self.transform(image)
        return image, cached_file

    def __getitem__(self, idx: int):
        real_fn = self.real_filename[idx]
        fake_fn = np.random.choice(self.real2fakes[real_fn])
        frames = self.bbox_df.xs(real_fn, level=0).index.get_level_values(1)
        if len(frames) == 0:
            print("Can not find face in " + real_fn + "!!!")
        frame = np.random.choice(frames)

        real_image, real_file = self.__get_transformed_face(real_fn, frame)
        fake_image, fake_file = self.__get_transformed_face(fake_fn, frame)

        return {'real': real_image, 'fake': fake_image, 'real_file': real_file, 'fake_file': fake_file}


def get_transforms():
    # TODO: re-calc mean and std
    pre_trained_mean, pre_trained_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # transforms.RandomAffine(degrees=40, scale=(.9, 1.1), shear=0),
        # transforms.RandomPerspective(distortion_scale=0.2),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        # transforms.RandomErasing(scale=(0.02, 0.16), ratio=(0.3, 1.6)),
        transforms.ToTensor(),
        transforms.Normalize(mean=pre_trained_mean, std=pre_trained_std),
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=pre_trained_mean, std=pre_trained_std)
    ])
    return train_transforms, val_transforms


def create_dataloaders(params: dict):
    train_transforms, val_transforms = get_transforms()
    metadata = pd.read_json(params['metadata_path']).T
    bbox = pd.read_json(params['bbox_path'])
    train_dl = _create_dataloader(metadata[metadata['split_kailu'] == 'train'], bbox, params, train_transforms, train_data_filter, batch_size=params['batch_size'])
    val_dl = _create_dataloader(metadata[metadata['split_kailu'] == 'validation'], bbox, params, val_transforms, val_data_filter, batch_size=params['batch_size'])
    test_dl = _create_dataloader(metadata[metadata['split_kailu'] == 'test'], bbox, params, val_transforms, val_data_filter, batch_size=params['batch_size'])
    # display_file_paths = [f'/datasets/{i}_deepfake/val' for i in ['base', 'augment']]
    # display_dl_iter = iter(
    #     _create_dataloader(display_file_paths, mode='val', batch_size=32, transformations=val_transforms,
    #                        sample_ratio=params['sample_ratio']))

    return train_dl, val_dl, test_dl, iter(val_dl)


def _create_dataloader(metadata: DataFrame, bbox: DataFrame, params: dict, transform, data_filter,
                       output_image_size: int = 224, batch_size=64, num_workers=64):
    assert len(metadata) != 0, f'metadata are empty'

    ds = DFDCDataset(metadata=metadata, bbox=bbox, video_path=params['data_path'], cached_path=params['cache_path'],
                     transform=transform, data_filter=data_filter, output_image_size=output_image_size)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=collate_fn)

    print(f"data: {len(ds)}")
    return dl


