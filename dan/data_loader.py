from torchvision import datasets, transforms
import torch
from torch.utils import data
import os
import tqdm
import pandas as pd
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dir):
        self.dir = dir
        features, labels = self.read_feature(dir)
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def read_feature(dir):
        features = []
        labels = []

        filenames = os.listdir(dir)
        for filename in tqdm.tqdm(filenames, desc='loading data from %s...' % dir):
            df = pd.read_csv(os.path.join(dir, filename))
            features.append(df.values)
            category = os.path.splitext(filename)[0].split('_')[-1]
            if category == 'depr':
                labels.append(0)
            else:
                labels.append(1)

        return features, labels


def collate_fn(batch):
    batch.sort(key=lambda data: len(data[0]), reverse=True)
    features = [torch.tensor(data[0], dtype=torch.float32, device='cuda') for data in batch]
    labels = [data[1] for data in batch]
    padded_feature = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
    padded_feature = padded_feature.unsqueeze(1)
    return padded_feature, torch.tensor(labels, dtype=torch.long, device='cuda')


def loader(dir, batch_size, drop_last=True, num_workers=0):
    dataset = Dataset(dir)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             drop_last=drop_last,
                                             collate_fn=collate_fn,
                                             num_workers=num_workers)

    return dataloader