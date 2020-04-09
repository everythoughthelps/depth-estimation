import os

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
from nyu_transform import *


class depthDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, args, csv_file, transform=None):
        self.data_path = args.data
        self.frame = pd.read_csv(os.path.join(self.data_path, csv_file), header=None)
        self.transform = transform
        self.e = args.e
        self.q = (np.log10(10) - np.log10(self.e)) / (args.num_classes-1)

    def __getitem__(self, idx):
        image_name = self.frame.at[idx, 0]
        depth_name = self.frame.at[idx, 1]
        print(image_name,depth_name,idx)

        image = Image.open(os.path.join(self.data_path, image_name))
        depth = Image.open(os.path.join(self.data_path, depth_name))

        sample = {'image': image, 'depth': depth }
        if self.transform:
            sample = self.transform(sample)

        sample['depth'] = sample['depth']

        sample['label'] = torch.squeeze(
            torch.floor((torch.log10(sample['depth'] + self.e) - np.log10(self.e)) / self.q).long(), dim=0)
        sample['image_name'] = image_name
        return sample

    def __len__(self):
        return len(self.frame)


def getTrainingData(args):
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    transformed_training = depthDataset(args,
                                        csv_file='data/nyu2_train.csv',
                                        transform=transforms.Compose([
                                            Scale(240),
                                            RandomHorizontalFlip(),
                                            RandomRotate(5),
                                            CenterCrop([304, 228], [152, 114]),
                                            ToTensor(),
                                            # Lighting(0.1, __imagenet_pca[
                                            #     'eigval'], __imagenet_pca['eigvec']),
                                            # ColorJitter(
                                            #     brightness=0.4,
                                            #     contrast=0.4,
                                            #     saturation=0.4,
                                            # ),
                                            # Normalize(__imagenet_stats['mean'],
                                            #           __imagenet_stats['std'])
                                        ]))

    dataloader_training = DataLoader(transformed_training, args.batch_size,
                                     shuffle=True, num_workers=4, pin_memory=False)

    return dataloader_training


def getTestingData(args):
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    # scale = random.uniform(1, 1.5)
    transformed_testing = depthDataset(args,
                                       csv_file='data/nyu2_test.csv',
                                       transform=transforms.Compose([
                                           Scale(240),
                                           CenterCrop([304, 228], [304, 228]),
                                           ToTensor(is_test=True),
                                           # Normalize(__imagenet_stats['mean'],
                                           #           __imagenet_stats['std'])
                                       ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size=1,
                                    shuffle=False, num_workers=0, pin_memory=False)

    return dataloader_testing
