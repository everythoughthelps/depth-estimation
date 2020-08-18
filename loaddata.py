import os
import cv2 as cv

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from nyu_transform import *
from kitti_transform import kitti_ToTensor


class depthDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, args, csv_file, transform=None):
        self.data_path = args.data
        self.frame = pd.read_csv(os.path.join(self.data_path, csv_file), header=None)
        self.transform = transform
        self.e = args.e
        self.q = (np.log10(10+self.e) - np.log10(self.e)) / (args.num_classes-1)
        self.discrete_strategy = args.discrete_strategy
        self.classes = args.num_classes


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
        if self.discrete_strategy == 'log':
            sample['label'] = torch.squeeze(
                torch.round((torch.log10(sample['depth'] + self.e) - np.log10(self.e)) / self.q).long(), dim=0)
        if self.discrete_strategy == 'linear':
            sample['label'] = torch.squeeze(sample['depth'] // (256/self.classes),dim=0)
        sample['image_name'] = image_name[11:]
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
                                        csv_file='train_labeled.csv',
                                        transform=transforms.Compose([
                                            Scale(240),
                                            RandomHorizontalFlip(),
                                            RandomRotate(5),
                                            CenterCrop(args.image_size[0],args.image_size[1]),
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
                                       csv_file='val_labeled.csv',
                                       transform=transforms.Compose([
                                           Scale(240),
                                           CenterCrop(args.image_size[0],args.image_size[0]),
                                           ToTensor(is_test=True),
                                           # Normalize(__imagenet_stats['mean'],
                                           #           __imagenet_stats['std'])
                                       ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size=1,
                                    shuffle=False, num_workers=0, pin_memory=False)

    return dataloader_testing

class kitti_dataset(Dataset):
    def __init__(self, args, mode, transform=None):
        self.data_path = args.data
        self.mode = mode
        self.img_path =  self.data_path + 'data/' + self.mode + '/' + 'X/'
        self.depth_path = self.data_path + 'data/' + self.mode + '/' +'y/'
        self.transform = transform
        self.e = args.e
        self.q = (np.log10(80+self.e) - np.log10(self.e)) / (args.num_classes-1)
        self.discrete_strategy = args.discrete_strategy
        self.classes = args.num_classes

    def __getitem__(self, item):
        img_list= os.listdir(self.img_path)
        depth_list= os.listdir(self.depth_path)
        image = Image.open(self.img_path + img_list[item])
        depth = Image.open(self.depth_path + depth_list[item]).convert('L')
        image_name = img_list[item]
        sample = {'image':image,'depth':depth}

        if self.transform:
            sample = self.transform(sample)

        sample['depth'] = sample['depth']
        if self.discrete_strategy == 'log':
            sample['label'] = torch.squeeze(
                torch.round((torch.log10(sample['depth'] + self.e) - np.log10(self.e)) / self.q).long(), dim=0)
        if self.discrete_strategy == 'linear':
            sample['label'] = torch.squeeze(sample['depth'] // (256 / self.classes), dim=0)
        sample['image_name'] = image_name
        return sample

    def __len__(self):
        return len(os.listdir(self.img_path))

def get_kitti_train_data(args):
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

    transformed_training = kitti_dataset(args,
										 'train',
                                        transform=transforms.Compose([
                                            RandomHorizontalFlip(),
                                            RandomRotate(5),
                                            CenterCrop(args.image_size[0],args.image_size[1]),
											kitti_ToTensor()
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


def get_kitti_test_data(args):
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    # scale = random.uniform(1, 1.5)
    transformed_testing = kitti_dataset(args,
                                        'val',
                                       transform=transforms.Compose([
                                           CenterCrop(args.image_size[0],args.image_size[0]),
                                           kitti_ToTensor()
                                           # Normalize(__imagenet_stats['mean'],
                                           #           __imagenet_stats['std'])
                                       ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size=1,
                                    shuffle=False, num_workers=0, pin_memory=False)

    return dataloader_testing
