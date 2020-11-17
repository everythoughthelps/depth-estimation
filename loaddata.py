import os

import pandas as pd
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

from nyu_transform import *
from kitti_transform import kitti_ToTensor


class PairBatchSampler(Sampler):
	def __init__(self, dataset, batch_size, num_iterations=None):
		self.dataset = dataset
		self.batch_size = batch_size
		self.num_iterations = num_iterations

	def __iter__(self):
		indices = list(range(len(self.dataset)))
		random.shuffle(indices)
		for k in range(len(self)):
			if self.num_iterations is None:
				offset = k*self.batch_size
				batch_indices = indices[offset:offset+self.batch_size]
			else:
				batch_indices = random.sample(range(len(self.dataset)),
											  self.batch_size)

			pair_indices = []
			for idx in batch_indices:
				data_class = self.dataset.get_class(idx)
				pair_index = random.choice(indices)
				while data_class not in self.dataset.stack_frame.at[pair_index,0]:
					pair_index= random.choice(indices)
				pair_indices.append(pair_index)

			yield batch_indices + pair_indices

	def __len__(self):
		if self.num_iterations is None:
			return (len(self.dataset)+self.batch_size-1) // self.batch_size
		else:
			return self.num_iterations

class depthDataset(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, args, nyu_csv_file, kitti_txt_file, transform=None):
		self.data_path = args.nyu_data
		self.kitti_data_path = args.kitti_data
		self.nyu_frame = pd.read_csv(nyu_csv_file, header=None)
		self.kitti_frame = pd.read_fwf(kitti_txt_file, header=None)
		self.stack_frame = pd.concat([self.kitti_frame,self.nyu_frame],axis=0,ignore_index=True)
		self.transform = transform
		self.e = args.e
		self.q = (np.log10(10 + self.e) - np.log10(self.e)) / (args.num_classes - 1)
		self.discrete_strategy = args.discrete_strategy
		self.classes = args.num_classes

	def __getitem__(self, idx):
		print(idx)
		image_name = self.stack_frame.at[idx, 0]
		depth_name = self.stack_frame.at[idx, 1]
		print(image_name, depth_name, idx)

		if 'kitti' in image_name:
			image = Image.open(os.path.join(self.data_path, 'kitti/',image_name))
			depth = Image.open(os.path.join(self.data_path, 'kitti/',depth_name))

		else:
			image = Image.open(os.path.join(self.data_path, 'nyuv2/',image_name))
			depth = Image.open(os.path.join(self.data_path, 'nyuv2/',depth_name))

		sample = {'image': image, 'depth': depth}
		if self.transform:
			sample = self.transform(sample)

		sample['depth'] = sample['depth']
		if self.discrete_strategy == 'log':
			sample['label'] = torch.squeeze(
				torch.round((torch.log10(sample['depth'] + self.e) - np.log10(self.e)) / self.q).long(), dim=0)
		if self.discrete_strategy == 'linear':
			sample['label'] = torch.squeeze(sample['depth'] / (10 / self.classes), dim=0)
		sample['image_name'] = image_name[15:]
		return sample

	def __len__(self):
		return len(self.stack_frame)

	def get_class(self, id):
		data_name = self.stack_frame.at[id,0]
		data_class = 'kitti' if 'kitti' in data_name else 'nyu'
		return data_class

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
										nyu_csv_file='./data/' + str(args.data_sample_interval) + 'nyu2_train.csv',
										kitti_txt_file='./data/eigen_train_pairs.txt',
										transform=transforms.Compose([
											Scale(240),
											RandomHorizontalFlip(),
											RandomRotate(5),
											CenterCrop(args.image_size[0], args.image_size[1]),
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

	dataloader_training = DataLoader(transformed_training,batch_sampler=PairBatchSampler(transformed_training,args.batch_size),
									 num_workers=4, pin_memory=False)

	return dataloader_training


def getTestingData(args):
	__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
						'std': [0.229, 0.224, 0.225]}
	# scale = random.uniform(1, 1.5)
	transformed_testing = depthDataset(args,
									   nyu_csv_file='./data/nyu2_test.csv',
									   kitti_txt_file='./data/eigen_train_pairs.txt',
									   transform=transforms.Compose([
										   Scale(240),
										   CenterCrop(args.image_size[0], args.image_size[0]),
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
		self.img_path = self.data_path + 'data/' + self.mode + '/' + 'X/'
		self.depth_path = self.data_path + 'data/' + self.mode + '/' + 'y/'
		self.transform = transform
		self.e = args.e
		self.q = (np.log10(80 + self.e) - np.log10(self.e)) / (args.num_classes - 1)
		self.discrete_strategy = args.discrete_strategy
		self.classes = args.num_classes

	def __getitem__(self, item):
		img_list = os.listdir(self.img_path)
		depth_list = os.listdir(self.depth_path)
		image = Image.open(self.img_path + img_list[item])
		depth = Image.open(self.depth_path + depth_list[item]).convert('L')
		image_name = img_list[item]
		sample = {'image': image, 'depth': depth}

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
											 CenterCrop(args.image_size[0], args.image_size[1]),
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
											CenterCrop(args.image_size[0], args.image_size[0]),
											kitti_ToTensor()
											# Normalize(__imagenet_stats['mean'],
											#           __imagenet_stats['std'])
										]))

	dataloader_testing = DataLoader(transformed_testing, batch_size=1,
									shuffle=False, num_workers=0, pin_memory=False)

	return dataloader_testing
