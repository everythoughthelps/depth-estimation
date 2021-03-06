import argparse

import time
import visdom
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import loaddata
import util
import numpy as np
from torchvision.transforms import ToPILImage
from models import modules, net, resnet, densenet, senet, resnext, unet_model
import os

parser = argparse.ArgumentParser(description='PyTorch DABC Training')
parser.add_argument('--experiment', default='./experiments', type=str, help='path of experiments')
parser.add_argument('--nyu_data', default='/data/', type=str, help='path of nyu dataset')
parser.add_argument('--kitti_data', default='/data/kitti/', type=str, help='path of kitti dataset')
parser.add_argument('--num_classes', default=120, type=int, help='number of depth classes')
parser.add_argument('--net_arch', default='resnext_64x4d', type=str, help='architecture of feature extraction')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--data_sample_interval', default=3, help='how many imgs samples one img each')
parser.add_argument('--discrete_strategy', default='linear', help='')
parser.add_argument('--rebuild_strategy', default='max', help='')
parser.add_argument('--label_smooth', default='True', help='')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--load', default=False)


def define_model():
	if args.net_arch == 'resnext_64x4d':
		clz_model = resnext.resnext(groups=64, width_per_group=4)
		fea_model = modules.FeatureResnext(clz_model.features)
		model = modules.DABC(fea_model, args.num_classes)
	elif args.net_arch == 'resnet':
		pass
	elif args.net_arch == 'densenet':
		clz_model = densenet.densenet161(pretrained=True,)
		fea_model = modules.FeatureResnext(clz_model.features)
		model = modules.DABC(fea_model, args.num_classes)
	elif args.net_arch == 'senet':
		pass
	else:
		raise NotImplementedError("Network Architecture [%s] is not recognized." % args.net_arch)

	return model


vis = visdom.Visdom(env='depth estimation')
vis.line([[0., 0.]], [0], win='train', opts=dict(title='loss&acc', legend=['loss', 'rmse']))


def main():
	global args
	args = parser.parse_args()
	model = define_model()
	if args.load:
		model.load_state_dict(torch.load(args.load)['state_dict'])
		print('Model loaded from {}'.format(args.load))

	if torch.cuda.is_available():
		model = model.cuda()
		device_num = torch.cuda.device_count()
		if device_num > 1:
			device_ids = [x for x in range(device_num)]
			model = torch.nn.DataParallel(model, device_ids=device_ids)
			args.batch_size *= device_num

	cudnn.benchmark = True
	optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
	print(args)
	'''
	if 'kitti' in args.data:
		parser.add_argument('--image_size', default=[[640,192], [320,96]], help='')
		parser.add_argument('--e', default=0.25, type=float, help='avoid log0')
		parser.add_argument('--range', default=80, type=int)
		args = parser.parse_args()
		train_loader = loaddata.get_kitti_train_data(args)
		test_loader = loaddata.get_kitti_test_data(args)
		print('kitti')
	'''

	parser.add_argument('--image_size', default=[[304, 228], [152,114]], help='')
	parser.add_argument('--e', default=0.01, type=float, help='avoid log0')
	parser.add_argument('--range', default=10, type=int)
	args = parser.parse_args()
	train_loader = loaddata.getTrainingData(args)
	test_loader = loaddata.getTestingData(args)
	print('nyu')

	setup_logging()

	for epoch in range(args.start_epoch, args.epochs):
		adjust_learning_rate(optimizer, epoch)
		loss = train(train_loader, model, optimizer, epoch)
		rmse = test(test_loader, model, epoch)
		vis.line([[np.array(loss)], [np.array(rmse)]], [np.array(epoch)], win='train', update='append')

		save_checkpoint({'state_dict': model.state_dict()},
						filename=os.path.join(args.ckp_path, '%02dcheckpoint.pth.tar' % epoch))


def train(train_loader, model, optimizer, epoch):
	criterion_clz = nn.CrossEntropyLoss()
	# criterion_depth = nn.L1Loss()
	batch_time = AverageMeter()
	losses = AverageMeter()
	smooth_loss_L1= util.smooth_loss()
	model.train()

	end = time.time()
	for i, sample_batched in enumerate(train_loader):
		image, depth, label = sample_batched['image'], sample_batched['depth'], sample_batched['label']
		continue
		print('image', image.size())
		print('depth', depth.size())
		print('label', label.size())

		image = image.cuda()
		# depth = depth.cuda()
		label = label.cuda().long()

		optimizer.zero_grad()

		output = model(image)
		print(output.size())
		_,index = output.max(1)
		loss_smooth = smooth_loss_L1(index)
		loss_clz = criterion_clz(output, label)
		# loss_depth = criterion_depth(soft_sum(output), depth)

		loss = loss_clz + loss_smooth

		losses.update(loss.item(), image.size(0))
		loss.backward()
		optimizer.step()

		batch_time.update(time.time() - end)

		end = time.time()

		print('Epoch: [{0}][{1}/{2}]\t'
			  'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
			  'Loss {loss.val:.4f} ({loss.avg:.4f})'
			  .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses))

		with open(os.path.join(args.save_path, 'records_batch.csv'), 'a') as f:
			f.write('%d,%d/%d,%f,%f,%f,%f\n' % (
				epoch, i, len(train_loader), batch_time.val, batch_time.sum, losses.val, losses.avg))
		break

	with open(os.path.join(args.save_path, 'records_epoch.csv'), 'a') as f:
		f.write('%d,%f\n' % (epoch, losses.avg))
	return losses.avg


def test(test_loader, model, epoch):
	model.eval()

	totalNumber = 0

	errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
				'MAE': 0, 'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

	for i, sample_batched in enumerate(test_loader):
		image, depth, label, image_name = sample_batched['image'], sample_batched['depth'], sample_batched['label'], \
										  sample_batched['image_name']

		image = image.cuda()
		depth = depth.cuda()
		print(image_name)
		# label = label.cuda()

		output = F.softmax(model(image))
		if args.rebuild_strategy == 'soft_sum':
			depth_pred = soft_sum(output,args.discrete_strategy)
		if args.rebuild_strategy == 'max':
			depth_pred = max(output,args.discrete_strategy)
		depth_pred = F.interpolate(depth_pred.float(), size=[depth.size(2), depth.size(3)], mode='bilinear')
		t = depth_pred.squeeze().float().cpu() / args.range
		print(t.size())
		results_imgs = ToPILImage()(depth_pred.squeeze().float().cpu() / args.range)
		if not os.path.exists(str(args.img_path) + '/' + str(epoch) + 'epochs_results/'):
			os.mkdir(str(args.img_path) + '/' + str(epoch) + 'epochs_results/')
		results_imgs.save(str(args.img_path) + '/' + str(epoch) + 'epochs_results/' +
						  str(image_name).strip(str([''])))

		batchSize = depth.size(0)
		totalNumber = totalNumber + batchSize
		errors = util.evaluateError(depth_pred, depth)
		errorSum = util.addErrors(errorSum, errors, batchSize)
		averageError = util.averageErrors(errorSum, totalNumber)
	averageError['RMSE'] = np.sqrt(averageError['MSE'])
	print('epoch %d testing' % epoch)
	print(averageError)

	with open(os.path.join(args.save_path, 'records_val.csv'), 'a') as f:
		f.write('%d,%f,%f,%f,%f,%f,%f,%f,%f\n' %
				(epoch,
				 averageError['MSE'],
				 averageError['RMSE'],
				 averageError['ABS_REL'],
				 averageError['LG10'],
				 averageError['MAE'],
				 averageError['DELTA1'],
				 averageError['DELTA2'],
				 averageError['DELTA3']))
	return averageError['RMSE']


def soft_sum(probs,rebuild):
	depth_value = 0
	ones = torch.ones(probs.size()).float().cuda()
	unit = torch.arange(0, args.num_classes).view(1, probs.size(1), 1, 1).float()
	weight = unit
	for _ in range(probs.size(0) - 1):
		weight = torch.cat((weight, unit), dim=0)
	weight = ones * weight.cuda()
	if rebuild == 'log':
		q = (np.log10(args.range + args.e) - np.log10(args.e)) / (args.num_classes - 1)
		weight = weight * q + np.log10(args.e)
		depth_value = 10 ** (torch.sum(weight * probs, dim=1)) - args.e
		depth_value = torch.unsqueeze(depth_value, dim=1)
	if rebuild == 'linear':
		depth_value = 10 / args.num_classes *(torch.sum(weight * probs, dim = 1))
		depth_value = torch.unsqueeze(depth_value, dim=1)
	return depth_value


def max(probs,rebuild):
	depth_value = 0
	q = (np.log10(args.range + args.e) - np.log10(args.e)) / (args.num_classes - 1)
	_, label = probs.max(dim=1)
	label = label.float()
	if rebuild == 'log':
		lgdepth = label * q + np.log10(args.e)
		depth_value = 10 ** (lgdepth) - args.e
		depth_value = torch.unsqueeze(depth_value, dim=1)
	if rebuild == 'linear':
		depth_value = label * 10 / args.num_classes
		depth_value = torch.unsqueeze(depth_value, dim=1)
	return depth_value


def adjust_learning_rate(optimizer, epoch):
	lr = args.lr * (0.5 ** (epoch // 5))
	# lr = args.lr / 10 if epoch > 30 else args.lr

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def save_checkpoint(state, filename='checkpoint.pth.tar'):
	torch.save(state, filename)


def setup_logging():
	save_path = os.path.join(args.experiment, time.strftime("%Y_%m_%d_%H_%M_%S"))
	os.makedirs(save_path)
	ckp_path = os.path.join(save_path, 'ckp')
	img_path = os.path.join(save_path, 'img')
	os.mkdir(img_path)
	os.mkdir(ckp_path)
	args.img_path = img_path
	args.ckp_path = ckp_path
	args.save_path = save_path
	with open(os.path.join(save_path, 'records_batch.csv'), 'w') as f:
		f.write('Epoch,Batch,Time,Time_sum,Loss,Loss_avg\n')

	with open(os.path.join(save_path, 'args'), 'w') as f:
		f.write(str(args))

	with open(os.path.join(save_path, 'records_epoch.csv'), 'w') as f:
		f.write('Epoch,Loss\n')

	with open(os.path.join(save_path, 'records_val.csv'), 'w') as f:
		f.write('Epoch,MSE,RMSE,ABS_REL,LG10,MAE,DELTA1,DELTA2,DELTA3\n')


if __name__ == '__main__':
	main()
