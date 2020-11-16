import os
from PIL import Image
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

def load_mat(file):
	depth_imgs = os.listdir(file)
	depth_imgs.sort()
	for i in depth_imgs:
		depth = loadmat(path + str(i))
		depth = depth['Position3DGrid']
		depth = depth.transpose(2, 0, 1)
		depth = depth[3]
		depth_img = Image.fromarray(depth)
		depth_img = depth_img.resize((345,460))
		'''
		depth_np = np.array(depth_img)
		cmap = 'rainbow'
		plt.imshow(depth_np, cmap=plt.get_cmap(cmap))
		plt.show()
		'''

def read_kitti_depth(file):
	depth = Image.open(file)
	depth_np = np.array(depth)
	pass


if __name__ == '__main__':
	path = '/data/make3d/Test134Depth/'
	kitti_path = '/data/kitti/official/train/2011_10_03_drive_0034_sync/proj_depth/' \
				 'groundtruth/image_02/0000000978.png'

	read_kitti_depth(kitti_path)