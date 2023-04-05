import tensorflow as tf
import numpy as np
import scipy.io as sio
from six.moves import xrange
import time
import h5py
import pickle
import argparse
from utils import *
import model as sdm_net
import os
from config import *
import icp
import random


# vae
timecurrent = time.strftime('%m%d%H%M', time.localtime(time.time())) + '_' + str(random.randint(1000,9999))
test = False
parser = argparse.ArgumentParser()

parser.add_argument('--model', default = 'car', type = str) # training model name
parser.add_argument('--l0', default = 1.0, type = float) # 1.0  generation loss
# parser.add_argument('--l1', default = 0.5, type = float) # 0.5  distance loss
parser.add_argument('--l2', default = 1.0, type = float) # 0.5 	weight loss
parser.add_argument('--l3', default = 1.0, type = float) # 0.005 Kl loss
parser.add_argument('--l4', default = 0.001, type = float) # 0.005 l2 loss
# parser.add_argument('--l5', default = 0.2, type = float) # 0.005 region loss
parser.add_argument('--joint', default = 0, type = int) # jointly training
parser.add_argument('--trcet', default = 0.75, type = float) # training percent of the dataset

# --------------------------not include in file name-------------------------------------
parser.add_argument('--lz_d', nargs='+', default = [64, 128], type=int) # dimension of latent space
parser.add_argument('--layer', default = 2, type = int) # the number of mesh convolution layer
parser.add_argument('--activ', default = '', type = str) # activate function of structure net : every layer
parser.add_argument('--lr', default = 0.001, type = float) # learning rate need adjust if the loss is too large in the early training
parser.add_argument('--batch_size', default = 16, type = int)
parser.add_argument('--iddatfile', default = 'id.dat', type = str)
parser.add_argument('--autoencoder', default='vae', type = str)
parser.add_argument('--finaldim', default = 9, type = int)
parser.add_argument('--filename', default = 'car.obj', type = str)
parser.add_argument('--featurefile', default = 'car_vaefeature.mat', type = str)
parser.add_argument('--controlid', nargs='+', default = [0], type=int)
parser.add_argument('--epoch_deform', default=20000, type = int)
parser.add_argument('--epoch_structure', default=40200, type = int)
parser.add_argument('--output_dir', default='./result'+timecurrent, type = str)

args = parser.parse_args()

if not args.autoencoder.find('tan') == -1:
	args.l3 = 0.00

args.featurefile = args.model + '_vaefeature.mat'
args.filename = args.model + '.obj'
args.iddatfile = args.model + '_vaeid.dat'
theList = ['activ','lr','layer','lz_d']
if not args.output_dir.find('./result'+timecurrent) == -1:
	a = './' + timecurrent + "-".join(["{}_{}".format(k,args.__dict__[k]) for k in sorted(args.__dict__.keys()) if len(k) < 6 and k not in theList])
	a = a.replace(' ','')
	a = a.replace('[','')
	a = a.replace(']','')
	a = a.replace(',','--')
	args.output_dir = a.replace(',','--')

# args.output_dir = './1219002003K_3-fcvae_0-gcnn_0-l0_1000.0-l3_1.0-l4_0.1-layer_2-lr_0.001-lz_d_128-model_chairbacksub'


datainfo = Config(args)

print(args.output_dir)
print(args.lz_d)
model = sdm_net.SDM_NET(datainfo)

if not test:
	model.train_scvae()

with tf.device('/cpu:0'):
	model.obtain_embeddings_v2(datainfo)
	# model.recover_mesh(datainfo)
	# model.random_gen(datainfo)
	# model.interpolate1(datainfo, [1, 2])


# model.interpolate1(args.output_dir + '/convmesh-model-2000', datainfo, [2,3])
# model.recover_mesh(args.output_dir + '/convmesh-model-4000', datainfo)
# model.recover_mesh(args.output_dir + '/convmesh-model-6000', datainfo)
# model.recover_mesh(args.output_dir + '/convmesh-model-8000', datainfo)

# model.recover_mesh(args.output_dir + '/convmesh-modelbest', datainfo)
# model.individual_dimension(args.output_dir + '/convmesh-modelbest', datainfo)



