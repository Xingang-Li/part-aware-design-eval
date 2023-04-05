import tensorflow as tf
import numpy as np
import scipy.io as sio
from six.moves import xrange
import time
import h5py
import pickle
import argparse
from utils import *
import model_xl2 as sdm_net
import os
from config import *
import icp
import random


# vae
timecurrent = time.strftime('%m%d%H%M%S', time.localtime(time.time())) + '_' + str(random.randint(1000,9999))
test = True
parser = argparse.ArgumentParser()

# you must set this parameter
parser.add_argument('--output_dir', default='./result'+timecurrent, type = str)

# --------------------------not include in file name-------------------------------------
parser.add_argument('--model', default = 'scape', type = str) # training model name
parser.add_argument('--lz_d', nargs='+', default = [64, 128], type=int) # dimension of latent space
parser.add_argument('--l0', default = 1.0, type = float) # 1.0  generation loss
# parser.add_argument('--l1', default = 0.5, type = float) # 0.5  distance loss
parser.add_argument('--l2', default = 1.0, type = float) # 0.5 	weight loss
parser.add_argument('--l3', default = 1.0, type = float) # 0.005 Kl loss
parser.add_argument('--l4', default = 0.01, type = float) # 0.005 l2 loss
# parser.add_argument('--l5', default = 0.2, type = float) # 0.005 region loss
parser.add_argument('--joint', default = 0, type = int) # jointly training
parser.add_argument('--trcet', default = 0.75, type = float) # training percent of the dataset
parser.add_argument('--layer', default = 2, type = int) # the number of mesh convolution layer
parser.add_argument('--activ', default = '', type = str) # activate function of structure net : every layer
parser.add_argument('--lr', default = 0.001, type = float) # learning rate need adjust if the loss is too large in the early training
parser.add_argument('--batch_size', default = 8, type = int) # was 64
parser.add_argument('--iddatfile', default = 'id.dat', type = str)
parser.add_argument('--autoencoder', default='vae', type = str)
parser.add_argument('--finaldim', default = 9, type = int)
parser.add_argument('--filename', default = 'horse.obj', type = str)
parser.add_argument('--featurefile', default = 'hips_vaefeature.mat', type = str)
parser.add_argument('--controlid', nargs='+', default = [0], type=int)
parser.add_argument('--epoch_deform', default=20000, type = int)
parser.add_argument('--epoch_structure', default=40000, type = int)

# inter
# parser.add_argument('--beginid', default='1', type = str)
# parser.add_argument('--endid', default='2', type = str)
# parser.add_argument('--interid', nargs='+', default = ['1','2'], type=str)

args = parser.parse_args()

if not args.autoencoder.find('tan') == -1:
	args.l3 = 0.00

args.featurefile = args.model + '_vaefeature.mat'
args.filename = args.model + '.obj'
args.iddatfile = args.model + '_vaeid.dat'

ininame = getFileName(args.output_dir, '.ini')
if len(ininame)>1:
    x = int(input('Please select a number:'))
else:
    x = 0
args = inifile2args(args, os.path.join(args.output_dir, ininame[x]))
[print('{}: {}'.format(x,k)) for x,k in vars(args).items()]
# args.output_dir = './1219002003K_3-fcvae_0-gcnn_0-l0_1000.0-l3_1.0-l4_0.1-layer_2-lr_0.001-lz_d_128-model_chairbacksub'

# parafile = args.output_dir + '/checkpoint/convmesh-modelbest'# + str(args.maxepoch)


datainfo = Config(args, False)

print(args.output_dir)
print(args.lz_d)
model = sdm_net.SDM_NET(datainfo)

with tf.device('/cpu:0'):
    # model.obtain_embeddings_v2(datainfo)
    # model.obtain_partVAE_embeddings(datainfo)
    model.obtain_embeddings_total(datainfo)
    # model.recover_mesh(datainfo)
# model.random_gen(datainfo)
# model.interpolate1(datainfo, [args.beginid, args.endid])
# model.interpolate1(datainfo, args.interid)

