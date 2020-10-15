import argparse
import numpy as np
import math
import random
import os
import cv2
import tarfile
import io
import av
import time
import tensorboardX
import imgaug.augmenters as iaa
from IPython.display import Image, display
from functools import lru_cache
from megengine.optimizer import Adam
from model import *
from IPython import embed
from megengine.data.dataset import Dataset
from megengine.data import RandomSampler, SequentialSampler
from megengine.data import DataLoader
from tensorboardX import *
from megengine.jit import trace, SublinearMemoryConfig

TRAIN_DATA_STORAGE = "/data/npy/"

train_epoch = 10
batch_size = 64

net = SimpleUNet()
optimizer = Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

MODEL_PATH = "model.mge.state"
with open(MODEL_PATH, 'rb') as f:
    net.load_state_dict(mge.load(f)['net'])

random.seed(100)

train_patches = []
# tmp = sorted([os.path.join(TRAIN_DATA_STORAGE, f) for f in os.listdir(TRAIN_DATA_STORAGE)])
for i in range(10): #1850
    train_patches.append(TRAIN_DATA_STORAGE + str(i)) # + '_down.npy')
image_list = []
config = SublinearMemoryConfig(genetic_nr_iter=20)
print("starting")

@trace(symbolic=True, sublinear_memory_config=config)
def train_iter(batch_train, batch_gt):
    pred = net(batch_train)
    loss = (((batch_gt - pred)**2 + 1e-6)**0.5).mean()
    optimizer.backward(loss)
    return loss, pred

@trace(symbolic=True, sublinear_memory_config=config)
def train_iterv2(batch_train, batch_gt):
    pred = net(batch_train)
    loss = ((batch_gt - pred)**2).mean()
    optimizer.backward(loss)
    return loss, pred

def validate():
    psnr_list = []
    mse_list = []
    base_list = []
    for image_file in val_dataset:
        inp = image_file[0][None, :, :, :]
        pred = inference(np.float32(inp)).numpy()[0]
        mse = ((image_file[1]/1.0-pred/1.0) ** 2).mean()
        base = ((image_file[0]/1.0-image_file[1]/1.0) ** 2).mean()
        mse_list.append(mse)
        base_list.append(base)
    print(10 * math.log10(1.0/np.array(mse_list).mean()), 10 * math.log10(1.0/np.array(base_list).mean()))
    return 10 * math.log10(1.0/np.array(mse_list).mean()), np.array(mse_list).mean()

# validation                                                                                                                                                 
@mge.jit.trace
def inference(inp):
    # inp = inp.transpose(1, 2, 0)
    img = net(inp)
    # img1 = net(inp[:, ::-1, :, :])[:, ::-1, :, :]    
    return img # (img0 + img1) / 2

class ImageDataSet(Dataset):
    def __init__(self, now_dataset, name='train'):
        super().__init__()
        self.data = []
        self.name = name
        cnt = 0
        for image_file in now_dataset:
            img = np.load(image_file + '_down.npy')
            gt = np.load(image_file + '.npy')
            for i in range(2000):
                self.data.append(np.concatenate((img[:, i*128:i*128+128], gt[:, i*128:i*128+128]), 2))
                cnt += 1
                if cnt % 20000 == 0:
                    print("loading", cnt, len(now_dataset) * 200)
        self.len = len(self.data)

    def __getitem__(self, index):
        img0 = np.float32(self.data[index][:, :, :3] / 255.), np.float32(self.data[index][:, :, 3:6] / 255.)
        if self.name == 'val':
            return np.transpose(img0[0], (2, 0, 1)), np.transpose(img0[1], (2, 0, 1))
        aug = iaa.Sequential([
            iaa.ChannelShuffle(0.5),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Rot90([0, 3]),
            ]).to_deterministic()        
        p = np.random.rand()
        if p < 0.5:
            return np.transpose(img0[0], (2, 0, 1)), np.transpose(img0[1], (2, 0, 1))
        index2 = np.random.randint(0, self.len - 1)              
        img1 = np.float32(self.data[index2][:, :, :3] / 255.), np.float32(self.data[index2][:, :, 3:6] / 255.)            
        mix0 = aug(image = img0[0] * p + img1[0] * (1-p))
        mix1 = aug(image = img0[1] * p + img1[1] * (1-p))
        return np.transpose(mix0, (2, 0, 1)), np.transpose(mix1, (2, 0, 1))

    def shuffle(self):
        np.random.shuffle(self.data)
    
    def __len__(self):
        return self.len
        
train_dataset = ImageDataSet(train_patches[8:])
val_dataset = ImageDataSet(train_patches[:8], 'val')

loss_acc = 0
loss_acc0 = 0

cnt = 1
writer = SummaryWriter('./log')

state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
        }

net.eval()
val_res = validate()
val_psnr = val_res[0]
net.train()
# writer.add_scalar("psnr", val_psnr, cnt)
print(val_res)
# with open('log.txt','a') as f: 
#        print(val_res, file=f)

