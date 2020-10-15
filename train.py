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
from utils import *
import tensorboardX
import imgaug.augmenters as iaa
from functools import lru_cache
from megengine.optimizer import Adam
from model import *
from megengine.data.dataset import Dataset
from megengine.data import RandomSampler, SequentialSampler
from megengine.data import DataLoader
from tensorboardX import *
from megengine.jit import trace, SublinearMemoryConfig

train_epoch = 50
batch_size = 64

net = SimpleUNet()
optimizer = Adam(net.parameters(), lr=3e-4)

train_patches = []
gt_patches = []
for i in range(5): 
    train_patches.append(TRAIN_DATA_STORAGE + str(i))
image_list = []
config = SublinearMemoryConfig(genetic_nr_iter=20)

@trace(symbolic=True, sublinear_memory_config=config)
def train_iter(batch_train, batch_gt):
    pred = net(batch_train)
    loss = (((batch_gt - pred)**2 + 1e-6)**0.4).mean()
    optimizer.backward(loss)
    return loss, pred

@trace(symbolic=True, sublinear_memory_config=config)
def train_iterv2(batch_train, batch_gt):
    pred = net(batch_train)
    loss = ((batch_gt - pred)**2).mean()
    optimizer.backward(loss)
    return loss, pred

def validate():
    img_list = []
    gt_list = []
    l2_list = []
    cubic_list = []
    for i in range(537):
        gt = cv2.imread('/data/validate/{}.png'.format(i + 1))[256:512, 256:512, :]
        img = cv2.imread('/data/validate/{}_down.png'.format(i + 1))[64:128, 64:128, :]
        img = cv2.resize(img, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC) / 255.        
        img = np.float32(img)
        h, w, _ = img.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        img = np.pad(img, ((0,ph-h),(0,pw-w),(0,0)), 'constant')
        if len(img_list) != 0 and img_list[-1].shape != img.shape:
            img_list = []
            gt_list = []
        img_list.append(img)
        gt_list.append(gt)
        if(len(img_list) > 5):
            img_list = img_list[-5:]
            gt_list = gt_list[-5:]
        if(len(img_list) == 5):
            inp = np.zeros([5, ph, pw, 3])
            for k in range(5):
                inp[k] = img_list[k]
            inp = np.float32(inp)
            for i in range(4):
                if np.abs(inp[4-i] - inp[4-i-1]).mean() > 0.2:
                    inp[4-i-1] = inp[4-i]
                if np.abs(inp[4+i] - inp[4+i+1]).mean() > 0.2:
                    inp[4+i+1] = inp[4+i]
            inp = inp.transpose((0, 3, 1, 2)).reshape(1, 15, ph, pw)
            img_out = inference(inp).numpy()[0]
            img_out = ((img_out * 255).transpose((1, 2, 0)).copy()[:h, :w]).astype('uint8')
            l2_list.append(((gt_list[4][:h, :w]/255. - img_out/255.)**2).mean())
            cubic_list.append(((gt_list[4][:h, :w]/255. - img_list[4][:h, :w])**2).mean())
    print(10 * np.log10(1 / np.array(l2_list).mean()), 10 * np.log10(1 / np.array(cubic_list).mean()))
    return 10 * np.log10(1 / np.array(l2_list).mean())
    
@trace(symbolic=True)
def inference(inp):
    inp = net(inp)
    return inp

class ImageDataSet(Dataset):
    def __init__(self, now_dataset):
        super().__init__()
        self.data = []
        cnt = 0

    def load(self):
        TRAIN_RAW_DATA='/data/train_png/'
        img_num = 0
        self.data = []
        from tqdm import tqdm
        tasks = sorted([i for i in os.listdir(TRAIN_RAW_DATA) if 'down4x' in i])
        for task in tqdm(tasks):
            num = task[:2]
            if num[0] == '0':
                num = num[1]
            if(eval(num) < 10): # the first 10 video are used for validation
                continue
            task = TRAIN_RAW_DATA + task
            down4x_list = []
            task_origin = task.replace('_down4x.mp4','')
            frames_origin = sorted([os.path.join(task_origin,i) for i in os.listdir(task_origin)])
            frames_down4x = sorted([os.path.join(task,i) for i in os.listdir(task)])
            for k, (frame_down4x) in enumerate(frames_down4x):
                img_down4x = cv2.imread(frame_down4x)
                down4x_list.append(img_down4x)
            assert(len(frames_origin) == len(down4x_list))
            for k, (frame_origin) in enumerate(frames_origin):
                if k < 4 or k + 4 >= len(down4x_list):
                    continue
                if np.random.uniform(0, 1) < 0.95:
                    continue
                img_origin = cv2.imread(frame_origin)
                x0 = 0
                tmp = np.array(down4x_list[k-4:k+5])
                while x0 < img_origin.shape[0]:
                    if x0 + 128 > img_origin.shape[0]:
                        x0 = img_origin.shape[0] - 128
                    y0 = 0                    
                    while y0 < img_origin.shape[1]:
                        if y0 + 128 > img_origin.shape[1]:
                            y0 = img_origin.shape[1] - 128;
                        img0 = tmp[:, x0//4:x0//4 + 32, y0//4:y0//4 + 32].copy()
                        img1 = img_origin[x0:x0 + 128, y0:y0 + 128].copy()
                        if img1.mean() > 10:
                            self.data.append((img0, img1))
                            img_num += 1
                        y0 += 128
                    x0 += 128
        self.len = len(self.data)
        print(self.len)

    def __getitem__(self, index):
        img0 = (self.data[index][0].copy() / 255., self.data[index][1].copy() / 255.)
        aug = iaa.Sequential([
            iaa.ChannelShuffle(0.5),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Rot90([0, 3]),
            ]).to_deterministic() 
        if np.random.rand() < 0.5:
            inp = img0[0]
            gt = img0[1]
        else:
            p = np.random.uniform(0.1, 0.9)
            index2 = np.random.randint(0, self.len - 1)              
            img1 = (self.data[index2][0].copy() / 255., self.data[index2][1].copy() / 255.)
            inp = img0[0] * p + img1[0] * (1-p)
            gt = img0[1] * p + img1[1] * (1-p)
        if np.random.rand() < 0.5:
            inp = inp[::-1]
        base = []
        for i in range(9):
            inp[i] = aug(image=inp[i])
        gt = aug(image=gt)
        for i in range(9):
            base.append(cv2.resize((inp[i]*255).astype('uint8'), (128, 128), interpolation=cv2.INTER_CUBIC) / 255.)
        for i in range(4):
            if np.abs(base[4-i] - base[4-i-1]).mean() > 0.2:
                base[4-i-1] = base[4-i]
            if np.abs(base[4+i] - base[4+i+1]).mean() > 0.2:
                base[4+i+1] = base[4+i]
        base = np.transpose(np.array(base), (0, 3, 1, 2)).reshape(15, 128, 128)
        gt = np.transpose(gt, (2, 0, 1))
        return np.float32(base), np.float32(gt)

    def shuffle(self):
        np.random.shuffle(self.data)

    def __len__(self):
        return self.len
        
train_dataset = ImageDataSet(train_patches[1:])

loss_acc = 0
loss_acc0 = 0

cnt = 0
writer = SummaryWriter('./log')

state = {
    'net': net.state_dict(),
    'opt': optimizer.state_dict(),
}
for epoch in range(train_epoch):
    if epoch % 2 == 0:
        train_dataset.load()
        random_sampler = RandomSampler(dataset=train_dataset, batch_size=batch_size, seed=epoch)
        image_dataloader = DataLoader(
            dataset=train_dataset,
            sampler=random_sampler,
            num_workers=8,
        )
    begin = time.time()    
    for idx, (img, label) in enumerate(image_dataloader):
        if idx % 100 == 0:
            print('{} / {}'.format(idx, train_dataset.__len__() // batch_size))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * cnt / (train_epoch * 2700) ) )
            lr = 3e-4 * cosine_decay
            for g in optimizer.param_groups:
                g['lr'] = lr
        train_begin = time.time()
                         
        optimizer.zero_grad()            
        loss, pred = train_iter(img, label)
        optimizer.step()        
        loss_acc = loss_acc * 0.99 + loss
        loss_acc0 = loss_acc0 * 0.99 + 1
        end = time.time()

        total_time = end - begin
        data_load_time = total_time - (end - train_begin)

        begin = time.time()
        if idx % 100 == 0:
            writer.add_scalar("loss",(loss_acc / loss_acc0).numpy(), cnt)
            writer.add_scalar("learning_rate", lr, cnt)
        cnt += 1

    print(
        "{}: loss: {}, speed: {:.2f}it/sec, tot: {:.4f}s, data: {:.4f}s, data/tot: {:.4f}"
        .format(epoch, loss_acc / (loss_acc0+1e-6), 1 / (total_time+1e-6), total_time,
                data_load_time, data_load_time / (total_time+1e-6)))
    with open('log.txt','a') as f: 
        print(
            "{}: loss: {}, speed: {:.2f}it/sec, tot: {:.4f}s, data: {:.4f}s, data/tot: {:.4f}"
            .format(epoch, loss_acc /  (loss_acc0+1e-6), 1 / (total_time+1e-6), total_time,
                    data_load_time, data_load_time / (total_time+1e-6)), file=f)

    if (epoch+1) % 1 == 0:
        val_res = validate()
        val_psnr = val_res
        net.train()
        writer.add_scalar("psnr", val_psnr, cnt)
        print(val_res)
        with open('log.txt','a') as f: 
            print(val_res, file=f)

        # save our model
        state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
        }
        with open('model.mge.state', 'wb') as fout:
            mge.save(state, fout)
