#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import cv2
from config.cfg import Cfg
import torch
from torch.backends import cudnn
from datasets.bases import read_image
sys.path.append('.')
from datasets import make_dataloader
from processor import do_inference
from model import make_model
from utils.logger import setup_logger
import torchvision.transforms as T
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#rename img
import string
import random


device = "cuda"
WEIGHT_PATH = '/nfs-data/lujj/projects/tmp_pose_tranfer_2/log/model_G_1800.pth'
#'/nfs-data/lujj/pretrained_model/pose-transfer/model_G_45.pth'
#'/nfs-data/lujj/projects/pose-transfer-jack-reid-01/log/tmp/model_G_180.pth'
Cfg.freeze()
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
cudnn.benchmark = True

test_transforms = T.Compose([
        T.Resize(Cfg.MODEL.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

model_G, _, _, _ = make_model(Cfg)
model_G.to(device)
#model_G = nn.DataParallel(model_G)
model_G.load_state_dict(torch.load(WEIGHT_PATH))


# In[ ]:


dataset = 'Market-1501-v15.09.15'
root_dir = '/home/lujj/datasets/{}/'.format(dataset)
data_dir = 'p4'
target_dir = '/home/lujj/datasets/{}/{}_g/'.format(dataset,data_dir)
target_dir2 = '/home/lujj/datasets/{}/{}_g_bak/'.format(dataset,data_dir)
img_list = []
pid_set = set()
for img in os.listdir(root_dir+data_dir):
    pid = img.split('_')[0]
    if pid in pid_set:
        continue
    else:
        pid_set.add(pid)
for img in os.listdir('/home/lujj/datasets/{}/bounding_box_train/'.format(dataset)):
    pid = img.split('_')[0]
    if pid in pid_set:
        continue
    else:
        pid_set.add(pid)
        img_list.append(img)
print('to generate pid:',len(img_list))
pose_list = os.listdir('/home/lujj/datasets/Market-1501-v15.09.15/pose_list/')
len_pose = len(pose_list)
print('body-part:',len_pose)


# In[ ]:


num_imgs = 17
model_G.eval()
for img in img_list:
    if img[-3:] == 'jpg':
        img1_path = '/home/lujj/datasets/{}/bounding_box_train/{}'.format(dataset,img)
        for pose2_idx in np.random.choice(range(len_pose),num_imgs, replace=False):
            target_pose = pose_list[pose2_idx]
            pose2_path = '/home/lujj/datasets/Market-1501-v15.09.15/train_part_heatmap/{}.npy'.format(target_pose)
            img1 = read_image(img1_path)
        # plt.imshow(img1)
        # plt.show()
            img1 = torch.unsqueeze(test_transforms(img1),0).to(device)
            pose_heatmap2 = np.load(pose2_path).astype(np.float32)
            pose2 = torch.tensor(pose_heatmap2.transpose((2, 0, 1)))
            pose2 = torch.unsqueeze(pose2,0).to(device)
            input_G = (img1, pose2)

            fake_img2 = model_G(input_G)
            result = fake_img2.cpu().detach().numpy()
            img1 = (np.transpose(result[0],(1,2,0))+ 1) / 2.0 * 255.0
            cv2.imwrite(target_dir+'{}-{}.jpg'.format(img[:-4],target_pose[:-4]),cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
            cv2.imwrite(target_dir2+'{}-{}.jpg'.format(img[:-4],target_pose[:-4]),cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))


# In[ ]:



for img in os.listdir(target_dir):
    src = target_dir+img
    target_img = ''.join(random.sample(string.ascii_letters + string.digits, 10))+'.jpg'
    img_ = img.split('-')
    dst = target_dir+img_[0]+target_img
    os.rename(src, dst)


# In[ ]:




