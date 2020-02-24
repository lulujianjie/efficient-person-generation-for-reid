from PIL import Image,ImageFile
import numpy as np
from torch.utils.data import Dataset
import os.path as osp
import torch
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, epoch_size=200, split='train'):
        self.dataset = dataset.datalist
        self.transform = transform
        self.epoch_size = epoch_size
        self.dataset_size = len(self.dataset)
        self.split = split
    def __len__(self):
        if self.epoch_size == 'small':
            return 100
        elif self.epoch_size == 'medium':
            return 4000
        elif self.epoch_size == 'large':
            return len(self.dataset)
        else:
            return self.epoch_size

    def __getitem__(self, index):
        if self.split == 'train':
            index = random.randint(0, self.dataset_size-1)
        img_path1, pose_path1, img_path2, pose_path2 = self.dataset[index]
        img1 = read_image(img_path1)
        img2 = read_image(img_path2)
        pose_heatmap1 = np.load(pose_path1).astype(np.float32)
        pose_heatmap2 = np.load(pose_path2).astype(np.float32)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

            pose_heatmap1 = pose_heatmap1.transpose((2, 0, 1))
            pose_heatmap2 = pose_heatmap2.transpose((2, 0, 1))

        return {'img1':img1, 'pose1':pose_heatmap1,
                'img2':img2, 'pose2':pose_heatmap2,
                'img_path1':img_path1, 'img_path2':img_path2}