import torch.utils.data as data
import os
import os.path as osp
from PIL import Image
import numpy as np

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

class NewDataset(data.Dataset):
    def __init__(self,Cfg, transform=None):
        self.transform = transform
        self.img_path_list = []
        self.root = '/xxx/DukeMTMC-reID/p1_g_bak/'
        for file in os.listdir(self.root):
            if file[-3:] == 'jpg':
                self.img_path_list.append(os.path.join(self.root, file))

    def __getitem__(self, idx):
        img = read_image(self.img_path_list[idx])
        path = self.img_path_list[idx]
        if self.transform is not None:
            img = self.transform(img)

        return img, path

    def __len__(self):
        return len(self.img_path_list)