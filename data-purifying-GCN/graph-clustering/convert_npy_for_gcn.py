import torch
import os
import numpy as np
from config.config import Configuration
FEATS_PATH_NPY = '/xxx/projects/tmp_extraction_features/log/feats.pth'
IMG_PATH_NPY = '/xxx/projects/tmp_extraction_features/log/img_path.npy'


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

if __name__ == "__main__":
    Cfg = Configuration()
    log_dir = Cfg.LOG_DIR

    os.environ['CUDA_VISIBLE_DEVICES'] = Cfg.DEVICE_ID
    feats = torch.load(FEATS_PATH_NPY)
    feats_numpy = feats.cpu().numpy()
    np.save('./log/feats.npy', feats_numpy)
    print('feats shape:{}'.format(feats_numpy.shape))

    paths = np.load(IMG_PATH_NPY)
    labels = np.zeros((len(paths), 1))
    for idx in range(len(paths)):
        labels[idx] = int(paths[idx].split('/')[-1][:4])
    np.save('./log/labels.npy', labels)

    dist_mat = euclidean_distance(feats, feats)
    np.save('./log/dist_mat.npy', dist_mat)

    indices = np.argsort(dist_mat, axis=1)
    np.save('./log/knn.npy', indices)