import torch
from sklearn.metrics import precision_score, recall_score
import numpy as np

def euclidean_distance(qf,gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf,2).sum(dim=1, keepdim=True).expand(m,n) +\
        torch.pow(gf,2).sum(dim=1, keepdim=True).expand(n,m).t()
    dist_mat.addmm_(1,-2,qf,gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf,gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True) #mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True) #nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1/qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1+epsilon,1-epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat

class Dist_Mat():
    def __init__(self, first_query=0, num_query=1, feat_norm='yes', method='euclidean'):
        super(Dist_Mat, self).__init__()
        self.first_query = first_query
        self.num_query = num_query
        self.feat_norm = feat_norm
        self.method = method
        self.reset()

    def reset(self):
        self.feats = []

    def update(self, output):#called once for each batch
        feat = output
        self.feats.append(feat)

    def compute(self):#called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2) #along channel
        # query
        qf = feats[self.first_query:self.num_query]
        # gallery
        gf = feats
        if self.method == 'euclidean':
            print("=> Computing DistMat with Euclidean Distance")
            distmat = euclidean_distance(qf, gf)
        elif self.method == 'cosine':
            print("=> Computing DistMat with Cosine Similarity")
            distmat = cosine_similarity(qf,gf)
        return distmat,feats
def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def accuracy(pred, label):
    pred = torch.argmax(pred, dim=1).long()
    acc = torch.mean((pred == label).float())
    pred = to_numpy(pred)
    label = to_numpy(label)
    p = precision_score(label, pred)
    r = recall_score(label, pred)
    return p,r,acc