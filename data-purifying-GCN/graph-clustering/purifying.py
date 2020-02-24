#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import os
import numpy as np
import random
import string
import shutil

from config.config import Configuration
Cfg = Configuration()
os.environ['CUDA_VISIBLE_DEVICES'] = Cfg.DEVICE_ID

preds = np.load('./log/preds.npy')
cids = np.load('./log/cids.npy')
h1ids = np.load('./log/h1ids.npy')
print(h1ids.shape)


# In[ ]:


k_at_hop = Cfg.NUM_HOP
knn = np.load('./log/knn.npy')
knn_graph = knn[:, :k_at_hop[0] + 1]
print(knn.shape)


# In[ ]:


pred_graph = {}
for idx in range(knn_graph.shape[0]):
    hops = list()
    center_node = idx
    depth = len(k_at_hop)
    hops.append(set(knn_graph[center_node][1:]))

    # Actually we dont need the loop since the depth is fixed here,
    # But we still remain the code for further revision
    for d in range(1, depth):
        hops.append(set())
        for h in hops[-2]:
            hops[-1].update(set(knn_graph[h][1:k_at_hop[d] + 1]))

    hops_set = set([h for hop in hops for h in hop])
    hops_set.update([center_node, ])
    unique_nodes_list = list(hops_set)
    # node_list including pivot, 1-hop, and 2-hop nodes
    unique_nodes_map = {j: i for i, j in enumerate(unique_nodes_list)}
    tmp_ = []
    for i, pred_edge in enumerate(preds[idx]):
        score = np.exp(pred_edge[1])/np.exp(pred_edge).sum()
        #print(score)
        #if np.argmax(pred_edge) == 1:
        if score > 0.5:
            #print(unique_nodes_map.keys())
            tmp_.append(list(unique_nodes_map.keys())[list(unique_nodes_map.values()).index(h1ids[idx][0][i])])
    print('=>Processing {}'.format(idx+1))
    pred_graph[idx] = tmp_


# In[ ]:


IMG_PATH_NPY = 'xxx/tmp_extraction_features/log/img_path.npy'
img_paths = np.load(IMG_PATH_NPY)
print(img_paths.shape)
pseudo_labels = []
for img_path in img_paths:
    pseudo_labels.append(img_path.split('/')[-1].split('_')[0])
np.save('./log/persudo_labels.npy', pseudo_labels)
print(len(pseudo_labels))
pseudo_labels_dict = {}
for v,k in enumerate(pseudo_labels):
    pseudo_labels_dict[k]=[]
for v,k in enumerate(pseudo_labels):
    pseudo_labels_dict[k].append(v)
print(len(pseudo_labels_dict))


# In[ ]:


def IoU(list1, list2):
    union = []
    inter = []
    union.extend(list1)
    union.extend(list2)
    for item in list1:
        if item in list2:
            inter.append(item)
    return len(inter)/(len(set(union))+0.0001)

def AffinityClusterPreservation(unlabel_list, len_unlabel_list, tmp_i2idx):
    preserved_indices = []
    max_cluster_len = max(len_unlabel_list)
    max_cluster = unlabel_list[np.argmax(len_unlabel_list)]
    if max_cluster_len == 0:
        return preserved_indices
    else:
        for i, cluster in enumerate(unlabel_list):
            if IoU(max_cluster, cluster) >=0.1:
                preserved_indices.extend(cluster)
    return preserved_indices

preserved_indices = []
for pid in pseudo_labels_dict.keys():
    print('=> Processing PID {}'.format(pid))
    indices = pseudo_labels_dict[pid]
    unlabel_list = []
    len_unlabel_list = []
    tmp_i2idx = {}
    for tmp_i,idx in enumerate(indices):
        tmp_i2idx[tmp_i] = idx
        unlabel_list.append(pred_graph[idx])
        len_unlabel_list.append(len(pred_graph[idx]))
    print(len(AffinityClusterPreservation(unlabel_list, len_unlabel_list, tmp_i2idx)))
    preserved_indices.extend(AffinityClusterPreservation(unlabel_list, len_unlabel_list, tmp_i2idx)) 


# In[ ]:


print(len(set(preserved_indices)))


# In[ ]:


#tmp
_l = set()
unused_img_list = []
for idx in range(knn_graph.shape[0]):
    for item in pred_graph[idx]:
        if item not in _l:
            _l.add(item)
print('Retained #image:',len(_l))
for item in range(h1ids.shape[0]):
    if item not in _l:
        unused_img_list.append(item)


# In[ ]:


# IMG_PATH_NPY = 'xxx/log/img_path.npy'
# img_paths = np.load(IMG_PATH_NPY)
# for i, img_path in enumerate(img_paths):
#     #if i not in unused_img_list:
#     if i in len(set(preserved_indices)):
#         src = img_path
#         camid = str(np.random.randint(1,7))
#         pid = str(img_path.split('/')[-1].split('_')[0])
#         target_img = '{}'.format(pid)+'_c{}s0_'.format(camid)+ ''.join(random.sample(string.ascii_letters + string.digits, 10))+'.jpg'
#         dst = '/xxx/Market-1501-v15.09.15/p2_g_gcn/'+target_img
#         shutil.copy(src, dst)


# In[ ]:


#for Duke
IMG_PATH_NPY = '/xxx/tmp_extraction_features/log/img_path.npy'
img_paths = np.load(IMG_PATH_NPY)
for i, img_path in enumerate(img_paths):
    #if i not in unused_img_list:
    if i in set(preserved_indices):
        src = img_path
        camid = str(np.random.randint(1,7))
        pid = str(img_path.split('/')[-1].split('_')[0])
        target_img = '{}'.format(pid)+'_c{}_'.format(camid)+ ''.join(random.sample(string.ascii_letters + string.digits, 10))+'.jpg'
        dst = '/xxx/DukeMTMC-reID/p3_g_gcn/'+target_img
        shutil.copy(src, dst)


# In[ ]:


source = '0020_c6_f0031012'
target = '0085_c8_f0024220'
img1_path = '/home/lujj/datasets/DukeMTMC-reID/bounding_box_train/{}.jpg'.format(source)
pose2_path = '/home/lujj/datasets/DukeMTMC-reID/train_part_heatmap/{}.jpg.npy'.format(target)
img1 = read_image(img1_path)
plt.imshow(img1)
plt.show()
img1 = torch.unsqueeze(test_transforms(img1),0).to(device)
pose_heatmap2 = np.load(pose2_path).astype(np.float32)
pose2 = torch.tensor(pose_heatmap2.transpose((2, 0, 1)))
pose2 = torch.unsqueeze(pose2,0).to(device)
input_G = (img1, pose2)


# In[ ]:




