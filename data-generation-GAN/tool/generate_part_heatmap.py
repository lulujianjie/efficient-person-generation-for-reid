import numpy as np
import pandas as pd
import json
import os

MISSING_VALUE = -1
split='test'
annotations_file = '/xxx/Market-1501-v15.09.15/market-annotation-{}.csv'.format(split)  # pose annotation path
save_path = '/xxx/Market-1501-v15.09.15/{}_part_heatmap'.format(split)  # path to store pose maps


def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)


# def cords_to_map(cords, img_size, sigma=6):
#     result = np.zeros(img_size + cords.shape[0:1], dtype='float32')
#     for i, point in enumerate(cords):
#         if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
#             continue
#         xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
#         result[..., i] = np.exp(-((yy - point[0]) ** 2 + (xx - point[1]) ** 2) / (2 * sigma ** 2))
#     return result

def L2Norm(v):
    return np.sqrt(np.power(v,2).sum(0))

def get_part_cords(part, cords, step=2):
    p1 = 0
    p2 = 1
    cord_list = []
    while p1 < p2 and p2 < len(part):
        start_point = cords[part[p1]]
        if start_point[0] == -1:
            p1+=1
            if p1 >= p2:
                p2+=1
            continue
        end_point = cords[part[p2]]
        if end_point[0] == -1:
            p2+=1
            continue
        direct_vector = end_point-start_point
        _norm = L2Norm(direct_vector)
        if _norm == 0:#avoid nan
            p1 += 1
            p2 += 1
            continue
        unit_vector = direct_vector/_norm
        nstep = np.int(_norm)#+1
        for s in range(0,nstep,step):
            new_point = start_point + s*unit_vector
            cord_list.append(np.rint(new_point))

        p1+=1
        p2+=1

    if len(cord_list) == 0:
        for i in range(len(part)):
            cord_list.append(cords[part[i]])

    return cord_list

# def cords_to_map(cords, img_size, sigma=6):
#     result = np.zeros(img_size, dtype='float32')
#     if len(cords) == 0:
#         print('=> find no pose part')
#         return result
#
#     for i, point in enumerate(cords):
#         scale = i/len(cords)
#
#         if point[0] == -1 or point[1] == -1:
#             continue
#         xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
#         result +=  np.exp(-((yy - point[0]) ** 2 + (xx - point[1]) ** 2) / (3*(2-scale) * sigma ** 2))
#     return np.clip(result/(i+1),0,1)

def cords_to_map(cords, img_size, sigma=6, using_scale=False):
    result = np.zeros(img_size, dtype='float32')
    if len(cords) == 0:
        print('=> find no pose part')
        return result
    for i, point in enumerate(cords):
        if using_scale:
            scale = i / len(cords)
        else:
            scale = 0
        if point[0] == -1 or point[1] == -1:
            continue
        xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
        result +=  np.exp(-((yy - point[0]) ** 2 + (xx - point[1]) ** 2) / (3*(2-scale) * sigma ** 2))
    return np.clip(result/(i+1),0,1)

def compute_pose(annotations_file, savePath):
    annotations_file = pd.read_csv(annotations_file, sep=':')
    annotations_file = annotations_file.set_index('name')
    image_size = (128, 64)
    cnt = len(annotations_file)
    part_groups = {'head':[0,16,14,15,17,0],
                   'body':[2,1,5,11,8,2],
                   'l_up_limb':[2,3,4],
                   'r_up_limb':[5,6,7],
                   'l_low_limb':[8,9,10],
                   'r_low_limb':[11,12,13]
                   }
    for i in range(cnt):
        print('processing %d / %d ...' % (i+1, cnt))
        row = annotations_file.iloc[i]
        name = row.name
        file_name = os.path.join(savePath, name + '.npy')
        cords = load_pose_cords_from_strings(row.keypoints_y, row.keypoints_x)
        part_map = np.zeros((image_size[0], image_size[1], len(part_groups)))
        #for idx, key in enumerate(['head', 'body', 'l_up_limb', 'r_up_limb', 'l_low_limb', 'r_low_limb']):
        cord_list = get_part_cords(part_groups['head'],cords)
        part_map_c = cords_to_map(cord_list, img_size=(128, 64), sigma=5, using_scale=False)
        part_map[..., 0] = part_map_c

        cord_list = get_part_cords(part_groups['body'],cords)
        part_map_c = cords_to_map(cord_list, img_size=(128, 64), sigma=4, using_scale=False)
        part_map[..., 1] = part_map_c

        cord_list = get_part_cords(part_groups['l_up_limb'], cords)
        part_map_c = cords_to_map(cord_list, img_size=(128,64), sigma=3, using_scale=True)
        part_map[..., 2] = part_map_c

        cord_list = get_part_cords(part_groups['r_up_limb'], cords)
        part_map_c = cords_to_map(cord_list, img_size=(128,64), sigma=3, using_scale=True)
        part_map[..., 3] = part_map_c

        cord_list = get_part_cords(part_groups['l_low_limb'], cords)
        part_map_c = cords_to_map(cord_list, img_size=(128, 64), sigma=4, using_scale=True)
        part_map[..., 4] = part_map_c

        cord_list = get_part_cords(part_groups['r_low_limb'], cords)
        part_map_c = cords_to_map(cord_list, img_size=(128, 64), sigma=4, using_scale=True)
        part_map[..., 5] = part_map_c

        assert np.max(part_map) <= 1
        np.save(file_name, part_map)
    # for i in range(cnt):
    #     print('processing %d / %d ...' % (i+1, cnt))
    #     row = annotations_file.iloc[i]
    #     name = row.name
    #     file_name = os.path.join(savePath, name + '.npy')
    #     cords = load_pose_cords_from_strings(row.keypoints_y, row.keypoints_x)
    #     part_map = np.zeros((image_size[0], image_size[1], len(part_groups)))
    #     for idx, key in enumerate(['head', 'body', 'l_up_limb', 'r_up_limb', 'l_low_limb', 'r_low_limb']):
    #         cord_list = get_part_cords(part_groups[key],cords)
    #         part_map_c = cords_to_map(cord_list, img_size=(128, 64), sigma=6)
    #         #print(np.max(part_map_c))
    #         assert np.max(part_map_c) <= 1
    #         part_map[..., idx] = part_map_c
    #     np.save(file_name, part_map)


compute_pose(annotations_file, save_path)