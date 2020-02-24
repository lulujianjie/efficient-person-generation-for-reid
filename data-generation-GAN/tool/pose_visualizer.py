import pandas as pd
import numpy as np
import cv2
import json

def pose_visualizer(csv_path, data_path, mode='random'):
    pose_df = pd.read_csv(csv_path, sep=':')
    pose_df = pose_df.set_index('name')
    if mode == 'random':
        idx = np.random.randint(len(pose_df))
    else:
        assert ('unsupported mode, expect:random, but got {}'.format(mode))
    row = pose_df.iloc[idx]
    img_bgr = cv2.imread(data_path + row.name)  # bgr
    img_size = (64, 128)  # WxH
    img_bgr = cv2.resize(img_bgr, img_size, interpolation=cv2.INTER_CUBIC)  #
    cordx = json.loads(row.keypoints_x)
    cordy = json.loads(row.keypoints_y)

    for i in range(len(cordx)):
        cv2.circle(img_bgr, (cordx[i], cordy[i]), 3, (0, 0, 225), 1)
    print(img_bgr)
    cv2.imwrite('./log/pose1.jpg', img_bgr)

csv_path = '/xxx/Market-1501-v15.09.15/market-annotation-train.csv'
data_path = '/xxx/Market-1501-v15.09.15/bounding_box_train/'
pose_visualizer(csv_path, data_path)