import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__':
    input = '0773_c4s4_017010_02'
    part = np.load('/xxx/Market-1501-v15.09.15/train_part_heatmap/{}.jpg.npy'.format(input))
    body = np.zeros((128, 64, 6))
    for i in range(6):
        data = part[:, :, i]
        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=data.min(), vmax=data.max())

        body[:, :, i] = norm(data)
    norm2 = plt.Normalize(vmin=body.min(), vmax=body.max())
    # save the image
    plt.imsave('./log/{}.png'.format(input), cmap(norm2(body.max(2))))