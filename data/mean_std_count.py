import torch
from torch.utils.data import DataLoader, Dataset
import os
from pathlib import Path
import numpy as np
from PIL import Image


def cal_mean_std(path: str):
    channels_sum, channels_squared_sum, nums = 0, 0, 0
    path_list = os.listdir(path)
    for img_path in path_list:
        image_path = os.path.join(path, img_path)
        # image = torch.from_numpy(np.array(Image.open(image_path)) / 255).permute([2, 0, 1]).float()
        image = torch.from_numpy(np.array(Image.open(image_path))).permute([2, 0, 1]).float()
        channels_sum += torch.mean(image, dim=[1, 2])
        channels_squared_sum += torch.mean(image**2, dim=[1, 2])
        nums += 1
    mean = channels_sum / nums
    std = (channels_squared_sum / nums - mean**2)**0.5
    return mean, std


if __name__ == '__main__':
    path = os.path.abspath("../data1/coco/train2017/JPEGImages")
    mean, std = cal_mean_std(path=path)
    print(f'mean : {mean}, std : {std}')
# mean : tensor([131.5901, 129.8181, 126.6339]), std : tensor([56.4909, 55.3395, 56.3917])