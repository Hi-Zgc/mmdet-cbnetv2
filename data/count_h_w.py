import os
import pandas as pd
from PIL import Image
from collections import Counter


source = './coco/train2017/JPEGImages'
imgfile = os.listdir(source)
shot_cut = []
long_cut = []

print('filenum:', len([lists for lists in os.listdir(source)]))

for img in imgfile:
    img_path = os.path.join(source, img)
    img = Image.open(img_path)
    imgSize = img.size  # 图片的长和宽
    #print(imgSize)
    maxSize = max(imgSize)  # 图片的长边
    long_cut.append(maxSize)
    minSize = min(imgSize)  # 图片的短边
    shot_cut.append(minSize)

#shot_result = Counter(shot_cut)
#long_result = Counter(long_cut)
shot_result = pd.value_counts(shot_cut, normalize=True)
long_result = pd.value_counts(long_cut, normalize=True)

print(shot_result)
print(long_result)

shot_result.to_csv('shot.csv', header=0)
long_result.to_csv('long.csv', header=0)
