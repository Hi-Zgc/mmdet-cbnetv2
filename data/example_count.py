from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt


dataDir='coco/'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
cat_nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(cat_nms)))

x_data = []
y_data = []

# 正确显示中文和负号
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 统计各类的图片数量和标注框数量
for cat_name in cat_nms:
    catId = coco.getCatIds(catNms=cat_name)
    imgId = coco.getImgIds(catIds=catId)
    annId = coco.getAnnIds(imgIds=imgId, catIds=catId, iscrowd=None)

    # 准备数据
    x_data.append(cat_name)
    y_data.append(len(annId))
    print("{:<15} {:<6d}     {:<10d}".format(cat_name, len(imgId), len(annId)))

# 画图，plt.bar()可以画柱状图
for i in range(len(x_data)):
	plt.bar(x_data[i], y_data[i])
# 设置图片名称
plt.title("实例分析")
# 设置x轴标签名
plt.xlabel("类别")
# 设置y轴标签名
plt.ylabel("实例数")
# 显示
plt.show()