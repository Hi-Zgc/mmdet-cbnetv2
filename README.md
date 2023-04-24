# mmdet-cbnetv2
基于mmdet2.25.3实现Cbnetv2。在数据增强方法中，将实例分割任务的Simple Copy Paste 应用于目标检测。此版本还包括Albu, AutoAugment, Mosaic, MixUp和其他数据增强方法。

# 安装过程

## 一、安装版本

| 名称        | 版本   |
| ----------- | ------ |
| pytorch     | 1.12.1 |
| python      | 3.8    |
| torchvision | 0.13.1 |
| torchaudio  | 0.12.1 |
| openmim     | 0.3.2  |
| mmcv-full   | 1.7.0  |
| mmdet       | 2.25.3 |

## **二、官网下载安装Anaconda**

[Anaconda官网](https://www.anaconda.com/)

## **三、创建 conda 环境并激活**

```java
conda create --name openmmlab python=3.8 -y #创建虚拟环境
conda activate openmmlab #激活虚拟环境

nvcc --version #查看cuda版本
```

## **四、安装 PyTorch**

```java
conda install pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 cudatoolkit=11.3

# pip 安装whl文件
# https://download.pytorch.org/whl/torch_stable.html
```

## **五、安装mmdet**

**MMDetection 的运行依赖 MMCV ，它是一个面向计算机视觉的基础库，其中支持了很多开源工具，如图像分类工具、目标检测工具、语义分割工具、姿态估计工具等常用工具。**

```java
pip install -U openmim
mim install mmcv-full==1.7.0
```

## **六、解压mmdet压缩包**

```java
tar -xvf mmdet.rar
# or
git clone https://github.com/Hi-Zgc/mmdet-cbnetv2.git
cd mmdetection
pip install -r requirements.txt
pip install -v -e .
```

**注：AttributeError: module 'numpy' has no attribute 'float'.**

```java
# 1.24删除了float，安装1.23.5
pip install -U numpy==1.23.5
```

# **这里有几处改动**

1. **基于mmdet2.25.3 修改CBNetV2 中针对Swin Transfomer的实现**
2. **Simple Copy Paste 多应用于实例分割任务当中，为适配目标检测任务，修改相关代码**
3. **新增脚本对数据集进行分析计算**



## 改动1

cbnet下配置文件中embed_dim修改为embed_dims，ape与use_checkpoint参数需删除，正确主干网络配置文件示例如下：

```python
backbone=dict(
        type='CBSwinTransformer',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
   )
```



## 改动2

对于配置文件来说，只需要在MultiScaleFlipAug下加入CopyPaste，因为原本json里并没有mask标签，那LoadAnnotations里不需要设置with_mask=True，Collect的key里面也不需要添加‘gt_masks'，与正常目标检测的config一致。

两点注意事项

1. 论文中对输入图像先进行了resize到相同大小，在使用copypaste前需进行resize

2.  keep_ratio 需设成False

   数据增强正确示例如下：

```
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
load_pipeline= [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
]
train_pipeline = [
    dict(type='Resize', img_scale=(300, 200), keep_ratio=False),
    dict(type='CopyPaste', max_num_pasted=100),
    dict(type='Mosaic', img_scale=(250, 150), pad_val=114.0),
    dict(
        type='MixUp',
        img_scale=(250, 150),
        ratio_range=(0.8, 1.6),
        pad_val=114.0)
]      
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            pipeline=load_pipeline,
            filter_empty_gt=False,),
        pipeline=train_pipeline,))
```



## 改动3

新增脚本

[count_h_w.py](./data/count_h_w.py)    计算数据集长宽分布

[example_count.py](./data/example_count.py)  每类实例数计算

[mean_std_count.py](./data/mean_std_count.py)  数据集均值，标准差计算

[voc2coco.py ](./data/voc2coco.py) voc格式转coco格式

