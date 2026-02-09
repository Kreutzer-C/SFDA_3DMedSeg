# SFDA 3D Med Seg

我们来在当前目录中从头创建一个 Source-Free Domain Adaptation （SFDA）3D医学影像分割任务代码项目，我需要你帮助完成数据预处理、分割模型搭建、训练推理流程、实验结果记录等重要功能，下面将介绍具体要求：

## 项目结构

本代码项目以这个结构构建：

```python
SFDA_3DMedSeg
|-- dataloader # 存放数据加载器、数据增强函数等
|-- datasets # 数据集存储路径
|   `-- ABDOMINAL # 数据集之一
|       |-- preprocessed # 预处理后数据存放路径
|       |-- raw # 原始数据
|       |-- metadata.json # 保存数据集元数据，例如名称、域名、类别名、训练/测试case划分
|       `-- abominal_preprocessor.py # 针对该数据集的预处理全流程脚本
|-- models # 存放各种模型定义
|-- trainer # 包含各种方法的训练、推理类
|-- utils # 存放各种工具，如损失函数、评价指标等
|-- results # 训练、推理结果保存路径
`-- main.py # 入口，定义各种超参数调用trainer
```

下面将对某些部分做单独要求

## Dataloader

实现各种数据加载器，将 `./datasets/xxx/preprocessed/` 中的数据进行合适的随机增强、patchify切分、转化为模型输入格式。

## Datasets

我会在`./datasets/xxx/raw/` 中提供原始数据，这些数据的格式不定，可能是dicom，nii，甚至png等格式，你需要严格检查这些数据 image 和 label 的空间像素匹配情况（因为这对分割任务很重要），然后填写一份 metadata.json 元文件（metadata中不确定的部分可以留给我来后续修改）

然后在 [preprocessor.py](http://preprocessor.py) 中针对当前原始数据进行预处理，这可能包含裁剪 non-body 区域、窗宽截取、像素值归一化、轴向resize、训练测试集划分等工作。注意，对同一数据集中的不同域可能有不同的预处理方式。

## Models

在这里实现各种模型网络结构的定义，我们暂时只需要一个3D U-Net来作为基线网络，关于这个3D U-Net的实现参考另一个代码项目，绝对路径是 `/opt/data/private/pytorch-3dunet` 。

## Trainer

这是最核心的部分，所有方法的训练推理核心逻辑都包含在这里，在设计这部分时需要兼顾灵活性和可读性，方便我后续对这部分进行方法层创新性修改。暂时我们只需要实现 source domain pretrain 和 target domain oracle adaptation 两个有监督的 trainer 类。

关于推理类，加载训练好的模型checkpoint和测试集进行推理然后计算Dice，ASSD性能指标，需要注意的是处理模型的 patch 级分割结果如何拼成完整 case

## Utils

主要包括需要自定义的 loss, metrics 等，如果有其他杂项也可以放在这里

## Results

本地保存所有实验结果、配置参数、日志等，需要注意按数据集，域名，实验名（启动训练时的必选自定义参数）等信息分层有序存储

## main.py

通过argparser传入参数，调用各种训练推理类的入口

## 注意事项

1. 因为分割任务的特点，在数据预处理、增强全流程中应该始终关注 image 和 label 的空间像素一致性
2. 使用 wandb 上传实验中间过程信息（loss, metrics等），当然本地也要存一份log文档
3. 原生支持DP多卡训练