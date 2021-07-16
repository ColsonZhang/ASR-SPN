# The ASR based on the SPN

## 项目简介

本项目为复微杯数字一赛题的SPN解决方案——即使用SPN网络构建一个孤立词语音识别模型。该项目文件给出了波形数据前处理的代码和模型训练的代码。

## 项目目录

```
----ASR\
    |----PreProcess\           数据预处理
    |    |----preprocess.py    前处理
    |----Train\                模型训练
    |    |----datasplit.py     训练集、测试集划分
    |    |----dtwsplit.py      DTW对齐分割
    |    |----train_spn.ipynb  模型训练和评估
    |----requirements.txt       需要的pypi包
    |----readme.md             使用说明
```

## 使用说明

1. 环境要求：Python3.8, 需要安装requirement.txt中的pypi包，可以直接通过命令`pip install -r requirements.txt`进行安装

2. 数据准备：请准备Google Speech Command的数据集合，并按照label分文件夹放置；

3. 数据前处理：打开文件`preprocess.py`修改输入波形数据文件路径并配置好输出路径；

4. 模型训练: 在进行模型训练时，配置`./Train/`目录下各个文件中的路径，然后`依次`执行`dtwsplit.py`、`datasplit.py`和`train_spn.ipynb`文件。其中`train_spn.ipynb`请使用jupyter lab/notebook打开，从上到下顺序执行即可。

5. 保存模型: 在`train_spn.ipynb`中存在导出SPN模型的功能，只需要配置好路径即可。

## 其他说明

由于完整的训练数据较大，只在该仓库中放置了小型的数据集合，如有需要请自行下载Google Speech Command 或 从百度云盘下载(链接: https://pan.baidu.com/s/1fXTGaAYHVPDtipNF287x-w 提取码: qwi2)。
