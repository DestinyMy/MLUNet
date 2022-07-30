## 介绍
本项目为“面向寒武纪加速卡MLU270-F4的多目标神经结构搜索方法”论文源码

## 安装依赖
```
pytorch>=1.4.0
torchvision>=0.5.0
```

## 使用方法
```
# search process
python EMO.py

# best solutions
CIFAR: [64, [5, 4, 6, 1, 6], [2, 1, 1, 4, 7, 2], [1, 2, 7, 0, 3, 7, 2, 6], [2, 1]]
ImageNet: [64, [1, 3, 5], [1, 1, 5, 4, 1, 2, 6, 1, 2, 2], [5, 3, 4], [0, 5]]

# training process（you can use our public checkpoints to reduce training time）
python train_cifar.py
python train_imagenet.py
```

## 致谢
感谢[NSGA-II](https://ieeexplore.ieee.org/abstract/document/996017)，[NSGA-Net](https://dl.acm.org/doi/abs/10.1145/3321707.3321729)以及[MFENAS](https://ieeexplore.ieee.org/abstract/document/9786036)论文的帮助
