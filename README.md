# CIFAR10_mxnet
### abstract
kaggle CIFAR10 compitition code, implement by mxnet gluon.</br>
we got 0.9688 by merge some ideas from https://discuss.gluon.ai/t/topic/1545/423</br>
![](submission.png)

### directroy and file descriptor

file | descriptor
--- | ---
log | some train log file
models | some trianed model
result | the forward result file on kaggle test set
submission | the finnal kaggle submission result
CIFAR10_train | main train and exp code
plot | the visulization  of train acc and valid acc and loss with epoch
netlib.py | ResNet18, ResNet164_v2, densenet, Focal Loss implement code by gluon, invoke by CIFAR10_train
utils.py | some tool function


使用mxnet编写的kaggle CIFAR10比赛的代码
### 目录文件描述

文件名 | 描述
--- | ---
log | 一些训练的日志，主要是训练的loss和acc
models | 一些训练的模型
result | 程序forward的最终结果，保留了10个类别的output
submission | 最终提交的结果文件
CIFAR10_train | CIFAR10上训练模型和产生结果的代码，主程序。
plot | 绘制一些模型训练过程的精度和loss曲线
netlib.py | ResNet18, ResNet164_v2, densenet, Focal Loss 的gluon的实现,被调用。
utils.py | 一些工具函数

models、result、log等内容加起来有点大，等传到网盘上，供各位小伙伴参考下载。

### 方法描述
参考[论坛](https://discuss.gluon.ai/t/topic/1545/423)几个小伙伴的方法，我总结了一下，大致如下:
1. 使用不同的网络的数据增强的方法，我们做了多个实验，得到了多个网络模型(全部放到了models下面)，然后ensemble，发现下面5个网络的效果最好。</br>
这5个网络的训练策略和单独提交的精度分别是：

policy | kaggle 精度
--- | ---
res164_v2 + DA1| 0.9529
res164_v2 + DA2| 0.9527
res164_v2 + focal loss + DA3| 0.9540
res164_v2 + focal loss + DA3 | 只使用train_data训练: 0.9506
[sherlock_densenet](https://discuss.gluon.ai/t/topic/1545/273)| 0.9539

上面的DA是3中不同的数据增强的方法，
DA1就是最常用的那种padding到40,然后crop的方法，就是sherlock代码里使用的加强
DA2是先resize到一定的大小，然后crop的方法，同时设置了HSI的几个参数为0.3,PCA噪声为0.01
DA3时在DA2后，将图片的颜色clip导（0,1）之间（动机时创建更符合人感官的自然图片数据）

五个网络按照各自的精度加权求和作为最后的结果，就有了0.9688的效果。

根据各位大佬们的想法，做的一些小实验，一点小小的总结，希望能帮其他几位坛友提供新idea
