```{.python .input  n=1}
from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon.data import vision
import numpy as np
import random
import mxnet as mx
from netlib import *
ctx = mx.gpu(1)
```

```{.python .input  n=6}
"""
data loader
"""
data_dir = "/root/Workspace/data/CIFAR10_kaggle/train_valid_test/"

def _transform_test(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), mean=np.array([0.4914, 0.4822, 0.4465]),
                                   std=np.array([0.2023, 0.1994, 0.2010]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1))
    return im, nd.array([label]).astype('float32')


def data_loader(batch_size, transform_train, transform_test=None):
    if transform_train is None:
        transform_train = _transform_train
    if transform_test is None:
        transform_test = _transform_test
        
    # flag=1 mean 3 channel image
    train_ds = vision.ImageFolderDataset(data_dir + 'train', flag=1, transform=transform_train)
    valid_ds = vision.ImageFolderDataset(data_dir + 'valid', flag=1, transform=transform_test)
    train_valid_ds = vision.ImageFolderDataset(data_dir + 'train_valid', flag=1, transform=transform_train)
    test_ds = vision.ImageFolderDataset(data_dir + "test", flag=1, transform=transform_test)

    loader = gluon.data.DataLoader
    train_data = loader(train_ds, batch_size, shuffle=True, last_batch='keep')
    valid_data = loader(valid_ds, batch_size, shuffle=True, last_batch='keep')
    train_valid_data = loader(train_valid_ds, batch_size, shuffle=True, last_batch='keep')
    test_data = loader(test_ds, batch_size, shuffle=False, last_batch='keep')
    return train_data, valid_data, train_valid_data, test_data, test_ds, train_valid_ds
```

```{.python .input  n=3}
"""
data argument
"""
def transform_train_DA1(data, label):
    im = data.asnumpy()
    im = np.pad(im, ((4, 4), (4, 4), (0, 0)), mode='constant', constant_values=0)
    im = nd.array(im, dtype='float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), resize=0, rand_mirror=True,
                                    rand_crop=True,
                                   mean=np.array([0.4914, 0.4822, 0.4465]),
                                   std=np.array([0.2023, 0.1994, 0.2010]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1)) # channel x width x height
    return im, nd.array([label]).astype('float32')


def transform_train_DA2(data, label):
    im = data.astype(np.float32) / 255
    auglist = [image.RandomSizedCropAug(size=(32, 32), min_area=0.49, ratio=(0.5, 2))]
    _aug = image.CreateAugmenter(data_shape=(3, 32, 32), resize=0, 
                                rand_crop=False, rand_resize=False, rand_mirror=True,
                                mean=np.array([0.4914, 0.4822, 0.4465]),
                                std=np.array([0.2023, 0.1994, 0.2010]),
                                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3,
                                pca_noise=0.01, rand_gray=0, inter_method=2)
    auglist.append(image.RandomOrderAug(_aug))
    
    for aug in auglist:
        im = aug(im)
    
    im = nd.transpose(im, (2, 0, 1))
    return (im, nd.array([label]).asscalar().astype('float32'))
    

random_clip_rate = 0.3
def transform_train_DA3(data, label):
    im = data.astype(np.float32) / 255
    auglist = [image.RandomSizedCropAug(size=(32, 32), min_area=0.49, ratio=(0.5, 2))]
    _aug = image.CreateAugmenter(data_shape=(3, 32, 32), resize=0, 
                                rand_crop=False, rand_resize=False, rand_mirror=True,
#                                mean=np.array([0.4914, 0.4822, 0.4465]),
#                                std=np.array([0.2023, 0.1994, 0.2010]),
                                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3,
                                pca_noise=0.01, rand_gray=0, inter_method=2)
    auglist.append(image.RandomOrderAug(_aug))

    for aug in auglist:
        im = aug(im)
        
    if random.random() > random_clip_rate:
        im = im.clip(0, 1)
    _aug = image.ColorNormalizeAug(mean=np.array([0.4914, 0.4822, 0.4465]),
                   std=np.array([0.2023, 0.1994, 0.2010]),)
    im = _aug(im)
    
    im = nd.transpose(im, (2, 0, 1))
    return (im, nd.array([label]).asscalar().astype('float32'))
```

```{.python .input  n=4}
"""
train
"""
import datetime
import utils
import sys

def abs_mean(W):
    return nd.mean(nd.abs(W)).asscalar()

def in_list(e, l):
    for i in l:
        if i == e:
            return True
    else:
        return False

def train(net, train_data, valid_data, num_epochs, lr, lr_period, 
          lr_decay, wd, ctx, w_key, output_file=None, verbose=False, loss_f=gluon.loss.SoftmaxCrossEntropyLoss()):
    if output_file is None:
        output_file = sys.stdout
        stdout = sys.stdout
    else:
        output_file = open(output_file, "w")
        stdout = sys.stdout
        sys.stdout = output_file
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    prev_time = datetime.datetime.now()
    
    if verbose:
        print " #", utils.evaluate_accuracy(valid_data, net, ctx)
    
    i = 0
    for epoch in range(num_epochs):
        train_loss = 0.
        train_acc = 0.
        if in_list(epoch, lr_period):
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for data, label in train_data:
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                loss = loss_f(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            
            _loss = nd.mean(loss).asscalar()
            _acc = utils.accuracy(output, label)
            train_loss += _loss
            train_acc += _acc
            
            if verbose:
                print " # iter", i,
                print "loss %.5f" % _loss, "acc %.5f" % _acc,
                print "w (",
                for k in w_key:
                    w = net.collect_params()[k]
                    print "%.5f, " % abs_mean(w.data()),
                print ") g (",
                for k in w_key:
                    w = net.collect_params()[k]
                    print "%.5f, " % abs_mean(w.grad()),
                print ")"
                i += 1
            
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        
        train_loss /= len(train_data)
        train_acc /= len(train_data)
        
        if valid_data is not None:
            valid_acc = utils.evaluate_accuracy(valid_data, net, ctx)
            epoch_str = ("epoch %d, loss %.5f, train_acc %.4f, valid_acc %.4f" 
                         % (epoch, train_loss, train_acc, valid_acc))
        else:
            epoch_str = ("epoch %d, loss %.5f, train_acc %.4f"
                        % (epoch, train_loss, train_acc))
        prev_time = cur_time
        output_file.write(epoch_str + ", " + time_str + ",lr " + str(trainer.learning_rate) + "\n")
        output_file.flush()  # to disk only when flush or close
    if output_file != stdout:
        sys.stdout = stdout
        output_file.close()
```

### Exp1: res164_v2 + DA1: 0.9529

```{.python .input}
batch_size = 128
transform_train = transform_train_DA1
train_data, valid_data, train_valid_data, test_data, test_ds, train_valid_ds = data_loader(batch_size, transform_train)
net = ResNet164_v2(10)
loss_f = gluon.loss.SoftmaxCrossEntropyLoss()

num_epochs = 200
learning_rate = 0.1
weight_decay = 1e-4
lr_period = [90, 140]
lr_decay=0.1
log_file = None

net.collect_params().initialize(mx.init.Xavier(), ctx=ctx, force_reinit=True)
net.hybridize()
w_key = []
train(net, train_data, valid_data, num_epochs, learning_rate, 
      lr_period, lr_decay, weight_decay, ctx, w_key, log_file, False, loss_f)

net.save_params("v1/models/shelock_resnet_orign")
```

### Exp2:res164_v2 + DA2: 0.9527

```{.python .input}
batch_size = 128
transform_train = transform_train_DA2
train_data, valid_data, train_valid_data, test_data, test_ds, train_valid_ds = data_loader(batch_size, transform_train2)
net = ResNet164_v2(10)
loss_f = gluon.loss.SoftmaxCrossEntropyLoss()

num_epochs = 300
learning_rate = 0.1
weight_decay = 1e-4
lr_period = [150, 225]
lr_decay=0.1
log_file = None

net.collect_params().initialize(mx.init.Xavier(), ctx=ctx, force_reinit=True)
net.hybridize()
w_key = []
train(net, train_data, valid_data, num_epochs, learning_rate, 
      lr_period, lr_decay, weight_decay, ctx, w_key, log_file, False, loss_f)
net.save_params("v1/models/resnet164_e300")
```

### Exp3: res164_v2 + focal loss + DA3: 0.9540

```{.python .input}
batch_size = 128
tranform_train = transform_train_DA3
train_data, valid_data, train_valid_data, test_data, test_ds, train_valid_ds = data_loader(batch_size, transform_train)
net = ResNet164_v2(10)
loss_f = FocalLoss()

num_epochs = 255
learning_rate = 0.1
weight_decay = 1e-4
lr_period = [150, 225]
lr_decay=0.1
log_file = None

net.collect_params().initialize(mx.init.Xavier(), ctx=ctx, force_reinit=True)
net.hybridize()
w_key = []
train(net, train_valid_data, None, num_epochs, learning_rate, 
      lr_period, lr_decay, weight_decay, ctx, w_key, log_file, False, loss_f)
net.save_params("models/res164__2_e255_focal_clip_all_data")
```

### Exp4: res164_v2 + focal loss + DA3 + only train_data: 0.9506

```{.python .input}
batch_size = 128
transform_train = transform_train_DA3
train_data, valid_data, train_valid_data, test_data, test_ds, train_valid_ds = data_loader(batch_size, transform_train)
net = ResNet164_v2(10)
loss_f = FocalLoss()

num_epochs = 255
learning_rate = 0.1
weight_decay = 1e-4
lr_period = [150, 225]
lr_decay=0.1
log_file = None

net.collect_params().initialize(mx.init.Xavier(), ctx=ctx, force_reinit=True)
net.hybridize()
w_key = []
train(net, train_data, valid_data, num_epochs, learning_rate, 
      lr_period, lr_decay, weight_decay, ctx, w_key, log_file, False, loss_f)
net.save_params("models/resnet164_e0-255_focal_clip")
```

### Exp5: sherlock_densenet: 0.9539

```{.python .input  n=9}
batch_size = 128
transform_train = transform_train_DA1
train_data, valid_data, train_valid_data, test_data, test_ds, train_valid_ds = data_loader(batch_size, transform_train)
net = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=10)
loss_f = gluon.loss.SoftmaxCrossEntropyLoss()

num_epochs = 200
learning_rate = 0.1
weight_decay = 1e-4
lr_period = [90, 140]
lr_decay=0.1
log_file = None

net.hybridize()
net.initialize(ctx=ctx)
w_key = []
train(net, train_data, valid_data, num_epochs, learning_rate, lr_period, lr_decay, weight_decay, ctx, w_key, log_file, False, loss_f)
net.save_params("models/shelock_densenet_orign")
```

# merge result

```{.python .input  n=16}
import os
import numpy as np
import pandas as pd

def save_net_result(net, filename, test_data, ctx):
    output = nd.zeros(shape=(300000, 10), ctx=ctx)
    for i, (data, label) in enumerate(test_data):
        output[i*batch_size:i*batch_size+data.shape[0],:] = net(data.as_in_context(ctx))
    nd.save(filename, output)

def test_net(data):
    return data.reshape((data.shape[0], -1))[:, :10]

def save_model_result(model_name, ctx):
    net.load_params("models/" + model_name, ctx=ctx)
    save_net_result(net, "result/" + model_name, test_data, ctx)

model_list = ['resnet164_e255_focal_clip', 'res164__2_e255_focal_clip_all_data', 'resnet164_e300','resnet164_e0-255_focal_clip',
              'res18_9', 
              'log_shelock_densenet', 'shelock_densenet_orign',
              'shelock_resnet_orign']
weight_list = [0.9535, 0.9540, 0.95270, 0.95, 0.93230, 0.9346, 0.9539, 0.95]

net = ResNet164_v2(10)
for model_name in model_list[:4]:
    if not os.path.exists("result/"+model_name):
        save_model_result(model_name, ctx)
        
net = ResNet(10)
for model_name in model_list[4:5]:
    if not os.path.exists("result/"+model_name):
        save_model_result(model_name, ctx)
        
net = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=10)
for model_name in model_list[5:7]:
    if not os.path.exists("result/"+model_name):
        save_model_result(model_name, ctx)
        
net = ResNet164_v2(10)
for model_name in model_list[7:]:
    if not os.path.exists("result/"+model_name):
        save_model_result(model_name, ctx)
```

```{.python .input  n=11}
"""
classfiy test set
"""
import numpy as np
import pandas as pd

train_data, valid_data, train_valid_data, test_data, test_ds, train_valid_ds = data_loader(128, transform_train_DA1)

def mesuare_sum(preds, weight_list=None):
    if weight_list is None:
        weight_list = [1] * len(preds)
    output = preds[0] * weight_list[0]
    for i in range(1, len(preds)):
        output = output + preds[i] * weight_list[i]
    preds = output.argmax(axis=1).astype(int).asnumpy() % 10
    return preds

def mesuare_softmax_sum(preds, weight_list=None):
    if weight_list is None:
        weight_list = [1] * len(preds)
    output = nd.softmax(preds[0], axis=1) * weight_list[0]
    for i in range(1, len(preds)):
        output = output + nd.softmax(preds[i], axis=1) * weight_list[i]
    preds = output.argmax(axis=1).astype(int).asnumpy() % 10
    return preds

def mesuare_biggest(preds, weight_list=None):
    if weight_list is not None:
        for i in range(len(preds)):
            preds[i] = preds[i] * weight_list[i]
    output = nd.concat(*preds, dim=1)
    preds = output.argmax(axis=1).astype(int).asnumpy() % 10
    return preds

model_list = ['res164__2_e255_focal_clip_all_data', 'resnet164_e300', 'resnet164_e0-255_focal_clip',
              'shelock_densenet_orign', 'shelock_resnet_orign']
weight_list = [0.9540, 0.95270, 0.95, 0.9539, 0.95]
#weight_list=None

preds = []
for result_name in model_list:
    preds.append(nd.load("result/"+result_name)[0].as_in_context(ctx))

#preds = mesuare_biggest(preds, weight_list)
#preds = mesuare_sum(preds, weight_list)
preds = mesuare_softmax_sum(preds, weight_list)

sorted_ids = list(range(1, 300000 + 1))
sorted_ids.sort(key=lambda x: str(x))

df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
df.to_csv('submission/concat_5_softmax_sum_weight.csv', index=False)
```
