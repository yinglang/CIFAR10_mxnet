from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon.model_zoo import vision as model
from mxnet.gluon import nn
from mxnet.gluon.data import vision
import numpy as np
import mxnet as mx
from cifar10_utils import show_images, parse_log, show_log, update
from mxnet.gluon.data.dataset import ArrayDataset
from dataset import *
from netlib import *

"""
data argument;data loader
"""
def _transform_test(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), mean=np.array([0.4914, 0.4822, 0.4465]),
                                   std=np.array([0.2023, 0.1994, 0.2010]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1))
    return im, nd.array([label]).astype('float32')


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


def load_all_data_label(pathes):
    all_data, all_label = None, None
    for path in pathes:
        data, label = nd.load(path)
        label = label.reshape((-1,)).astype('float32')
        if all_data is None:
            all_data, all_label = data, label
        else:
            all_data = nd.concat(all_data, data, dim=0)
            all_label = nd.concat(all_label, label, dim=0)
    return all_data, all_label

arrayds_dir = '/home/hui/dataset/CIFAR10/arraydataset/'
_origin_data_array, _origin_label_array = None, None
def data_loader2(batch_size=[128, 128, 128], transform=[_transform_test, transform_train_DA1, _transform_test], num_workers=[1, 2, 2], origin_shuffle=False):
    global _origin_data_array, _origin_label_array
    if _origin_data_array is None:
        _origin_data_array, _origin_label_array = load_all_data_label([arrayds_dir + 'origin.ndarray'])
    origin_data_array, origin_label_array = _origin_data_array, _origin_label_array
    origin_ds = MyArrayDataset2([origin_data_array, origin_label_array], transform=transform[0])
    origin_data = gluon.data.DataLoader(origin_ds, batch_size[0], shuffle=origin_shuffle, last_batch='keep', num_workers=num_workers[0])
    
    datas = [[origin_data_array, origin_data_array.copy()], [origin_label_array, origin_label_array.copy()]]
    train_ds = MyArrayDataset2(datas, transform=transform[1])
    train_data = gluon.data.DataLoader(train_ds, batch_size[1], shuffle=True, last_batch='keep', num_workers=num_workers[1])
    
    test_ds = gluon.data.vision.datasets.CIFAR10(root='~/.mxnet/datasets/cifar10', train=False, transform=transform[2])
    test_data = gluon.data.DataLoader(test_ds, batch_size[2], shuffle=False, last_batch='keep', num_workers=num_workers[2])
    return (origin_data, train_data, test_data), (origin_ds, train_ds, test_ds)

def mixup(x1, y1, x2, y2, alpha, num_class):
    y1 = nd.one_hot(y1, num_class)
    y2 = nd.one_hot(y2, num_class)
    
    lam = np.random.beta(alpha, alpha)
    x = lam * x1 + (1 - lam) * x2
    y = lam * y1 + (1 - lam) * y2
    return x, y

"""
backgrad data generate
"""
def show_data(data, clip=True):
    images = inv_normalize(data, clip=clip)
    show_images(images, clip=clip)
    
def get_soft_label(y, num_class, soft_label_th):
    y = y.reshape((-1,))
    ny = (y.one_hot(num_class) + soft_label_th / num_class)
    y = y.asnumpy()
    ny[range(y.shape[0]), y.astype('int32')] -= soft_label_th
    return nd.array(ny)
   
def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))
    
def statistic(diffs):
    t = np.mean(diffs), np.max(diffs), np.min(diffs), np.std(diffs)
    return t
    
def inv_normalize(data, mean=None, std=None, clip=True, asnumpy=True):
    if mean is None: mean=nd.array([0.4914, 0.4822, 0.4465])
    if std is None: std=nd.array([0.2023, 0.1994, 0.2010])
    if asnumpy: 
        data, mean, std = data.asnumpy(), mean.asnumpy(), std.asnumpy()
    images = data.transpose((0, 2, 3, 1))
    images = images * std + mean
    images = images.transpose((0, 3, 1, 2)) * 255
    if clip: 
        images = images.clip(0, 255)
    return images

class BNControl():
    """
        only support renet18 by me now.
    """
    @staticmethod
    def collect_BN(blocks):
        BN = []
        for blk in blocks:
            _type = str(blk).split('(')[0]
            if _type == 'BatchNorm':
                BN.append(blk)
            elif _type == 'Residual':
                BN.extend([blk.bn1, blk.bn2])
        return BN
    
    def __init__(self, blocks, use_batch=True):
        self.bns = BNControl.collect_BN(blocks)
        self.use_batch = use_batch
        self.data_list = []
        
    def store(self):
        if self.use_batch: # use batch data and no change running mean/std
            if len(self.data_list) == 0:
                for i, bn in enumerate(self.bns):
                    self.data_list.append(bn.params.get('running_mean').data().copy())
                    self.data_list.append(bn.params.get('running_var').data().copy())
            else:
                for i, bn in enumerate(self.bns):
                    self.data_list[2*i][:] = bn.params.get('running_mean').data()
                    self.data_list[2*i+1][:] = bn.params.get('running_var').data()
        else: # no use batch data and no change running mean/std
            for i, bn in enumerate(self.bns):
                self.data_list.append(bn._kwargs['use_global_stats'])
                bn._kwargs['use_global_stats'] = True
        
    def load(self):
        if self.use_batch:
            for i in range(len(self.bns)):
                bn, mean, std = self.bns[i], self.data_list[2*i], self.data_list[2*i+1]
                bn.params.get('running_mean').set_data(mean)
                bn.params.get('running_var').set_data(std)
        else:
            for i in range(len(self.bns)):
                bn, data = self.bns[i], self.data_list[i]
                bn._kwargs['use_global_stats'] = data
                

class BackGradDataGenerator:
    """
        just a set of many static function
    """
    # 2. function to generate back_grad data
    
    @staticmethod
    def SGD(data, lr):
        # print nd.mean(data.grad).asscalar()
        data[:, :, :, :] = data - data.grad * lr
    
    @staticmethod
    def generate_backgrad_data(net, data, label, max_iters=60, lr=0.1, iter_log=False, clip=False, 
                               loss_f=gluon.loss.SoftmaxCrossEntropyLoss(), bn_control=None, sgd=None, threshold=None):
        """
            data is better in cpu, if data in ctx(global var), the returned backgrad_data is shallow copy of data.
        """
        context = data.context
#         if str(context)[:3] != "cpu":
#             print "warring: data was not in CPU, the returned backgrad_data is shallow copy of data."
        if bn_control is not None:
            bn_control.store()
        if sgd is None:
            sgd = BackGradDataGenerator.SGD

        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        for iters in range(1, max_iters+1):
            with autograd.record():
                data.attach_grad()
                output = net(data)
                loss = -loss_f(output, label)
            loss.backward()
            mean_loss = nd.mean(loss).asscalar()     # reduce will make memory release

            if iter_log and iters % 50 == 0:
                show_data(data[:5], clip)
                #print data[0, 0, :2, :10]
                #print data.grad[0, 0, :2, :10]
            if iter_log and iters % 5 == 0:
                print 'iter:', iters, 'loss:', mean_loss

            sgd(data, lr)
        if threshold is not None:
            for i in range(3):
                data[:, i, :, :] = data[:, i, :, :].clip(threshold[0, i].asscalar(), threshold[1, i].asscalar())
        if bn_control is not None:
            bn_control.load()
            
        return data.as_in_context(context), (loss.as_in_context(context), mean_loss, )
    
    @staticmethod
    def MSE_constraint(mse, same_rate=False):
        def constraint(backgrad_data, data, diff, _loss, MSE, SNR):
            if same_rate:
                rate = nd.mean(nd.sqrt(mse / MSE))
                backgrad_data[:, :, :, :] = data + ((backgrad_data - data) * rate)
            else:
                backgrad_data[:, :, :, :] = data + ((backgrad_data - data).transpose((1, 2, 3, 0))
                                                * (nd.sqrt(mse / MSE))).transpose((3, 0, 1, 2))
        return constraint
    
    @staticmethod
    def generate_backgrad_data_constraint(net, data, label, max_iters=60, lr=0.1, iter_log=False, clip=False, 
                               loss_f=gluon.loss.SoftmaxCrossEntropyLoss(), bn_control=None, post_deal=None, 
                                          sgd=None, threshold=None):
        backgrad_data = data.as_in_context(mx.cpu()) if post_deal is not None else data
        backgrad_data, (_loss, _bloss, ) = BackGradDataGenerator.generate_backgrad_data(net, backgrad_data, label, 
                                            max_iters, lr, iter_log, clip, loss_f, bn_control, sgd, None)
        
        if post_deal is not None:
            tmp = (backgrad_data - data) ** 2
            diff = nd.sqrt(nd.sum(tmp, axis=0, exclude=True))
            MSE = nd.mean(tmp, axis=0, exclude=True)
            Savg = nd.mean((data) ** 2, axis=0, exclude=True)
            # SNR = 10 * nd.log10(Savg / (MSE))
            SNR = 10 * nd.log10(Savg / (MSE + EPS))
            post_deal(backgrad_data, data, diff, _loss, MSE, SNR)
            
        if threshold is not None:
            for i in range(3):
                backgrad_data[:, i, :, :] = backgrad_data[:, i, :, :].clip(threshold[0, i].asscalar(), threshold[1, i].asscalar())
                
        return backgrad_data, (_loss, _bloss, ) 
            
    @staticmethod
    def generate_data_for_out(net, origin_data, max_iters=10, lr=0.1,
                            use_batch_mean_std=False, use_statistic=True, show_per_iters=None,
                            loss_f = gluon.loss.SoftmaxCrossEntropyLoss(), out_data=None, begin_index=0, 
                            bn_control=None, post_deal=None, sgd=None, threshold=None):
        out_idx, iters = begin_index, 0
        diffs, losses, MSEs, SNRs, bloss = None, None, None, None, 0
        
        if bn_control is None:
            bn_backup = BNControl(net.net, use_batch_mean_std) # to avoid update moving_mean/std when generate image
            bn_backup.store()
        for data, label in origin_data:
            backgrad_data, (_loss, _bloss, ) = BackGradDataGenerator.generate_backgrad_data_constraint(
                net, data, label, max_iters=max_iters, lr=lr, iter_log=False, clip=False, loss_f=loss_f, 
                bn_control=bn_control, post_deal=post_deal, sgd=sgd, threshold=threshold)
            
            tmp = (backgrad_data - data) ** 2
            diff = nd.sqrt(nd.sum(tmp, axis=0, exclude=True))
            MSE = nd.mean(tmp, axis=0, exclude=True)
            Savg = nd.mean((data) ** 2, axis=0, exclude=True)
            # SNR = 10 * nd.log10(Savg / (MSE))
            SNR = 10 * nd.log10(Savg / (MSE + EPS))
            if sgd is not None or post_deal is not None: # new change get new loss
                _loss = loss_f(net(backgrad_data.as_in_context(ctx)), label.as_in_context(ctx))
                _bloss = nd.mean(_loss).asscalar()
            bloss += _bloss

            if diffs is None: 
                diffs, losses, MSEs, SNRs = diff, _loss, MSE, SNR
            else:
                diffs, losses = nd.concat(diffs, diff, dim=0), nd.concat(losses, _loss, dim=0)
                MSEs, SNRs = nd.concat(MSEs, MSE, dim=0), nd.concat(SNRs, SNR, dim=0)
                
            # must copy to cpu, or will make gpu memory leak(not release)
            backgrad_data = inv_normalize(backgrad_data, clip=False, asnumpy=False)
            out_data[out_idx:out_idx+data.shape[0], :, :, :] = backgrad_data.transpose((0, 2, 3, 1)).as_in_context(mx.cpu())
            out_idx += data.shape[0]
                    
            if show_per_iters is not None and iters % show_per_iters == 0:
                show_images(inv_normalize(data[np.array(range(0, 25, 5)) % data.shape[0]], clip=False), clip=False)
                show_images(backgrad_data[np.array(range(0, 25, 5)) % data.shape[0]].asnumpy(), clip=False)
            iters += 1
        if bn_control is None:
            bn_backup.load()
        
        diffs, losses, MSEs, SNRs = diffs.asnumpy(), losses.asnumpy(), MSEs.asnumpy(), SNRs.asnumpy()
        if use_statistic:
            return statistic(diffs), statistic(losses), statistic(MSEs), statistic(SNRs)
        else:
            return diffs, losses, MSEs, SNRs    
        
BGG = BackGradDataGenerator

"""
train
"""
import datetime
import utils
import sys
from random import random
from mxnet import gluon, nd

def abs_mean(W):
    return nd.mean(nd.abs(W)).asscalar()

def in_list(e, l):
    for i in l:
        if i == e:
            return True
    else:
        return False
    
    
class TrainPipeline(object):
    def __init__(self, net, train_data, valid_data, start_epoch, num_epochs, policy, ctx, w_key, trainers=None, 
                 output_file=None, verbose=False, loss_f=gluon.loss.SoftmaxCrossEntropyLoss(), mixup_alpha=None,
                 back_grad_args=None):
        self.net, self.train_data, self.valid_data, self.start_epoch = net, train_data, valid_data, start_epoch
        self.num_epochs, self.policy, self.ctx, self.w_key, self.trainers = num_epochs, policy, ctx, w_key, trainers
        self.output_file, self.verbose, self.loss_f, self.mixup_alpha = output_file, verbose, loss_f, mixup_alpha
        self.back_grad_args = back_grad_args
        
    def initialize(self):
        """
            invoke before train
            1. reset output file
            2. set prev_time for cal cost time
            3. init trainers
            4. verbose set True print test_acc in valid_data
        """
        if self.output_file is None:
            self.output_file = sys.stdout
            self.stdout = sys.stdout
        else:
            self.output_file = open(self.output_file, "w")
            self.stdout = sys.stdout
            sys.stdout = self.output_file
       
        self.prev_time = datetime.datetime.now()

        if self.verbose:
            print " #", utils.evaluate_accuracy(self.valid_data, self.net, self.ctx)
            
        if self.trainers is None:
            self.trainers = [gluon.Trainer(self.net.collect_params(), 'sgd', 
                                  {'learning_rate': self.policy['lr'], 'momentum': 0.9, 'wd': self.policy['wd']})]
    
    def after_epoch(self, epoch, train_loss, train_acc):
        """
            invoke after every epoch of train
            1. cal and print cost time the epoch
            2. print acc/loss info
            3. print lr
            4. update lr
        """
        # log info
        self.cur_time = datetime.datetime.now()
        h, remainder = divmod((self.cur_time - self.prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)

        train_loss /= len(self.train_data)
        train_acc /= len(self.train_data)
        if train_acc < 1e-6:
            train_acc = utils.evaluate_accuracy(self.train_data, self.net, self.ctx)

        if self.valid_data is not None:
            valid_acc = utils.evaluate_accuracy(self.valid_data, self.net, self.ctx)
            epoch_str = ("epoch %d, loss %.5f, train_acc %.4f, valid_acc %.4f" 
                         % (epoch, train_loss, train_acc, valid_acc))
        else:
            epoch_str = ("epoch %d, loss %.5f, train_acc %.4f"
                        % (epoch, train_loss, train_acc))
        
        self.prev_time = self.cur_time
        
        self.output_file.write(epoch_str + ", " + time_str + ",lr " + str([trainer.learning_rate for trainer in self.trainers]) + "\n")
        self.output_file.flush()  # to disk only when flush or close
        
        if in_list(epoch+1, self.policy['lr_period']):
            for trainer in self.trainers:
                trainer.set_learning_rate(trainer.learning_rate * self.policy['lr_decay'])
    
    def after_iter(self, i, _loss, _acc):
        """
            invoke after every iteration
            1. print iter losss and acc
            2. print weight and grad for every iter
        """
        if self.verbose:
            print " # iter", i,
            print "loss %.5f" % _loss, "acc %.5f" % _acc,
            print "w (",
            for k in self.w_key:
                w = self.net.collect_params()[k]
                print "%.5f, " % abs_mean(w.data()),
            print ") g (",
            for k in self.w_key:
                w = self.net.collect_params()[k]
                print "%.5f, " % abs_mean(w.grad()),
            print ")"
        
    def run(self):
        self.initialize()
        for epoch in range(self.start_epoch, self.start_epoch+self.num_epochs):
            train_loss, train_acc, i = 0., 0., 0
            for data, label in self.train_data:
                with autograd.record():
                    data, label = data.as_in_context(ctx), label.as_in_context(ctx)
                    output = self.net(data)
                    loss = self.loss_f(output, label)
                loss.backward()
                for trainer in self.trainers:
                    trainer.step(data.shape[0])
                
                _loss = nd.mean(loss).asscalar() 
                _acc = utils.accuracy(output, label)
                train_loss += _loss
                train_acc += _acc
                
                self.after_iter(i, _loss, _acc)
                i += 1
            self.after_epoch(epoch, train_loss, train_acc)
        
        if self.output_file == sys.stdout:
            sys.stdout = self.stdout
            self.output_file.close()

class BackGradTrain(TrainPipeline):
    """
        1. add loss and diff/MSE/SNR to evaluate and control generate images
        2. BN layer: set use_gloabl_stats=True to use global mean/std and stop moving mean cal when genarate images, and set False after geneate.
        3. using DA after genereate images, means no DA when generate images
        4.
    """
    def __init__(self, net, train_data, valid_data, start_epoch, num_epochs, policy, ctx, w_key, trainers=None, 
                 output_file=None, verbose=False, loss_f=gluon.loss.SoftmaxCrossEntropyLoss(), mixup_alpha=None,
                 back_grad_args=None):
        super(BackGradTrain, self).__init__(net, train_data, valid_data, start_epoch, num_epochs, policy, ctx,
                                            w_key, trainers, output_file, verbose, loss_f, mixup_alpha, back_grad_args) 
        
    def initialize(self):
        super(BackGradTrain, self).initialize()
        if self.back_grad_args is not None:
            self.back_grad_args['num_epoch_per_round'] = self.back_grad_args.get('num_epoch_per_round', 1)
            self.back_grad_args['verbose'] = self.back_grad_args.get('verbose', False)
            self.back_grad_args['post_deal'] = self.back_grad_args.get('post_deal', None)
            self.back_grad_args['sgd'] = self.back_grad_args.get('sgd', None)
            self.back_grad_args['threshold'] = self.back_grad_args.get('threshold', None)
    
    def after_epoch(self, epoch, train_loss, train_acc):
        super(BackGradTrain, self).after_epoch(epoch, train_loss, train_acc)
    
    def after_iter(self, i, _loss, _acc):
        super(BackGradTrain, self).after_iter(i, _loss, _acc)
    
    def statistic(self, diffs):
        t = np.mean(diffs), np.max(diffs), np.min(diffs), np.std(diffs)
        return t
      
    def run(self):
        self.initialize()
        if self.back_grad_args is not None:
            num_epoch_per_round = self.back_grad_args['num_epoch_per_round']
            max_iters, lr = self.back_grad_args['max_iters'], self.back_grad_args['lr']
            origin_data, b_verbose = self.back_grad_args['origin_data'], self.back_grad_args['verbose']
            use_batch_mean_std, train_ds = self.back_grad_args['use_batch_mean_std'], self.back_grad_args['train_ds']
            post_deal, sgd, threshold = self.back_grad_args['post_deal'], self.back_grad_args['sgd'], self.back_grad_args['threshold']
        epoch_round = 0
        bn_backup = BNControl(self.net.net, use_batch_mean_std)
        
        for epoch in range(self.start_epoch, self.start_epoch+self.num_epochs):
            # 1. train net
            train_loss, train_acc, i = 0., 0., 0
            for data, label in self.train_data:
                with autograd.record():
                    data, label = data.as_in_context(ctx), label.as_in_context(ctx)
                    output = self.net(data)
                    loss = self.loss_f(output, label)
                loss.backward()
                for trainer in self.trainers:
                    trainer.step(data.shape[0])

                _loss = nd.mean(loss).asscalar() 
                _acc = utils.accuracy(output, label)
                train_loss += _loss
                train_acc += _acc

                self.after_iter(i, _loss, _acc)
                i += 1
                
            # 2. generate new backgrad data
            if self.back_grad_args is not None and (epoch - self.start_epoch + 1) % num_epoch_per_round == 0:
                diff, gloss, MSE, SNR = BGG.generate_data_for_out(self.net, origin_data, max_iters, lr, use_batch_mean_std,
                                            True, None, loss_f, out_data=train_ds._data[0][1], begin_index=0, bn_control=bn_backup,
                                                                 post_deal=post_deal, sgd=sgd, threshold=threshold)
                
                if b_verbose:
                    print "# epoch_round, generate data info (mean, max, min, std):", epoch_round
                    print "#     diff:", diff
                    print "#     loss:", gloss
                    print "#     MSE :", MSE
                    print "#     SNR :", SNR
                epoch_round += 1
                
            self.after_epoch(epoch, train_loss, train_acc)
            
        if self.output_file == sys.stdout:
            sys.stdout = self.stdout
            self.output_file.close()
            
    def run2(self, prob=0.5):
        self.initialize()
        if self.back_grad_args is not None:
            num_epoch_per_round = self.back_grad_args['num_epoch_per_round']
            max_iters, lr = self.back_grad_args['max_iters'], self.back_grad_args['lr']
            origin_data, b_verbose = self.back_grad_args['origin_data'], self.back_grad_args['verbose']
            use_batch_mean_std, train_ds = self.back_grad_args['use_batch_mean_std'], self.back_grad_args['train_ds']
            post_deal, sgd, threshold = self.back_grad_args['post_deal'], self.back_grad_args['sgd'], self.back_grad_args['threshold']
        epoch_round = 0
        
        bn_backup = BNControl(self.net.net, use_batch_mean_std) # to avoid update moving_mean/std when generate image
        
        for epoch in range(self.start_epoch, self.start_epoch+self.num_epochs):
            # 1. train net
            train_loss, train_acc, i, bloss = 0., 0., 0, 0
            for data, label in self.train_data:
                if i > 0 and random() < prob:
                    # origin_d = data
                    data,( _, _bloss) = BGG.generate_backgrad_data_constraint(self.net, data.copy(), label, max_iters,
                                                lr, False, False, loss_f, bn_backup, post_deal, sgd, threshold)
                    bloss += _bloss
                data, label = data.as_in_context(ctx), label.as_in_context(ctx)
                with autograd.record():
                    output = self.net(data)
                    loss = self.loss_f(output, label)
                loss.backward()
                for trainer in self.trainers:
                    trainer.step(data.shape[0])

                _loss = nd.mean(loss).asscalar() 
                _acc = utils.accuracy(output, label)
                train_loss += _loss
                train_acc += _acc

                self.after_iter(i, _loss, _acc)
                i += 1
                
            if b_verbose:
                print "bloss %.4f, " % (bloss / len(self.train_data)),
            epoch_round += 1
                
            self.after_epoch(epoch, train_loss, train_acc)
            
        if self.output_file == sys.stdout:
            sys.stdout = self.stdout
            self.output_file.close()