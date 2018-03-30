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