import numpy as np
from math import ceil

import utils.precision as p
import train.data as data
import utils.paths as paths
import utils.utils as u

"""
Class for training a tac. The only public function is:

    # train()

Uses the following interface from the TacGraph class (see tacgraph.py):

    # init_training()
    # reset_optimizer()
    # assign_random_weights()
    # save_current_weights()
    # restore_saved_weights()
    # optimize_loss()
    # compute_metric()
    # end_training()
    
Uses the following functions from data (see data.py)

    # random_split()
    # random_data_batches()
    # data_batches()
"""

class Trainer:

    def __init__(self,tac):
        self.tac = tac
        assert tac.tac_graph.finalized and tac.tac_graph.trainable
                
        # supported losses and mertics
        self.loss_types         = ('CE', 'MSE')
        self.metric_types       = ('CE', 'CA', 'MSE')
        
        # set when initializing training
        self.loss_type          = None   # loss being optimized
        self.metric_type        = None   # metric for validation data
        self.metric_comp        = None   # > or < (maximizing or minimizing metric) 
        self.metric_best_value  = None   # best value obtained for metric so far
        self.metric_target      = None   # stop training if metric reaches this value
        self.metric_delta_min   = None   # metric change that is an ignorable improvement
        
        # updated during training
        self.patience_wait      = None   # number of epochs since last improvement
        
        # training parameters for gradient descent, including stopping criteria
        # these may need to be tuned for different problems
        self.split_ratio        =  .200  # percentage of validation data
        self.epochs_count       =   500  # max number of epochs during training
        self.weight_tries       =    5  # number of random initial weights to try
        self.burn_in            =     3  # number epochs before starting learning schedule
        self.initial_lr         =  .050  # initial lr after burn in epochs
        self.decay_epochs       =   100  # number of epochs to decay lr to final value
        self.final_lr           =  .005  # learning rate should not drop below this
        self.power              =     3  # 1 is linear decay (larger is faster decay)
        self.patience_max_wait  =    20  # epochs before stopping GD if no improvement
        self.CA_target          = .9999  # stop if this classification accuracy is reached
        self.CA_delta_min       = .0001  # min change for classification improvement
        self.CE_target          =  1e-4  # stop if this cross entropy is reached
        self.CE_delta_min       = -1e-5  # min change for cross entropy improvement
        self.MSE_target         =  1e-6  # stop if this mean-squared error is reached
        self.MSE_delta_min      = -1e-7  # min change for mean-squared improvement
   
        
    """ interface of trainer """

    # if batch_size None, batch size is computed based on various factors (see batch.py)
    def train(self,evidence,marginals,loss_type,metric_type,batch_size,fname=None):
        assert loss_type in self.loss_types and metric_type in self.metric_types
        
        # split data into training and validation (after randonly shuffling it)
        t_data, v_data = data.random_split(evidence,marginals,self.split_ratio) 
        
        t_evidence, t_marginals, t_size = t_data
        v_evidence, v_marginals, v_size = v_data
        
        # batch memory is based on tf graphs for tac, optimizer and metrics     
        tac_graph         = self.tac.tac_graph
        circuit_type      = self.tac.circuit_type
        parameter_count   = tac_graph.parameter_count 
        fixed_zeros_count = tac_graph.fixed_zeros_count
        batch_count       = ceil(t_size/batch_size)
           
        u.show(f'  loss: {loss_type}, metric: {metric_type}\n'
               f'  data: training {t_size}, validation {v_size}\n'
               f'  batch: size {batch_size}, count {batch_count}\n'
               f'  trainable parameters  {parameter_count}\n'
               f'  fixed zero parameters {fixed_zeros_count}')
        
        # initialize trainer and tac_graph optimizer
        batch_size = self.__init_training(loss_type,metric_type,t_size,batch_size)  
                
        # initialize the tac weights (try a few random weights and pick best)
        weights_epochs = self.__find_initial_weights(t_evidence,t_marginals,
                            v_evidence,v_marginals,batch_size)
        # train
        for epoch in range(self.epochs_count):
            
            # optimize loss on training data, compute metric on validation data
            t_loss, lr = self.__optimize_loss(loss_type,t_evidence,t_marginals,batch_size,epoch)
            v_metric   = self.__compute_metric(metric_type,v_evidence,v_marginals,batch_size)
            
            # logging for tensorboard
            tac_graph.log(epoch,t_loss,v_metric,lr) 
            
            # main control
            stop, save, event = self.__analyze_epoch(v_metric,epoch)
            if stop or event:
                u.show((f'\r  epoch {epoch:5d}: t_loss {t_loss:.8f}, '
                        f'v_metric {v_metric:.8f}, lr {lr:.4f}{event}'))
            if save: tac_graph.save_current_weights()
            if stop: break 
        
        # restore learned weights and write them to file
        if fname is None:
            fname = paths.cpts / f'{circuit_type}.txt'
        u.show(f'  writing learned CPTs to {fname}')
        tac_graph.end_training(fname)  
     
        return weights_epochs+epoch+1 # total number of epochs we performed
     
     
    """ analyzes outcome of epoch and returns possible actions and status events """

    # decides whether we improved compared to last epoch and whether we should stop GD
    def __analyze_epoch(self,metric_current_value,epoch):
       
        stop  = False
        save  = False
        event = ''
        
        best  = self.metric_best_value
        curr  = metric_current_value
        
        self.patience_wait += 1
        if epoch+1 == self.epochs_count: # this is the last epoch, we are done
            stop  = True
            save  = False
            event = 'reached last epoch'
        elif self.metric_comp(curr,self.metric_target): # good enough metric, take it
            stop  = True
            save  = True
            event = 'reached target metric'
        elif self.metric_comp(curr-self.metric_delta_min,best): # improved metric
            self.metric_best_value = curr
            self.patience_wait     = 0
            stop  = False
            save  = True
            event = 'improved metric'
        elif self.patience_wait == self.patience_max_wait: # waited enough for improvement
            stop  = True
            save  = False
            event = 'patience ran out'
                 
        if event: event = ', events: ' + event

        return stop, save, event
       
       
    """ initialize before training starts """
     
    # batch_size could be None in which case it will be computed based on some factors
    def __init_training(self,loss_type,metric_type,t_size,batch_size):
         
        self.patience_wait = 0
        self.loss_type     = loss_type
        self.metric_type   = metric_type
        
        if metric_type == 'CA':    # classification accuracy
            self.metric_comp       = np.greater_equal # maximize
            self.metric_best_value = 0.               # worst value CA
            self.metric_target     = self.CA_target
            self.metric_delta_min  = self.CA_delta_min
        elif metric_type == 'CE':  # cross entropy
            self.metric_comp       = np.less_equal    # minimize
            self.metric_best_value = np.inf           # worst CE
            self.metric_target     = self.CE_target
            self.metric_delta_min  = self.CE_delta_min 
        else:                      # MSE
            self.metric_comp       = np.less_equal    # minimize
            self.metric_best_value = np.inf           # worst MSE
            self.metric_target     = self.MSE_target
            self.metric_delta_min  = self.MSE_delta_min
        
        # initialize tac graph optimizer: pass on schedule for learning rate (lr)
        # schedule starts after burn_in epochs, lr changes after each epoch thereafter
        batch_count = ceil(t_size/batch_size)
        lr_schedule = (self.initial_lr,self.decay_epochs,self.final_lr,self.power)
        self.tac.tac_graph.init_training(self.burn_in,batch_count,lr_schedule)

        return batch_size


    """ search for initial weights by trying a few random ones """

    def __find_initial_weights(self,t_evidence,t_marginals,v_evidence,v_marginals,batch_size):
        #u.show(f'  optimizer warming up...\r',end='',flush=True)
        u.show(f'  finding initial weights (starting with uniform):',end='',flush=True)
        
        loss_type   = self.loss_type
        metric_type = self.metric_type
        tac_graph   = self.tac.tac_graph
        best_loss   = None
        best_metric = self.metric_best_value # initialized to worst possible value
        epochs      = 0 # number of epochs we will try
        
        # weights already set to uniform in ops.py so we will try these first        
        for i in range(self.weight_tries):
            epochs += 2 # we do a two-step lookahead 
            
            # loss and metric before trying to improve current weights            
            pre_loss   = self.__compute_metric(loss_type,t_evidence,t_marginals,batch_size)
            pre_metric = self.__compute_metric(metric_type,v_evidence,v_marginals,batch_size)
            
            # loss and metric after improving current weights using GD
            for _ in range(2): # two-step lookahead
                loss,_ = self.__optimize_loss(loss_type,t_evidence,t_marginals,batch_size,None)          
            metric = self.__compute_metric(metric_type,v_evidence,v_marginals,batch_size)
            
            # printing details that are helpful to sanity check behavior
            u.show(f'\n    t_loss {pre_loss:11.8f} -> {loss:11.8f}, ',
                   f'v_metric {pre_metric:11.8f} -> {metric:11.8f}',end='',flush=True) 
            
            # reset optimizer before quitting in case we decide we are done  
            tac_graph.reset_optimizer()
            
            # see if the current weights improve on the previous ones
            assert i != 0 or self.metric_comp(metric,best_metric)
            if self.metric_comp(metric,best_metric): 
                best_loss   = loss
                best_metric = metric
                tac_graph.save_current_weights()
                if self.metric_comp(metric,self.metric_target): 
                    break # found good-enough initial weights
            
            # try a new set of random weights if this is not the last iteration
            if i < self.weight_tries-1:
                tac_graph.assign_random_weights()
            
        u.show(f'\n  starting at: t_loss {best_loss:.8f}, v_metric {best_metric:.8f}, '
               f'found after {epochs} epochs',flush=True)
           
        # use the best found weights
        self.metric_best_value = best_metric
        tac_graph.restore_saved_weights()
        return epochs
        
        
    """ optimizing loss and computing metric: makes call to tac graph functions """
    
    # optimizes loss over training data: one pass over data batches
    def __optimize_loss(self,loss_type,evidence,marginals,batch_size,epoch):
        batches, batch_count = data.random_data_batches(evidence,marginals,batch_size)
        loss = 0
        for batch_index, (evd_batch, mar_batch) in enumerate(batches):
            bloss, lr = self.tac.tac_graph.optimize_loss(loss_type,evd_batch,mar_batch)
            loss     += bloss * len(mar_batch)
            if epoch: self.__print_progress(epoch,batch_index,batch_count,lr)
        size = len(marginals) # sum of all batch sizes (last one may be smaller)
        return loss/size, lr  # average weighted by batch size
        
    def __compute_metric(self,metric_type,evidence,marginals,batch_size):
        batches, _ = data.data_batches(evidence,marginals,batch_size)
        metric = 0
        for evd_batch, mar_batch in batches:
            bmetric = self.tac.tac_graph.compute_metric(metric_type,evd_batch,mar_batch)
            metric += bmetric * len(mar_batch)
        size = len(marginals) # sum of all batch sizes (last one may be smaller)
        return metric/size    # average weighted by batch size
        
    # prints percentage of processed training batches
    # e: epoch number (1...)
    # i: batch index (0...n-1)
    # n: batch count
    def __print_progress(self,e,i,n,lr):
        p = 100*(i+1)//n
        u.show(f'  epoch {e:5d}:{p:4d}%   lr {lr:.5f}',end='',flush=True)
        u.show((b'\x08' * 32).decode(),end='',flush=True)