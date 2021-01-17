import numpy as np
import tensorflow as tf
from math import log2
from datetime import datetime
from functools import reduce
from itertools import count
from random import randint, uniform

import train.data as d
import tensors.ops as ops 
import utils.precision as p
import utils.paths as paths
import utils.utils as u

force_profile = False # can be set by command line 

"""
A class that wraps callable tf graphs that compute marginals, metrics and losses
in addition to optimizing loss.

Provides functions for evaluating, training and profiling a tac.

Interface for evaluating TacGraph:
  # evaluate()
  # compute_metric()
        
Interface for training TacGraph:
  # init_training()
  # reset_optimizer()
  # assign_random_weights()
  # save_current_weights()
  # restore_saved_weights()
  # optimize_loss()
  # compute_metric()
  # end_training()
        
Operations not annotated with @tf.function will be evaluated with eager execution.

Functions with @tf.function annotations are not called directly but compiled into
callable tf graphs (using get_concrete_function()).
"""

# callable tf graphs accept only tensors as input
# utilities for converting evidence and labels (np arrays) to tf tensors
tf_lab = lambda l: tf.constant(l,dtype=p.float)
tf_evd = lambda e: tuple(tf.constant(e_,dtype=p.float) for e_ in e)

class TacGraph:

    def __init__(self,ops_graph,profile=False):
        
        profile                 = profile or force_profile 
        self.ops_graph          = ops_graph
        self.trainable          = ops_graph.trainable        # whether tac is trainable
        self.profile            = profile                    # for saving tac and timings
        self.train_cpt_labels   = ops_graph.train_cpt_labels # for saving cpts to file
        
        self.evidence_variables = None # variables: tac inputs
        self.weight_variables   = None # variables: only trainables in TacGraph
        self.cpt_variables      = None # weight variables grouped by cpt
        
        self.marginal_sepc      = None # shape and dtype of posterior marginal (tac output)
        self.fixed_cpt_tensors  = None # constants: cpts that are not trainable
        self.fixed_zeros_count  = None # number of fixed zeros in weight tensors
        self.parameter_count    = None # number of trainable parameters in weights
        self.size               = None # number of entries in tensors for tac graph
        self.rank               = None # max rank attained by any tensor
        self.binary_rank        = None # rank of largest tensor if dimensions were binary
        self.total_size         = None # size of tac, metrics and losses graphs
        
        self.MAR                = None # callable tf graph, computes marginals
        self.TCPTS              = None # callable tf graph, computes trainable cpts
        self.CE                 = None # callable tf graph, computes cross entropy
        self.CA                 = None # callable tf graph, computes class. accuracy
        self.MSE                = None # callable tf graph, computes mean squared error
        self.OPT                = None # callable tf graph, optimizes loss (one step)
        
        # CE (cross entropy), CA (classification accuracy), MSE (mean-squared error)
        self.loss_types         = ('CE','MSE')
        self.metric_types       = ('CE','CA','MSE')
        assert set(self.loss_types) <= set(self.metric_types)
        
        self.training_ongoing   = False
        self.optimizer          = None # Adam
        self.optimizer_lr       = None # tensor holding learning rate
        self.optimizer_state    = None # initial state of optimizer (for reseting it)
        self.update_lr          = None # function that updates learning rate
        self.loss_fns           = None # maps loss type (str) to loss function
        self.metric_fns         = None # maps metric type (str) to metric function
        self.learned_weights    = None # tuple of np arrays, ordered as weight_variables
        self.finalized          = False
        
        if self.profile:
            stamp          = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.logdir    = paths.tacs / stamp
            self.writer    = tf.summary.create_file_writer(self.logdir)
            self.saved_tac = False   
        
        self.__build_tac_graph(ops_graph)
            
    # "concrete functions" are callable tf graphs, compiled from python functions
    # that have @tf.function decorators.
    
    # Builds TacGraph by compiling callable tf graphs for the tac, metrics and optimizer.
    # The tac graph is obtained by executing OpsGraph operations, which is done in
    # two steps: we first execute operations that create variables, then execute 
    # operations that create other tensors (the former is done outside @tf.function)
    # The graphs for metrics and optimizers are constructed based on @tf.functions
    # specified later in this file.
    
    def __build_tac_graph(self,ops_graph):     
        assert not self.finalized
        u.show('  Constructing TacGraph: tac...',end='',flush=True)
           
        ### compiling tf graph for tac and another tf graph for its trainable cpts

        # step 1: create tac inputs (variables) by executing corresponding ops of OpsGraph
        self.__create_variables(ops_graph) # must be done outside of @tf.functions

        # step 2: create remaining tensors of tac tf graph
        # self.MAR() returns the tac marginals
        espec    = [tf.TensorSpec(shape=e.shape,dtype=e.dtype) for e in self.evidence_variables]
        self.MAR = self.__marginals.get_concrete_function(ops_graph,*espec) 
       
        # trainable cpts are created twice: initially as part of the tac tf graph (above), 
        # then in their own tf graph (so we can save them without evaluating tac graph)
        # self.TCPTS() returns trainable cpts (for saving)
        self.TCPTS = self.__trainable_cpts.get_concrete_function(ops_graph) 
        
        ### compiling tf graphs for computing metrics and losses (three tf graphs)
        
        u.show('metrics...',end='',flush=True)
        shape, dtype = self.marginal_spec
        mspec    = tf.TensorSpec(shape=shape,dtype=dtype)
        self.CA  = self.__classification_accuracy.get_concrete_function(mspec,mspec)
        self.CE  = self.__cross_entropy.get_concrete_function(mspec,mspec)
        self.MSE = self.__mean_squared_error.get_concrete_function(mspec,mspec)
        
        ### compiling tf graph for optimizer (to minimize loss)
        
        if self.trainable:
            u.show('optimizer...',end='',flush=True)
            self.optimizer_lr = tf.Variable(.1,dtype=p.float) # will be updated
            self.optimizer    = tf.optimizers.Adam(learning_rate=self.optimizer_lr)
            rspec             = tf.TensorSpec(shape=(),dtype=p.float)
            lspec             = tf.TensorSpec(shape=(),dtype=tf.string)
            self.OPT = self.__optimize_loss.get_concrete_function(rspec,lspec,mspec,*espec)
            self.optimizer_state = self.optimizer.get_weights() # after self.OPT is set
       
        # some book keeping
        self.loss_fns   = {'CE': self.CE, 'MSE': self.MSE}
        self.metric_fns = {'CA': self.CA, 'CE': self.CE, 'MSE': self.MSE}
        
        self.parameter_count = sum([t.shape[0] for t in self.weight_variables]) 
        
        # compute sizes of compiled tf graphs
        graph_size      = lambda fn: self.__graph_size(fn.graph)[0]
        self.size, self.binary_rank, self.rank = self.__graph_size(self.MAR.graph)
        concrete_fns    = (self.TCPTS, self.CE, self.CA, self.MSE)
        metrics_size    = sum(graph_size(fn) for fn in concrete_fns)
        self.total_size = self.size + metrics_size
        
        # printing statistics of compiled tf graphs
        if u.verbose: # do computations below only if we will print results
            stats = self.__graph_stats(self.MAR.graph)
            u.show(stats)
        u.show(f'      binary rank {self.binary_rank:.1f}, rank {self.rank} (for separators)')
        u.show(f'    metrics size {metrics_size:,}')
        if self.trainable:
            opt_size = graph_size(self.OPT)
            u.show(f'    optimizer size {opt_size:,}')
            self.total_size += opt_size 
        
        assert not self.trainable or self.weight_variables
        assert not self.weight_variables or self.trainable
        assert self.trainable or self.fixed_cpt_tensors
        
        self.finalized = True
        
    """ functions for building tf graph for tac, by executing an OpsGraph """
    
    # this function creates all tac variables: evidence variables and weight variables
    # the function is called outside @tf.function (a subtlety of tf.function mode in
    # comparison to eager mode)
    def __create_variables(self,ops_graph):
        self.evidence_variables,\
        self.cpt_variables,\
        self.fixed_zeros_count = ops.Op.create_variables(ops_graph)
        # cpt_variables are the weight variables grouped by cpt
        self.weight_variables = []
        for cpt_variable in self.cpt_variables: 
            self.weight_variables.extend(cpt_variable)
        self.weight_variables = tuple(self.weight_variables)
        
    @tf.function
    # this function creates all tac tensors beyond variables (including trainable cpts)
    # the function is compiled into the callable tf graph self.MAR
    def __marginals(self,ops_graph,*evidence):
        # assert evidence
        for t,e in zip(self.evidence_variables,evidence): 
            t.assign(e)
        # compute marginal
        marginal_tensor, self.fixed_cpt_tensors = ops.Op.execute(ops_graph,*evidence)
        self.marginal_spec = (marginal_tensor.shape,marginal_tensor.dtype)
        return marginal_tensor     
           
    @tf.function
    # this function creates tensors for trainable cpts (composed from trainable weights)
    # we use this to access the values of trainable cpts, for saving to file, etc
    # the function is compiled into the callable tf graph self.TCPTS
    def __trainable_cpts(self,ops_graph):
        return ops.Op.trainable_cpts(ops_graph)
                          
    """
    Interface for evaluating tac:
        # evaluate()
        # compute_metric()
    """

        
    # evaluates tac for given evidence batch
    # evidence is list of np arrays
    def evaluate(self,evidence):
        assert not self.trainable or self.learned_weights
        assert not self.training_ongoing
        evidence = tf_evd(evidence)    # convert to tf constants
        
        self.__trace_tac()             # will execute once, if profile = true
        marginal = self.MAR(*evidence) # evaluating callable tf graph
        self.__save_tac()              # will execute once, if profile = true
        
        return marginal.numpy()
        
    # computes metric that compares labels with predictions on evidence
    # evidence is list of np arrays, labels is np array
    def compute_metric(self,metric_type,evidence,labels):
        assert not self.trainable or self.training_ongoing or self.learned_weights
        assert metric_type in self.metric_types
        evidence    = tf_evd(evidence)    # convert to tf constants
        labels      = tf_lab(labels)      # convert to tf constant
         
        self.__trace_tac()                # will execute once, if profile = true
        predictions = self.MAR(*evidence) # evaluating a callable tf graph
        self.__save_tac()                 # will execute once, if profile = true
    
        metric_fn   = self.metric_fns[metric_type]
        metric      = metric_fn(labels,predictions) # evaluating a callable tf graph
        return metric.numpy()
    
    
    """
    Interface for training tac:
        # init_training()
        # reset_optimizer()
        # assign_random_weights()
        # save_current_weights()
        # restore_saved_weights()
        # optimize_loss()
        # end_training()
   
    We may fit a tac multiple times (using different datasets), so we need to reset
    the optimizer before each fit. This is done by saving its initial state and
    recovering it later before a new fit. We also reset the optimizer when searching
    for initial tac weights as we need each 'lookahead' step to be independent.
    
    Optimizer state is captured using its 'weights:' there are 2x+1 weights,
    where x is the number of trainable tac weights (2 for each tac weight plus
    one weight for the number of iterations). The optimizer weights, excluding 
    the one for iterations, are indexed in two other ways: as the optimizer
    variables and its slots. This can be seen by examining the tensorflow source 
    code for the various optimizer methods and properties.
    https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
    """
    
    def init_training(self,burn_in,batch_count,lr_function_args):
        assert self.finalized and self.trainable and not self.training_ongoing

        learning_rate_fn = tf.optimizers.schedules.PolynomialDecay(*lr_function_args)
 
        # will be called after each epoch by optimize_loss
        def update_lr(burn_in=burn_in,batch_count=batch_count,fn=learning_rate_fn):
            epochs = self.optimizer.iterations.numpy() // batch_count
            epochs = 0 if epochs < burn_in else epochs-burn_in+1
            return fn(epochs) # tensor: decayed lr as a function of number of epochs
        
        self.update_lr = update_lr 
        self.reset_optimizer()

    
        # initialize weights to uniform
        # becareful about initializing thresholds
        for t in self.weight_variables:
            value = [1.]*t.shape[0]
            t.assign(value)

        self.training_ongoing = True
            
    # called after weights have been learned and saved
    # saves learned cpts to file
    def end_training(self,fname):
        assert self.training_ongoing and self.learned_weights
        self.restore_saved_weights() # best learned weights
        self.training_ongoing = False
        labels = self.train_cpt_labels
        values = [t.numpy() for t in self.TCPTS()]
        for label,cpt in zip(labels,values):
            if label.startswith('cpt'):
                assert np.allclose(np.sum(cpt,axis=-1),1.) # normalized cpts
        formatter = {'float_kind': lambda n: f'{n:.4f}'}
        with open(fname,'w') as f:
            for label, cpt in zip(labels,values):
                f.write(label)
                f.write('\n')
                f.write(np.array2string(cpt,formatter=formatter))
                f.write('\n\n')

    # each call to this function will assign new random values to weight variables
    def assign_random_weights(self):
        # fixed glabal_seed ==> the same weights each time this function is called
        global_seed = randint(0,1000000)
        for cpt_variable in self.cpt_variables:
            cpt_weights = cpt_variable
            # 0-mean and 0-std leads to a uniform distribution
            # as std gets smaller, the distribution tends to uniform
            # as std gets larger,  the distribution can become more extreme (not good) 
            mean, stddev = 0.0, 1.0
            for i, w in enumerate(cpt_weights):
                # same normal if same shape and seed, so we get the same weights
                # for all cpts that have the same shape, but different weights
                # for distributions in the same cpt even if they have the same shape
                seed  = [i,global_seed]
                value = tf.random.stateless_normal(w.shape,mean=mean,stddev=stddev,
                            seed=seed,dtype=p.float)
                w.assign(value)
        """
        print('\nweights')
        weights = list(self.weight_variables)
        weights.sort(key=lambda w: w.name)
        for w in weights[:5]: 
            print('\n',w.name,u.normalize_weight(w.numpy()))
            #print(w.name,w.numpy())
        """
        
    # saves current values of weight tensors
    def save_current_weights(self):
        assert self.training_ongoing
        self.learned_weights = tuple(w.numpy() for w in self.weight_variables)
   
    # restores the saved weights so they are used when evaluating the tac
    def restore_saved_weights(self):
        assert self.learned_weights
        for t,v in zip(self.weight_variables, self.learned_weights):
            t.assign(v)
        
    # brings optimizer to its initial state (erases its memory of previous optimizations)
    def reset_optimizer(self):
        self.optimizer.set_weights(self.optimizer_state)
        assert self.optimizer.iterations.numpy() == 0
        
    # one-step optimization of loss for evidence/labels batch
    # loss_type: string
    # evidence: list of np arrays
    # labels: np array
    def optimize_loss(self,loss_type,evidence,labels):
        assert self.training_ongoing
        assert loss_type in self.loss_types
        # callable tf graphs require tensor inputs
        loss_type = tf.constant(loss_type,dtype=tf.string)
        evidence  = tf_evd(evidence)  # convert to tf constants
        labels    = tf_lab(labels)    # convert to tf constant
        lr        = self.update_lr()  # before optimizing loss
        loss      = self.OPT(lr,loss_type,labels,*evidence)
        return loss.numpy(), lr.numpy()

    @tf.function
    # evidence and labels and loss_type are tensors
    # this function is compiled into tf callable graph self.OPT
    def __optimize_loss(self,learning_rate,loss_type,labels,*evidence):
        self.optimizer_lr.assign(learning_rate)
        weights = self.weight_variables
        with tf.GradientTape() as tape:  
            for w in weights: tape.watch(w)
            predictions = self.MAR(*evidence) # posterior marginal
            if loss_type == tf.constant('CE',dtype=tf.string):
                loss = self.CE(labels,predictions)
            else:
                loss = self.MSE(labels,predictions)
        grads = tape.gradient(loss,weights)
        self.optimizer.apply_gradients(zip(grads,weights))
        return loss
        
    """
    loss and metric functions:
        # loss & metric: cross entropy, needs to be differentiable
        # loss & metric: mean squared error, need to be differentiable 
        # metric only: classification accuracy, need not be differentiable
        
    labels and predictions represent distributions (first axis is batch)
    """

    @tf.function
    # this function is compiled into the callable tf graph self.MSE
    # labels and predictions are tensors
    def __mean_squared_error(self,labels,predictions):
        batch_mse = tf.losses.mean_squared_error(labels,predictions)
        return tf.reduce_mean(batch_mse,name='MSE') # average across batch
        
    # deals with zero probabilities differently than the standard clipping method
    # handles predictions of the form (0,...,0), which arise for zero-probability evidence
    # (returns highest cross entropy in this case)
    @tf.function
    # this function is compiled into the callable tf graph self.CE
    # labels and predictions are tensors
    def __cross_entropy(self,labels,predictions):
        ones             = tf.ones_like(predictions)*p.eps
        safe_predictions = tf.where(tf.equal(predictions,0.),ones,predictions)
        batch_ce         = - tf.reduce_sum(labels*tf.math.log(safe_predictions),axis=-1)
        return tf.reduce_mean(batch_ce,name='CE') # average across batch
            
    # labels are deterministic distributions (one-hot vectors)
    # returns the percentage of predictions that classify as corresponding labels
    @tf.function
    # this function is compiled into the callable tf graph self.CA
    # one_hot_labels and predictions are tensors
    def __classification_accuracy(self,one_hot_labels,predictions):
        max_p   = tf.reduce_max(predictions,axis=-1,keepdims=True)
        equal1  = tf.equal(predictions,max_p)
        equal2  = tf.equal(one_hot_labels,1.)
        equal   = tf.math.equal(equal1,equal2)
        match   = tf.reduce_all(equal,axis=-1)
        total   = tf.size(match)
        correct = tf.math.count_nonzero(match,dtype=total.dtype)
        return tf.math.divide(correct,total,name='CA')
   
   
    """
    graph size and profiling
    """

    def __graph_size(self,graph):
        ops      = graph.get_operations()
        size     = 0     # size of all tensors in tf graph
        max_rank = 0     # largest rank attained by any tensor
        max_size = 0     # largest size attained by any tensor
        unique   = set() # tensors encountered
        for op in ops:
            tensors = op.values()
            if len(tensors) == 0: continue
            for tensor in tensors: # optimizer ops may have multiple outputs
                # batch is always first axis if present (batch has None dimension)
                assert all(i==0 or d != None for i,d in enumerate(tensor.shape))
                # shape could be [] or [1]
                shape = tuple(d for d in tensor.shape if d != None)
                if op.type != 'Reshape': # reshape does not use space (per tensorboard)
                    tsize    = reduce(lambda x,y: x*y, shape, 1)
                    size    +=  tsize
                    max_size = max(tsize,max_size)
                rank     = len(shape) # excludes batch dimension
                max_rank = max(max_rank,rank)
                assert tensor.name not in unique
                unique.add(tensor.name)
        max_binary_rank = log2(max_size) if max_size > 0 else 0
        # rank and binary rank only used/reported for TAC graph
        return size, max_binary_rank, max_rank


    # computes and print size of tensorflow graph:
    #  -some of the tensors are of type Const, created by tf to implement operations
    #  such as tf.reshape (a constant that holds the new shape).
    #  -variables also creates other tensors such as identity and assign.
    def __graph_stats(self,graph):
        ops             = graph.get_operations()
        ops_count       = 0     # number of operations in tf graph (that have output)
        tensors_size    = 0     # size of all tensors in tf graph
        detailed_counts = {}    # {op:count}
        addmul          = (0,0) # addition and multiplication
        reshape         = (0,0) # special, always report
        transpose       = (0,0) # special, always report
        for op in ops:
            tensors = op.values()
            if len(tensors) == 0: continue
            assert len(tensors) == 1 # only one tensor output for operation
            tensor        = tensors[0]
            shape         = tuple(d for d in tensor.shape if d != None)
            # shape could be [] or [1]
            if op.type == 'Reshape': # reshape does not use space (per tensorboard)
                tensor_size  = 0
            else:
                tensor_size = reduce(lambda x,y: x*y, shape, 1)
            tensors_size += tensor_size
            ops_count    += 1
            if op.type in detailed_counts:
                (c, s) = detailed_counts[op.type]
                detailed_counts[op.type] = (c+1,s+tensor_size)
            else:
                detailed_counts[op.type] = (1,tensor_size)
            if op.type in ('Sum','Mul','MatMul','BatchMatMulV2','DivNoNan'):
                addmul = (addmul[0]+1,addmul[1]+tensor_size)
            if op.type == 'Reshape':
                reshape = (reshape[0]+1,reshape[1]+tensor_size)
            if op.type == 'Transpose':
                transpose = (transpose[0]+1,transpose[1]+tensor_size) 
        nc, ns = addmul  
        rc, rs = reshape
        tc, ts = transpose   
        return (f'\n    tac size {tensors_size:,}, ops count {ops_count:,}\n'
                f'      arithmetic (s {ns:,}, c {nc:,}), '
                f'reshape (s {rs:,}, c {rc:,}), '
                f'transpose (s {ts:,}, c {tc:,})'
                #f'\n             {detailed_counts}'
                )

    """
    saving tac to file and profiling performance
    """
    
    # https://www.tensorflow.org/tensorboard/graphs
    def __trace_tac(self):
        if not self.profile or self.saved_tac: return
        tf.summary.trace_on(graph=True,profiler=True)
        
    # save trace to file
    # type in shell:  tensorboard --logdir .
    # then open in browser: http://localhost:6006/
    # to view the tf graph in tensorboard     
    def __save_tac(self):
        if not self.profile or self.saved_tac: return
        with self.writer.as_default():
            # trace_export will stop tracing
            tf.summary.trace_export(
                name='TAC',
                step=0,
                profiler_outdir=self.logdir)
        self.saved_tac = True
                
    def log(self,step,loss,metric,lr):
        if not self.profile: return
        with self.writer.as_default():
            tf.summary.scalar('loss',loss,step)
            tf.summary.scalar('metric',metric,step)
            tf.summary.scalar('learning rate',lr,step)
            self.writer.flush()