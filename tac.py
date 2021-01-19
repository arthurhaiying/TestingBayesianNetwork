import time
import numpy as np
from tbn.node import Node

import compile.opsgraph as og
import tensors.tacgraph as tg
import compile.jointree as jointree
import compile.inference as inference
import compile.decouple as decouple
import utils.precision as p
import train.data as data
import train.train as train
import utils.utils as u

from copy import copy,deepcopy

"""

The TAC object is a wrapper of a TacGraph (tacgraph.py), which is constructed based 
on an abstraction: the OpsGraph (opsgraph.py) object. 

The OpsGraph is constructed using the jointree algorithm, and it contains enough 
details to construct a TacGraph that represents a TAC. The inputs to the TacGraph 
are tensors of type Placeholder, whose values are evidence vectors (evidence_tensors). 
Multiple evidence vectors can be passed to a placeholder (a batch). Hence, a TAC can 
be evaluated over a dataset at once. 

The other type of inputs to the TAC depend on whether it is trainable. If the TAC is 
not trainable, the other inputs are tensors of type Variable (non-trainable), whose 
values are CPTs (cpt_tensors). If the TAC is trainable, the other inputs are tensors 
of type Variable (trainable), whose values are unnormalized distributions (weight_tensors). 

Each distribution in a CPT has a corresponding weight_tensor. Each weight_tensor is 
normalized usinga softmax tensor operation and the resulting normalized tensors are 
'stacked' together to form cpt_tensors that represent CPTs (of type Variable, 
non-trainable).

The output of the tac is marginal_tensor, whose value correspond to multiple 
distributions  (the number of distributions equals the batch size).

"""
class TAC:
    
    # only first three parameters can be positional, everything else is keyword
    def __init__(self,tbn,inputs,output,*,sel_type='linear',hard_inputs=[],trainable=False,
                    elm_method='minfill',elm_wait=30,profile=False):

        u.input_check(all(tbn.is_node_name(i) for i in inputs),
            'TAC inputs must be names of tbn nodes)')
        u.input_check(tbn.is_node_name(output),
            'TAC output must be a name of a tbn node)')
        u.input_check(set(hard_inputs) <= set(inputs),
            'TAC hard inputs must be a subset of its inputs')
        u.input_check(inputs,
            'TAC inputs cannot be empty')
        u.input_check(output not in inputs,
            'TAC output cannot be one of its inputs')
        u.input_check(sel_type in ('linear', 'threshold', 'sigmoid','nearest'), 'Invalid selection type')
        
        # inputs are names of tbn nodes
        # output is name of tbn node
        self.trainable        = trainable # whether tac parameters can be trained
        self.profile          = profile   # saves tac and profiles time
        self.sel_type         = sel_type  # save selection methold type
        self.tbn              = None      # copy prepared for inference
        self.input_nodes      = None      # tac input (tbn nodes)
        self.output_node      = None      # tac output (tbn node)
        self.hard_input_nodes = None      # whether evidence will always be hard
        self.ops_graph        = None      # ops graph representing tac
        self.tac_graph        = None      # tensor graph representing tac
        self.size             = None      # size of tensor graph
        self.rank             = None      # max rank of any tensor
        self.binary_rank      = None      # max rank of tensor dimensions were binary
        self.parameter_count  = None      # number of trainable parameters
        
        self.loss_types    = ('CE','MSE')
        self.metric_types  = ('CE','CA','MSE')
        
        self.circuit_type  = 'TAC' if tbn.testing else 'AC'
        self.network_type  = 'TBN' if tbn.testing else 'BN'
        
        # compiling the tbn
        self.__compile(tbn,inputs,output,hard_inputs,trainable,elm_method,elm_wait,profile) 
           
        # construct trainer for fitting tac (after compiling tbn)
        if trainable: 
            self.trainer = train.Trainer(self)

    # compile tbn into tac_graph
    def __compile(self,net,inputs,output,hard_inputs,trainable,elm_method,elm_wait,profile): 
        if profile: u.show('\n***PROFILER ON***')
        u.show(f'\nCompiling {self.network_type} into {self.circuit_type}')
        start_compile_time = time.time()
        
        # net1 and net2 have nodes corresponding to inputs and output (same names)
        net1                  = net.copy_for_inference()
        self.tbn              = net1
        self.input_nodes      = u.map(net1.node,inputs)
        self.output_node      = net1.node(output)
        self.hard_input_nodes = u.map(net1.node,hard_inputs)   

        # decouple net1 for more efficient compilation
        # net2 is only used to build jointree (duplicate functional cpts)
        net2, elm_order, _ = decouple.get(net1,self.hard_input_nodes,trainable,
                                 elm_method,elm_wait)
        # net2 may be equal to net1 (no decoupling)
        # if net2 != net1 (decoupling happened), then net2._decoupling_of = net1

        # compile tbn into an ops_graph
        jt        = jointree.Jointree(net2,elm_order,self.hard_input_nodes,trainable)
        ops_graph = og.OpsGraph(trainable,net.testing) # empty

        # inference will populate ops_graph with operations that construct tac_graph
        inference.trace(self.output_node,self.input_nodes,net1,jt,ops_graph,self.sel_type)
        if u.verbose: ops_graph.print_stats()
        
        # construct tac_graph by executing operations of ops_graph
        self.ops_graph       = ops_graph
        self.tac_graph       = tg.TacGraph(ops_graph,profile)
        self.size            = self.tac_graph.size
        self.rank            = self.tac_graph.rank
        self.binary_rank     = self.tac_graph.binary_rank
        self.parameter_count = self.tac_graph.parameter_count
        
        compile_time = time.time() - start_compile_time
        u.show(f'Compile Time: {compile_time:.3f} sec') 
        
    """
    Evaluate tac at given evidence.
    Returns marginals.
    """
    def evaluate(self,evidence,*,batch_size=64,report_time=False):
        evd_size   = data.evd_size(evidence)   # number of examples
        batch_size = min(evd_size,batch_size)  # used batch size

        u.input_check(data.is_evidence(evidence),
            f'TAC evidence is ill formatted')
        u.input_check(data.evd_is_hard(evidence,self.input_nodes,self.hard_input_nodes),
            f'TAC evidence must be hard')
        u.input_check(data.evd_matches_input(evidence,self.input_nodes),
            f'TAC evidence must match evidence tbn nodes')
            
        u.show(f'\nEvaluating {self.circuit_type}: evidence size {evd_size}, '
               f'batch size {batch_size}')

        marginals = None
        eval_time = 0
        for i, evd_batch in enumerate(data.evd_batches(evidence,batch_size)):
            u.show(f'{int(100*i/evd_size):4d}%\r',end='',flush=True)
            start_time = time.perf_counter()
            mar_batch  = self.tac_graph.evaluate(evd_batch)
            #print("evd batch size: %s" %(evd_batch[0].shape,))
            #print("mar_batch size: %s" %(mar_batch.shape,))
            eval_time  += time.perf_counter()-start_time
            if marginals is None: marginals = mar_batch
            else: marginals = np.concatenate((marginals,mar_batch),axis=0)                 
        
        time_per_example = eval_time / evd_size 
        time_per_million = time_per_example / (self.size / 1000000)
        
        u.show(f'\rEvaluation Time: {eval_time:.3f} sec '
               f'({1000*time_per_example:.1f} ms per example,'
               f' {1000*time_per_million:.1f} ms per 1M tac nodes)')
        
        assert data.mar_matches_output(marginals,self.output_node)
        assert data.mar_is_predictions(marginals)
        
        if report_time: 
            return marginals, eval_time, batch_size
        return marginals
    
    """   
    Trains the tac using a labeled dataset.
    Stopping criteria is based on monitoring metric part of the dataset.
    """
    def fit(self,evidence,marginals,loss_type,metric_type,*,batch_size=32,fname=None):
        evd_size   = data.evd_size(evidence)    # number of examples
        batch_size = min(evd_size,batch_size)   # used batch size
        
        u.input_check(self.trainable,
            f'TAC is not trainable')
        u.input_check(data.is_evidence(evidence),
            f'evidence is ill formatted')
        u.input_check(data.evd_is_hard(evidence,self.input_nodes,self.hard_input_nodes),
            f'evidence must be hard')
        u.input_check(data.evd_matches_input(evidence,self.input_nodes),
            f'evidence must match evidence nodes of tbn')
        u.input_check(data.is_marginals(marginals),
            f'marginals ill formatted')
        u.input_check(data.mar_matches_output(marginals,self.output_node),
            f'marginals must match query node of tbn')
        u.input_check(loss_type in self.loss_types,
            f'loss {loss_type} is not supported')
        u.input_check(metric_type in self.metric_types,
            f'metric {metric_type} is not supported')
        u.input_check(data.evd_size(evidence) == len(marginals),
            f'evidence size must match marginals size')
        
        u.show(f'\nTraining {self.circuit_type}:')
        start_training_time = time.perf_counter()

        epoch_count = self.trainer.train(evidence,marginals,loss_type,metric_type,batch_size,fname)
                            
        training_time  = time.perf_counter() - start_training_time
        time_per_epoch = training_time/epoch_count
        
        u.show(f'Training Time: {training_time:.3f} sec ({time_per_epoch:.3f} sec per epoch)')
        
    """
    Computes predictions on evidence and applies metric to predictions and labels.
    Labels are distributions.
    Returns computed metric.
    """
    def metric(self,evidence,labels,metric_type,*,batch_size=64):
        evd_size   = data.evd_size(evidence)   # number of examples
        batch_size = min(evd_size,batch_size)  # used batch size
        
        u.input_check(data.is_evidence(evidence),
            f'evidence is ill formatted')
        u.input_check(data.evd_is_hard(evidence,self.input_nodes,self.hard_input_nodes),
            f'evidence must be hard')
        u.input_check(data.evd_matches_input(evidence,self.input_nodes),
            f'evidence must match evidence nodes of tbn')
        u.input_check(data.is_marginals(labels,one_hot=(metric_type=='CA')),
            f'labels ill formatted')
        u.input_check(data.mar_matches_output(labels,self.output_node),
            f'labels must match query node of tbn')
        u.input_check(metric_type in self.metric_types,
            f'metric {metric_type} is not supported')
        
        u.show(f'\nComputing {metric_type}: evidence size {evd_size}, '
               f'batch size {batch_size}')
                
        start_eval_time = time.perf_counter()
            
        batches,_ = data.data_batches(evidence,labels,batch_size)
        result    = 0
        for evd_batch, lab_batch in batches:
            bresult = self.tac_graph.compute_metric(metric_type,evd_batch,lab_batch)
            result += bresult * len(lab_batch)
        result /= evd_size # average weighted by batch size (last batch may be smaller)
        
        evaluation_time  = time.perf_counter() - start_eval_time
        time_per_example = evaluation_time/evd_size
        
        u.show(f'{metric_type} Time: {evaluation_time:.3f} sec '
               f'({time_per_example:.4f} sec per example)')
                
        return result
    
    """  
    Returns a labeled dataset with size examples.
    'grid'  : evidence is generated according to an equally-spaced grid.
    'random': evidence is generated randomly.
    """
    def simulate(self,size,evidence_type,*,hard_evidence=False):
        u.input_check(evidence_type == 'grid' or evidence_type == 'random',
            f'evidence type {evidence_type} not supported')
        
        cards = u.map('card',self.input_nodes)
        
        if evidence_type is 'grid':
            assert len(cards)==2 and all(card==2 for card in cards)
            assert not hard_evidence
            evidence = data.evd_grid(size)
        else:
            evidence = data.evd_random(size,cards,hard_evidence)
                    
        marginals = self.tac_graph.evaluate(evidence)
        
        return (evidence, marginals)  



class TACV2:
    
    # only first three parameters can be positional, everything else is keyword
    def __init__(self,tbn,inputs,outputs,*,sel_type='linear',hard_inputs=[],trainable=False,
                    elm_method='minfill',elm_wait=30,profile=False):

        u.input_check(all(tbn.is_node_name(i) for i in inputs),
            'TAC inputs must be names of tbn nodes)')
        u.input_check(all(tbn.is_node_name(o) for o in outputs),
            'TAC outputs must be names of tbn nodes)')
        u.input_check(set(hard_inputs) <= set(inputs),
            'TAC hard inputs must be a subset of its inputs')
        u.input_check(inputs,
            'TAC inputs cannot be empty')
        u.input_check(all(o not in inputs for o in outputs),
            'TAC outputs cannot be one of its inputs')
        u.input_check(sel_type in ('linear', 'threshold', 'sigmoid','nearest'), 'Invalid selection type')
        
        # inputs are names of tbn nodes
        # output is name of tbn node
        self.trainable        = trainable # whether tac parameters can be trained
        self.profile          = profile   # saves tac and profiles time
        self.sel_type         = sel_type  # save selection methold type
        self.tbn              = None      # copy prepared for inference
        self.input_nodes      = None      # tac input (tbn nodes)
        self.original_outputs = outputs
        self.original_output_nodes = None
        self.sorted_outputs   = None
        self.sorted_output_nodes = None      # tac output (tbn nodes)
        self.dummy_node       = None      # dummy child with tac outputs as parents
        self.hard_input_nodes = None      # whether evidence will always be hard
        self.ops_graph        = None      # ops graph representing tac
        self.tac_graph        = None      # tensor graph representing tac
        self.size             = None      # size of tensor graph
        self.rank             = None      # max rank of any tensor
        self.binary_rank      = None      # max rank of tensor dimensions were binary
        self.parameter_count  = None      # number of trainable parameters
        
        self.loss_types    = ('CE','MSE')
        self.metric_types  = ('CE','CA','MSE')
        
        self.circuit_type  = 'TAC' if tbn.testing else 'AC'
        self.network_type  = 'TBN' if tbn.testing else 'BN'
        
        # compiling the tbn
        self.__compile(tbn,inputs,outputs,hard_inputs,trainable,elm_method,elm_wait,profile) 
           
        # construct trainer for fitting tac (after compiling tbn)
        if trainable: 
            self.trainer = train.Trainer(self)

    # compile tbn into tac_graph
    def __compile(self,net,inputs,outputs,hard_inputs,trainable,elm_method,elm_wait,profile): 
        if profile: u.show('\n***PROFILER ON***')
        u.show(f'\nCompiling {self.network_type} into {self.circuit_type}')
        start_compile_time = time.time()

        self.original_output_nodes = list(map(net.node, outputs))
        self.sorted_outputs = list(map(lambda x: x.name, sorted(self.original_output_nodes)))
        # the marginals will be over sorted outputs
        # net1 and net2 have nodes corresponding to inputs and output (same names)
        net1 = deepcopy(net)
        dummy_node = Node(name="dummy", parents=list(map(net1.node,self.sorted_outputs)),testing=False)
        net1.add(dummy_node)
        # add dummy node as a child of output nodes and compute its parent posterior
        net1                  = net1.copy_for_inference()
        self.tbn              = net1
        self.input_nodes      = u.map(net1.node,inputs)
        self.sorted_output_nodes      = u.map(net1.node,self.sorted_outputs)
        self.dummy_node       = net1.node(dummy_node.name)
        self.hard_input_nodes = u.map(net1.node,hard_inputs)   

        # decouple net1 for more efficient compilation
        # net2 is only used to build jointree (duplicate functional cpts)
        net2, elm_order, _ = decouple.get(net1,self.hard_input_nodes,trainable,
                                 elm_method,elm_wait)
        # net2 may be equal to net1 (no decoupling)
        # if net2 != net1 (decoupling happened), then net2._decoupling_of = net1

        # compile tbn into an ops_graph
        jt        = jointree.Jointree(net2,elm_order,self.hard_input_nodes,trainable)
        ops_graph = og.OpsGraph(trainable,net.testing) # empty

        # inference will populate ops_graph with operations that construct tac_graph
        inference.trace2(self.dummy_node,self.input_nodes,net1,jt,ops_graph,self.sel_type)
        if u.verbose: ops_graph.print_stats()
        
        # construct tac_graph by executing operations of ops_graph
        self.ops_graph       = ops_graph
        self.tac_graph       = tg.TacGraph(ops_graph,profile)
        self.size            = self.tac_graph.size
        self.rank            = self.tac_graph.rank
        self.binary_rank     = self.tac_graph.binary_rank
        self.parameter_count = self.tac_graph.parameter_count
        
        compile_time = time.time() - start_compile_time
        u.show(f'Compile Time: {compile_time:.3f} sec') 
        
    """
    Evaluate tac at given evidence.
    Returns marginals.
    """
    def evaluate(self,evidence,*,batch_size=64,report_time=False):
        evd_size   = data.evd_size(evidence)   # number of examples
        batch_size = min(evd_size,batch_size)  # used batch size

        u.input_check(data.is_evidence(evidence),
            f'TAC evidence is ill formatted')
        u.input_check(data.evd_is_hard(evidence,self.input_nodes,self.hard_input_nodes),
            f'TAC evidence must be hard')
        u.input_check(data.evd_matches_input(evidence,self.input_nodes),
            f'TAC evidence must match evidence tbn nodes')
            
        u.show(f'\nEvaluating {self.circuit_type}: evidence size {evd_size}, '
               f'batch size {batch_size}')

        marginals = None
        eval_time = 0
        for i, evd_batch in enumerate(data.evd_batches(evidence,batch_size)):
            u.show(f'{int(100*i/evd_size):4d}%\r',end='',flush=True)
            start_time = time.perf_counter()
            mar_batch  = self.tac_graph.evaluate(evd_batch)
            eval_time  += time.perf_counter()-start_time
            if marginals is None: marginals = mar_batch
            else: marginals = np.concatenate((marginals,mar_batch),axis=0)                 
        
        time_per_example = eval_time / evd_size 
        time_per_million = time_per_example / (self.size / 1000000)
        
        u.show(f'\rEvaluation Time: {eval_time:.3f} sec '
               f'({1000*time_per_example:.1f} ms per example,'
               f' {1000*time_per_million:.1f} ms per 1M tac nodes)')

        if len(marginals.shape) == len(self.sorted_outputs):
            # if output is indepenedent of inputs
            shape = marginals.shape
            marginals = np.broadcast_to(marginals,shape=(evd_size,)+shape) 

        sorted_axes = (0,) + tuple(self.sorted_outputs.index(o)+1 for o in self.original_outputs)
        #print("original outputs: %s sorted outputs: %s sorted axes: %s" %(self.original_outputs,
            #self.sorted_outputs,sorted_axes))
        #print("marginals shape: %s sorted axis: %s" %(marginals.shape,sorted_axes))
        marginals = np.transpose(marginals, axes=sorted_axes)
        assert data.mar_matches_output2(marginals,self.original_output_nodes)
        assert data.mar_is_predictions2(marginals)
        
        if report_time: 
            return marginals, eval_time, batch_size
        return marginals