from numpy import ndarray
from itertools import count
from collections import defaultdict

import tensors.ops as ops
import utils.utils as u

"""
An OpsGraph is constructed by a symbolic version of the jointree algorithm (inference.py).

An OpsGraph is an abstraction of a tensorflow graph (tf.Graph).
The tf.Graph is embedded in a TacGraph object (tacgraph.py).
Executing the operations of an OpsGraph adds operations to tf.Graph and
sets various attributes of the embedding TacGraph.

Variables (Var) are abstractions of tbn nodes that contain id, cardinality and name.

The batch variable is special: it is added to the variables of an operation if the
operation output depends on evidence. 

The batch variable has cardinality -1. The actual batch size determined when
evaluating or training a tac.
"""

# abstraction of tbn nodes
class Var:
  
    def __init__(self,id,card,name):
        assert type(id) is int and id >= -1
        assert type(card) is int and (card == -1 or card > 0)
        assert type(name) is str
        self.id       = id
        self.card     = card
        self.name     = name
        self.is_batch = (id == -1)
        
    def __lt__(self,other):
        return self.id < other.id
        
    def __str__(self):
        return f'{self.id}'
        

class OpsGraph:

    op_types   = ('m','p','mp','n','s','c')
    USE_FIXED_THRESHOLDS = True
    
    def __init__(self,trainable,testing):
        self.trainable        = trainable # whether includes trainable CPTs
        self.testing          = testing   # whether includes testing CPTs
        self.ops              = []   # list of all operations 
        self.evidence_vars    = None # tuple of evidence vars ordered as tac constructor
        self.train_cpt_labels = []   # for saving learned cpts with appropriate labels
        self.scale_after_M    = 5    # scale after M multiplications to avoid underflows
                                     # scaling is critical for learning
                                   
        # for validation
        self.evd_ops          = set()
        self.cpt_ops          = defaultdict(set)
        self.selected_cpt_ops = set()
        
        # ops cache: catch what is missed by message caching in inference.py
        # precautionary: message caching should catch all and normally the hit
        # rate for ops_cache should be 0.0%
        self.ops_cache = {op:{} for op in OpsGraph.op_types}
        self.lookups   = 0
        self.hits      = 0
        
        # dictionaries for implementing tied cpts: maps tie_id to op 
        self.tied_cpt_op = defaultdict(dict)
        
        # dictionary for caching vars (so we have a unique var for each tbn node)
        self.vars = {}

        # batch var: ops that depend on evidence will have a batch var
        self.batch_var = self.__get_var(-1,-1,'batch')
        
        
    """ operations cache """
    
    # complements message caching
    def __lookup(self,op_type,entry):
        assert op_type in OpsGraph.op_types
        self.lookups  += 1
        if entry in self.ops_cache[op_type]:
            self.hits += 1
            return self.ops_cache[op_type][entry]
        return None
        
    def __save(self,op_type,entry,op):
        assert op_type in OpsGraph.op_types
        self.ops_cache[op_type][entry] = op
        
        
    """ canonical representation of tensor vars """
    def __get_var(self,id,card,name):
        cache = self.vars
        if id in cache:
            var = cache[id]
            assert var.card == card and var.name == name
            return var
        var = Var(id,card,name)
        cache[id] = var
        return var
        
    # returns a sorted tuple of vars corresponding to tbn nodes
    # add a batch dimension (var) if needed
    def nodes2vars(self,nodes,add_batch):
        vars = [self.__get_var(n.id,n.card,n.name) for n in nodes]
        vars.sort()
        if add_batch:
            var = self.batch_var
            vars.insert(0,var) # [var,*vars]
        return tuple(vars)
        
    def shape(self,vars):
        return tuple(d.card for d in vars)

    """ adds an op that creates a tensor for a scalar """
    def add_scalar_op(self,scalar):
        op = self.__lookup('c',scalar)
        if not op:
            vars = tuple()
            op   = ops.ScalarOp(scalar,vars)
            self.ops.append(op)
            self.__save('c',scalar,op)
        return op
        
    """ adds an op that creates a tensor for multipling inputs """
    def add_multiply_op(self,input1,input2,nodes):
        op = self.__lookup('m',(input1,input2))
        if not op:
            add_batch = input1.has_batch or input2.has_batch
            vars      = self.nodes2vars(nodes,add_batch)
            assert set(vars) == set(input1.vars) | set(input2.vars)
            op = ops.MultiplyOp(input1,input2,vars)
            self.ops.append(op) 
            if add_batch and op.mul_depth == self.scale_after_M: 
                op = self.add_scale_op(op,nodes) # need to scale to avoid underflows
            self.__save('m',(input1,input2),op)
        return op
        
    """ adds an op that creates a tensor for projecting input """         
    def add_project_op(self,input,nodes):
        add_batch = input.has_batch
        vars      = self.nodes2vars(nodes,add_batch)  
        op = self.__lookup('p',(input,vars))
        if not op:
            assert set(vars) <= set(input.vars)
            op = ops.ProjectOp(input,vars)
            self.ops.append(op)
            self.__save('p',(input,vars),op)
        return op
        
    """ adds an op that creates a tensor for multipling then projecting inputs """
    def add_mulpro_op(self,input1,input2,nodes):
        add_batch = input1.has_batch or input2.has_batch
        vars      = self.nodes2vars(nodes,add_batch)
        # use multiply if no variables are summed out
        assert input1.vars != vars and input2.vars != vars 
        op = self.__lookup('mp',(input1,input2,vars))
        if not op:
            op = ops.MulProOp(input1,input2,vars)
            self.ops.append(op) 
            self.__save('mp',(input1,input2,vars),op)
        return op
        
    """ adds an op that creates a tensor for normalizing input """    
    def add_normalize_op(self,input,nodes): 
        op = self.__lookup('n',input)
        if not op:
            add_batch = input.has_batch
            vars      = self.nodes2vars(nodes,add_batch)
            assert vars == input.vars
            op = ops.NormalizeOp(input,vars)
            self.ops.append(op)
            self.__save('n',input,op)
        return op
        
    """ adds an op that creates a tensor for scaling input: citical for learning """
    def add_scale_op(self,input,nodes):  
        op = self.__lookup('s',input)
        if not op:
            add_batch = input.has_batch
            vars      = self.nodes2vars(nodes,add_batch) 
            if not vars == input.vars:
                u.ppn(vars)
                u.ppn(input.vars)
            assert vars == input.vars 
            op = ops.ScaleOp(input,vars)
            self.ops.append(op)
            self.__save('s',input,op)
        return op
    
    """ adds an op that creates a tensor for selecting cpt """
    # assumes (1) cpt matches orders of tbn nodes in family
    #         (2) family is sorted
    #         (3) var has largest id in family (last dimension in cpt)
    def add_selected_cpt_op(self,node,cpt1_op,cpt2_op,posterior,threshold_op=None,sel_type='linear'):
        assert isinstance(cpt1_op,ops.CptOp) and isinstance(cpt2_op,ops.CptOp)
        if threshold_op is not None:
            assert isinstance(threshold_op, ops.CptOp)
        assert node not in self.selected_cpt_ops
        self.selected_cpt_ops.add(node)
        nodes     = set(node.family)
        add_batch = posterior.has_batch # if false, dead testing node
        vars      = self.nodes2vars(nodes,add_batch)
        var       = vars[-1]            # dimension of var
        assert node.id == var.id        # var has last dimension in cpt
        op = ops.SelectCptOp(var,cpt1_op,cpt2_op,posterior,vars,threshold=threshold_op, sel_type=sel_type)
        self.ops.append(op)
        return op

    """ adds an op that creates a tensor for selecting cpt using multiple thresholds """
    # assumes (1) cpt matches orders of tbn nodes in family
    #         (2) family is sorted
    #         (3) var has largest id in family (last dimension in cpt)
    def add_selected_cpt_op_v2(self,node,cpt_ops,posterior,threshold_ops,sel_type):
        for op in cpt_ops+threshold_ops:
            assert isinstance(op, ops.CptOp)
        assert node not in self.selected_cpt_ops
        self.selected_cpt_ops.add(node)
        nodes     = set(node.family)
        add_batch = posterior.has_batch # if false, dead testing node
        vars      = self.nodes2vars(nodes,add_batch)
        var       = vars[-1]            # dimension of var
        assert node.id == var.id        # var has last dimension in cpt
        if sel_type in ops.SelectCptOpV2.sel_types:
            op = ops.SelectCptOpV2(var,vars,cpt_ops,threshold_ops,posterior,sel_type)
        else: 
            op = ops.SelectCptOpV3(var,vars,cpt_ops,threshold_ops,posterior,sel_type)

        self.ops.append(op)
        return op
        
    """ adds an op that creates a tensor for cpt """
    # assumes (1) cpt matches orders of tbn nodes in family 
    #         (2) family is sorted
    #         (3) var has largest id in family (last dimension in cpt)
    #             except that threshold type cpt does not have var
    def add_cpt_op(self,node,cpt,cpt_type):
        #print("Add cpt type: ", cpt_type)
        assert isinstance(cpt,ndarray) 
        assert cpt_type.startswith('cpt') or cpt_type.startswith('thres')
        assert node not in self.cpt_ops[cpt_type]
        self.cpt_ops[cpt_type].add(node)
        nodes  = set(node.family)
        tie_id = node.cpt_tie
        vars   = self.nodes2vars(nodes,add_batch=False)
        var    = vars[-1]                    # dimension of var
        assert node.id == var.id             # var has last dimension in cpt
        if cpt_type.startswith('thres'):
            vars = vars[:-1]
        assert self.shape(vars) == cpt.shape # cpt matches ordered family
        
        # returns an op for a fixed or trainable cpt
        def cpt_op():
            if not self.trainable or node.fixed_cpt:
                op = ops.FixedCptOp(var,cpt,cpt_type,vars)
            elif cpt_type.startswith('thres') and self.USE_FIXED_THRESHOLDS:
                # tac for training but use fixed thresholds
                op = ops.FixedCptOp(var,cpt,cpt_type,vars)
                self.train_cpt_labels.append('fixed_' + node.cpt_label[cpt_type])
        
            else:
                op = ops.TrainCptOp(var,cpt,cpt_type,node.fixed_zeros,vars)
                self.train_cpt_labels.append(node.cpt_label[cpt_type])
            return op
            
        # main code
        if tie_id is None: # cpt is not tied
            op = cpt_op()
        else: # cpt is tied
            if tie_id in self.tied_cpt_op[cpt_type]:
                # we already created an op for this cpt
                op = self.tied_cpt_op[cpt_type][tie_id]
                # op creates a tensor for cpt, just reference its tensor
                op = ops.RefCptOp(var,cpt_type,op,vars) # does not create a tensor
                assert self.shape(vars) == self.shape(op.vars) # tied cpts share shapes
            else: # we need an op that creates tensor for cpt
                op = cpt_op()
                self.tied_cpt_op[cpt_type][tie_id] = op # save so we can reference later
        
        self.ops.append(op)
        return op


    """ adds op that create thresholds tensor with increasing constraint for each testing node"""
    def add_trainable_thresholds_op(self,node,thresholds):
        assert node.is_node_v2()
        assert node not in self.cpt_ops["thresholds"]
        self.cpt_ops["thresholds"].add(node)
        nodes  = set(node.family)
        vars   = self.nodes2vars(nodes,add_batch=False)
        var,vars    = vars[-1],vars[:-1]     # dimension of var
        assert node.id == var.id             # var has last dimension in cpt
        assert node.num_intervals == len(thresholds)+1 or node.num_intervals == len(thresholds)
        assert self.shape(vars) == thresholds[0].shape # cpt matches ordered family

        # create op for training multiple thresholds
        op = ops.TrainThresholdsOp(var,thresholds,vars)
        self.ops.append(op)
        return op


        




    """ adds op that reference one individual threshold from N-1 thresholds"""
    def add_ref_threshold_op(self,node,thresholds_op,cpt_type):
        #print("Add threshold type: ", cpt_type)
        assert node.is_node_v2()
        assert isinstance(thresholds_op, ops.TrainThresholdsOp) 
        assert cpt_type.startswith('thres')
        assert node not in self.cpt_ops[cpt_type]
        self.cpt_ops[cpt_type].add(node)
        nodes  = set(node.family)
        vars   = self.nodes2vars(nodes,add_batch=False)
        var,vars    = vars[-1],vars[:-1]     # dimension of var
        assert node.id == var.id             # var has last dimension in cpt

        # add ref threshold op
        op = ops.RefThresholdOp(var,cpt_type,vars,thresholds_op)
        self.train_cpt_labels.append(node.cpt_label[cpt_type]) # add to train cpt labels as parameters
        self.ops.append(op)
        return op
        


    
    
    
    """ adds ops that create tensors for evidence when the ops are executed """       
        
    # order of evidence_nodes MUST MATCH order of inputs passed to TAC constructor, 
    # which also matches the order of evidence_tensors in TacGraph 
    def add_evidence_ops(self,evidence_nodes):
        ops = [self.__add_evidence_op(node) for node in evidence_nodes]
        self.evidence_vars = tuple(op.var for op in ops)
        return ops

    # it is possible that an evidence node will be pruned during inference
    # such evidence nodes will not be part of the jointree view during inference,
    # but they will exist in the ops graph and also in the tensor graph
    # (they will be disconnected from the rest of the ops graph/tensor graph)
    def __add_evidence_op(self,node):
        assert node not in self.evd_ops
        self.evd_ops.add(node)
        vars = self.nodes2vars(set([node]),add_batch=True)
        var  = vars[1] # vars[0] is batch
        op   = ops.EvidenceOp(var,vars)
        self.ops.append(op)
        return op
   
    """ OpsGraph stats """
    def print_stats(self): 
        mc = pc = mpc = nc = scc = sec = rc = ec = fc = tc = 0
        for op in self.ops:
            op_type = type(op)
            if    op_type == ops.MultiplyOp:  mc  += 1
            elif  op_type == ops.ProjectOp:   pc  += 1
            elif  op_type == ops.MulProOp:    mpc += 1
            elif  op_type == ops.NormalizeOp: nc  += 1
            elif  op_type == ops.ScaleOp:     scc += 1
            elif  op_type == ops.SelectCptOp or op_type == ops.SelectCptOpV2: sec += 1
            elif  op_type == ops.RefCptOp:    rc  += 1
            elif  op_type == ops.EvidenceOp:  ec  += 1
            elif  op_type == ops.FixedCptOp:  fc  += 1
            elif  op_type == ops.TrainCptOp:  tc  += 1
            #else: assert op_type in (ops.BatchSizeOp, ops.ScalarOp)
                
        rate = self.hits*100/self.lookups if self.lookups > 0 else 0
        stats = (f'  OpsGraph ops count {len(self.ops):,}:\n'
                 f'    mulpro {mpc:}, mul {mc:}, pro {pc:}, norm {nc}, scale {scc}\n'
                 f'    cpt trained {tc}, fixed {fc}, reference {rc}, selection {sec}\n'
                 f'    evidence {ec}'
                 #f'\n    cache lookups {self.lookups}, hits {self.hits}, rate {rate:.1f}%'
                 #f'     '
                 )
        print(stats)
