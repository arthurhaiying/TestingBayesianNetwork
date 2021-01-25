import numpy as np
import tensorflow as tf
from itertools import count

import utils.precision as p
import tensors.dims as d
import utils.utils as u

"""
An Op has inputs (possibly empty) and one output, which is computed based on the
inputs and other information stored with the Op.
The Op output is a tensor representing one of the following: 
evidence, cpt, selected cpt, projection, normalization, scaling, multiplication,
multiplication_projection, scalar or batch_size.
Every Op is associated with an ordered set of variables (abstraction of tbn nodes).
The Op constructs a tensor that represents a factor over these variables.
Every Op is also associated with a Dims object (dims.py), which represents a sequence
of dimensions. A dimension is an ordered set of variables. A Dims object is therefore
an aggregation of factor variables. The tensor computed by the Op is over Dims. That 
is, each axis of the tensor is a set of variables instead of a single variable. This
allows a more efficient implementation of factor operations using tensor operations.
A tensor will have a batch variable if it depends on evidence.
The batch variable has cardinality -1 (None) during compilation.
The batch cardinality is determined dynamically when the tac is evaluated or trained
and passed through the tensor of the batch-size op.
Weight tensors are special: they are the only trainable tensors, are not the output 
of operations and represent tac parameters (when normalized). Weight tensors are 
used to construct trainable CPT tensors.
CPT and weight tensors do not have a batch variable as they are shared between
all members of a batch. However, selected CPT tensors have a batch variable 
when the corresponding testing node is live.
Three types of "root" tensors are constructed when executing Ops:
    # constants for fixed CPTs and scalars
    # trainable variables for weights of trainable CPTs
    # untrainable variable for holding evidence and the batch size
"""


""" Op object (operation) takes tensors as inputs and outputs a tensor when executed """
class Op:

    cpt_cache         = None
    cpt_lookups       = None
    cpt_hits          = None
    reshape_cache     = None
    transpose_cache   = None
    restructure_cache = None
    multiply_cache    = None
    project_cache     = None
    mulpro_cache      = None

    GAMMA_VALUE = 100
    def __init__(self,inputs,vars):
        assert type(vars) is tuple

        self.inputs     = inputs # None (evidence, cpt and scalar ops) or list
        self.vars       = vars   # sorted tuple: var is an abstraction o tbn node
        self.has_batch  = any(var.is_batch for var in vars)
        self.strvars    = '-'.join([str(v.id) for v in vars if not v.is_batch])
        
        # the following are set when initializing a specific op
        self.dims       = None   # dimensions of output tensor
        self.input_dims = None   # restructured dimensions of input (used by op)
        self.static     = None   # does not depend on evidence or trainable cpts
        self.label      = None   # used to label tensors computed by op
        self.mul_depth  = 0      # 0 for all ops, except multiply, project and mulpro
        
        # the following is set when executing op
        self.tensor     = None   # output of op
        
    def __str__(self):
        return self.label
    
    """
    executing an OpsGraph happens in two, ordered steps: we first execute operations
    that create variables, then execute remaining operations
    
    @tf.functions must create variables only once, even if called multiple times:
    see https://www.tensorflow.org/tutorials/customization/performance 
    To address this cleanly, variables are created outside @tf.functions by first
    executing ops that create variables: EvidenceOp and TrainCptOp. 
    The latter has a special build_cpt execution function that composes the cpt 
    from variables (called inside @tf.function, otherwise the cpt variables 
    will be compiled away and no longer part of the compiled graph)
    """
   
    @staticmethod
    def init_caches():
        Op.cpt_cache         = {}
        Op.reshape_cache     = {}
        Op.transpose_cache   = {}
        Op.restructure_cache = {}
        Op.multiply_cache    = {}
        Op.project_cache     = {}
        Op.mulpro_cache      = {}
        
    @staticmethod
    def create_variables(ops_graph):
        
        evidence_variables = []
        weight_variables   = []
        fixed_zeros_count  = 0
        
        for op in ops_graph.ops: # parents before children
            if type(op) is EvidenceOp:
                op.execute()
                evidence_variables.append(op.tensor)
            elif type(op) is TrainCptOp:
                op.execute() # intialize weight variables for cpt
                # the weights of each cpt are grouped together
                weight_variables.append(op.weight_variables)
                fixed_zeros_count += op.fixed_zeros_count
            elif type(op) is TrainThresholdsOp:
                op.execute() # initialize weight variables for multiple thresholds 
                weight_variables.append(op.weight_variables)
#           elif op.static:
#                op.execute()
        
        # return all created variables
        return tuple(evidence_variables), tuple(weight_variables), fixed_zeros_count
        
    @staticmethod
    def execute(ops_graph,*evidence):
        
        Op.init_caches()
        fixed_cpt_tensors = []
        
        for op in ops_graph.ops:       # parents before children
 #           if op.static: continue
            if type(op) is EvidenceOp: continue # already executed
            elif type(op) is TrainCptOp: # already executed and created weight variables
                op.build_cpt_tensor()  # we now need to assemble weights into a cpt
            elif type(op) is TrainThresholdsOp: # already executed and created weight variables
                op.build_thresholds_tensor() # we now need to transform weights into multiple thresholds
            else:
                try:
                    op.execute()           # will not create variables
                except ValueError as e:
                    print("Error op type: %s"%type(op) + str(e))
                    exit(1)
                if type(op) is FixedCptOp:
                    fixed_cpt_tensors.append(op.tensor)
                
        # last op creates a tensor that holds the tac output (marginal)
        outop = ops_graph.ops[-1]       
        # structure output tensor to match expected structure of labels
        output_tensor = Op.flatten_and_order(outop.tensor,outop.dims)
        
        return output_tensor, tuple(fixed_cpt_tensors)
        
    # we build trainable cpts twice: once as part of the tac graph (above), and then
    # independently (next) to be able to get their values for saving into file, 
    # without having to evaluate the tac graph
    @staticmethod
    def trainable_cpts(ops_graph):
        train_cpt_tensors = []
        for op in ops_graph.ops: # parents before children
            if type(op) is TrainCptOp:
                op.build_cpt_tensor() # will not create variables
                train_cpt_tensors.append(op.tensor)
            elif type(op) is TrainThresholdsOp:
                op.build_thresholds_tensor() # will not create weight variables
            elif type(op) is RefThresholdOp:
                # retieve individual thresholds and append
                #train_cpt_tensors.append(op.tensor)
                op.execute()
                train_cpt_tensors.append(op.tensor)
            elif type(op) is FixedCptOp and op.type.startswith('thres') and ops_graph.trainable:
                # train TAC but using fixed threshold, also report
                op.execute()
                train_cpt_tensors.append(op.tensor)
            else:
                # other ops
                continue
        return tuple(train_cpt_tensors)
        
    @staticmethod
    # returns a tf constant for a cpt specified using an ndarray
    # ensures a unique tensor per cpt
    def get_cpt_tensor(cpt):
        shape = cpt.shape
        
        if shape in Op.cpt_cache:
            cpt_likes = Op.cpt_cache[shape]
        else:
            cpt_likes = []
            Op.cpt_cache[shape] = cpt_likes
            
        for cpt_,tensor in cpt_likes:
            if False: #np.array_equal(cpt_,cpt): 
                """ seems problematic: deactivating for now """
                # found in cache
                tf.debugging.assert_shapes([(tensor,cpt.shape)],message='check (cpt_tensor)')
                tf.debugging.assert_shapes([(tensor,cpt_.shape)],message='check (cpt_tensor)')
                return tensor
        # no tensor in cache for this cpt
        tensor = tf.constant(cpt,dtype=p.float)
        cpt_likes.append((cpt,tensor))
        return tensor
    
    
    """ 
    We work with structured tensors: a dimension/axis of a structured tensor corresponds
    to a set of tbn variables. Each structured tensor is represented by a classical
    tensor and a Dims object(see dims.y), which keeps track of the variables in each 
    tensor axis. The following operations on structred tensors are the core operations
    used to 'execute' operations in the ops graph (that is, map them into a tensor graph).
    """
       
    @staticmethod
    # tensor is over dims1
    # dims1 and dims2 have the same variable order
    # reshape tensor so it is over dims2
    def reshape(tensor,dims1,dims2):
        assert dims1.same_order(dims2) # necessary & sufficient condition for reshape
        if dims1 == dims2: return tensor
        
#        tf.debugging.assert_shapes([(tensor,dims1.shape)],message='shape check (reshape)')
        
        key = (tensor.name,dims2.shape) # dims2.shape, not dims2
        if key in Op.reshape_cache: return Op.reshape_cache[key]
            
        # TF converts dims2.shape to constant with dtype int32 by default,
        # which may overflow if one of the dimensions has more than 2^32 values
        shape  = tf.constant(dims2.shape,dtype=tf.int64)
        tensor = tf.reshape(tensor,shape)
        
        Op.reshape_cache[key] = tensor
        return tensor
                
    @staticmethod
    # tensor is over dims1
    # dims1 is congruent with dims2
    # transpose tensor so it is over dims2
    def order_as(tensor,dims1,dims2):
        assert dims1.congruent_with(dims2) # necessary & sufficient condition for order
        if dims1 == dims2: return tensor
        
#        tf.debugging.assert_shapes([(tensor,dims1.shape)],message='shape check (order_as)')
        
        perm = dims1.transpose_axes(dims2)
        key  = (tensor.name,perm) # perm, not dim2
        if key in Op.transpose_cache: return Op.transpose_cache[key]
            
        tensor = tf.transpose(tensor,perm)
        
        Op.transpose_cache[key] = tensor
        return tensor
        
    @staticmethod
    # tensor is over dims1
    # dims1 and dims2 have the same variables
    # restructure tensor so it is over dims2
    def restructure(tensor,dims1,dims2):
        assert dims1.same_vars(dims2) # necessary & sufficient condition for restructure
        if dims1 == dims2: return tensor
        
        key = (tensor.name,dims2)
        if key in Op.restructure_cache: return Op.restructure_cache[key]

        dims1_, dims2_ = dims1.restructure_into(dims2)

        # some of the following may be no-ops
        tensor = Op.reshape(tensor,dims1,dims1_)
        tensor = Op.order_as(tensor,dims1_,dims2_) 
        tensor = Op.reshape(tensor,dims2_,dims2)
        
        Op.restructure_cache[key] = tensor
        return tensor
        
    @staticmethod
    # tensor is over dims
    # flatten it and order its variables (batch, if any, will become first)
    def flatten_and_order(tensor,dims):
        fdims = dims
        if not dims.flat:
            fdims  = dims.flatten()
            tensor = Op.reshape(tensor,dims,fdims)
        if not fdims.ordered():
            odims  = fdims.order() 
            tensor = Op.order_as(tensor,fdims,odims)   
        return tensor
        
    @staticmethod
    # tensor is over dims1
    # sum out dimensions of tensor that do not appear in dims2
    # dims1 and dims2 must be aligned
    def project(tensor,dims1,dims2,keepdims=False):
        assert dims2.subset(dims1)
        if dims1 == dims2: return tensor
        axes   = dims1.project_axes(dims2)    
        tensor = tf.reduce_sum(input_tensor=tensor,axis=axes,keepdims=keepdims)
        return tensor
        
                
""" constructs tensor representing product of input1 and input2 """
class MultiplyOp(Op):
    
    id = count() # counter
      
    def __init__(self,input1,input2,vars):
        Op.__init__(self,[input1,input2],vars)
        self.static    = input1.static and input2.static
        self.mul_depth = 1 + max(input1.mul_depth, input2.mul_depth)
#        self.label     = 'M_%s__%d' % (self.strvars,next(MultiplyOp.id))
        
        self.input_dims, self.dims = \
            d.Dims.restructure_for_multiply(input1.dims,input2.dims)
        assert vars == self.dims.ordered_vars

        self.label     = 'xM_%d_%d_' % (self.dims.mem,len(vars))
        
    def execute(self):
        i1, i2         = self.inputs
        tensor1, dims1 = i1.tensor, i1.dims
        tensor2, dims2 = i2.tensor, i2.dims
        dims1_, dims2_ = self.input_dims
        
        key = (tensor1.name,tensor2.name)
        if key in Op.multiply_cache: 
            self.tensor = Op.multiply_cache[key]
            return
                
        with tf.name_scope(self.label):
            tensor1     = Op.restructure(tensor1,dims1,dims1_)
            tensor2     = Op.restructure(tensor2,dims2,dims2_)
            self.tensor = tf.multiply(tensor1,tensor2)
        
        Op.multiply_cache[key] = self.tensor
        
""" constructs tensor representing project(input1 *input2, vars) """
class MulProOp(Op):
    
    id = count() # counter
      
    def __init__(self,input1,input2,vars):
        Op.__init__(self,[input1,input2],vars)
        self.static    = input1.static and input2.static
        self.mul_depth = 1 + max(input1.mul_depth,input2.mul_depth)
#        self.label     = 'MP_%s__%d' % (self.strvars,next(MulProOp.id))
        
        self.input_dims, self.dims, self.invert, self.squeeze = \
            d.Dims.restructure_for_mulpro(input1.dims,input2.dims,vars)
        assert vars == self.dims.ordered_vars

        self.label     = 'xMP_%d_%d_' % (self.dims.mem,len(vars))

    def execute(self):
        i1, i2                     = self.inputs[0], self.inputs[1]
        tensor1, dims1             = i1.tensor, i1.dims
        tensor2, dims2             = i2.tensor, i2.dims
        (dims1_,tr1), (dims2_,tr2) = self.input_dims
        
        key = (tensor1.name,tensor2.name,self.dims)
        if key in Op.mulpro_cache:
            self.tensor = Op.mulpro_cache[key]
            return
        
        with tf.name_scope(self.label):
            tensor1 = Op.restructure(tensor1,dims1,dims1_)
            tensor2 = Op.restructure(tensor2,dims2,dims2_)
            # generalized matrix multiplication
            if self.invert: 
                self.tensor = tf.matmul(tensor2,tensor1,transpose_a=tr2,transpose_b=tr1)
            else:
                self.tensor = tf.matmul(tensor1,tensor2,transpose_a=tr1,transpose_b=tr2)   
            
        Op.mulpro_cache[key] = self.tensor
                       
""" constructs tensor representing projection of input """
class ProjectOp(Op):
    
    id = count() # counter
    
    def __init__(self,input,vars):        
        Op.__init__(self,[input],vars)
        assert vars != input.vars # otherwise, trivial project
        self.static = input.static
        self.mul_depth = input.mul_depth
#        self.label     = 'P_%s__%d' % (self.strvars,next(ProjectOp.id))
        
        self.input_dims, self.dims = d.Dims.restructure_for_project(input.dims,vars)
        assert vars == self.dims.ordered_vars
        
        self.label = 'xP_%d_%d_' % (self.dims.mem,len(vars))
     
    def execute(self):
        tensor, dims = self.inputs[0].tensor, self.inputs[0].dims
        dims_        = self.input_dims
        
        key = (tensor.name,self.dims)
        if key in Op.project_cache:
            self.tensor = Op.project_cache[key]
            return
        
        with tf.name_scope(self.label):
            tensor      = Op.reshape(tensor,dims,dims_)
            self.tensor = Op.project(tensor,dims_,self.dims)
            
        Op.project_cache[key] = self.tensor
            
""" constructs tensor representing a normalization of input """
class NormalizeOp(Op):
    
    id = count() # counter
    
    def __init__(self,input,vars):    
        Op.__init__(self,[input],vars)
        assert self.has_batch       # otherwise, no need to normalize
        self.static = input.static  # should be false
        self.label  = 'N_%s__%d' % (self.strvars,next(NormalizeOp.id))
        
        self.dims   = input.dims
        assert self.dims.batch_var
        assert vars == self.dims.ordered_vars
        
    def execute(self):
        tensor, dims = self.inputs[0].tensor, self.inputs[0].dims
        
        # NOTE: using divide_no_nan() to normalize implies that we will compute
        #       distribution (0,...,0) for evidence with zero probability (such
        #       a distribution is our way of representing 'not defined')
        with tf.name_scope(self.label):  
            axes        = tuple(i+1 for i in range(dims.rank-1)) # all but batch (first)
            pr_evd      = tf.reduce_sum(input_tensor=tensor,axis=axes,keepdims=True)    
            self.tensor = tf.math.divide_no_nan(tensor,pr_evd) # 0/0 = 0
        
""" constructs tensor representing a scaling of input: critical for learning """
class ScaleOp(Op):
    
    id = count() # counter
    
    def __init__(self,input,vars):
        Op.__init__(self,[input],vars)
        assert self.has_batch                    # otherwise, scaling is not needed
        assert not isinstance(input,NormalizeOp) # otherwise, scaling is redundant
        self.static = input.static               # should be false
        self.label  = 'S_%s__%d' % (self.strvars,next(ScaleOp.id))
    
        self.dims   = input.dims
        assert self.dims.batch_var
        assert vars == self.dims.ordered_vars
        
    def execute(self):
        tensor, dims = self.inputs[0].tensor, self.inputs[0].dims
        
        # NOTE: using divide_no_nan() implies that we return dist (0,...,0) for 
        #       evidence with zero probability
        # NOTE: scaling to > 1 is problematic as many scale operations may
        #       get multiplied together, leading to Inf
        with tf.name_scope(self.label):
            axes        = tuple(i+1 for i in range(dims.rank-1)) # all but batch (first)
            sum         = tf.reduce_sum(input_tensor=tensor,axis=axes,keepdims=True)
            ones        = tf.ones_like(sum) # neutral normalizing constants
            normalize   = tf.less(sum,1.)   # whether to normalize
            norm_const  = tf.compat.v1.where(normalize,sum,ones) # choose constants
            self.tensor = tf.math.divide_no_nan(tensor,norm_const) # 0/0 = 0  

            
""" constructs tensor representing the selected cpt of a tbn node """
class SelectCptOp(Op):
    sel_types = ('linear', 'threshold', 'sigmoid')
    def __init__(self,var,cpt1,cpt2,posterior,vars,threshold=None, sel_type="linear"):
        inputs = [cpt1,cpt2,posterior]
        if threshold is not None:
            # if threshold included in cpt selection
            inputs.append(threshold)
    
        Op.__init__(self,inputs,vars)
        self.static = posterior.static and cpt1.static and cpt2.static and (threshold is None or threshold.static) 
        self.var    = var
        self.label  = 'sel_cpt_%s_%s' % (var.name,self.strvars)
        self.dims   = d.get_dims(vars)
        self.sel_type = sel_type
        
    def execute(self):
        i1, i2, i3            = self.inputs[:3]
        cpt1, cpt2, posterior = i1.tensor, i2.tensor, i3.tensor
        if self.sel_type != 'linear':
            threshold = self.inputs[3].tensor
        
        with tf.name_scope(self.label):
            if self.sel_type == 'linear':
                #print("Use linear selection for var: {}".format(self.var))
                posterior = Op.flatten_and_order(posterior,i3.dims)
                # add a new dimension so posterior is over family not just parents
                posterior = tf.expand_dims(posterior,-1) # add trivial dimension at end (card 1)
                # selected cpt = (cpt1-cpt2)*posterior + cpt2 (linear combination)
                # selected cpt = cpt1 for posterior=1 and cpt2 for posterior=0
                x   = tf.subtract(cpt2,cpt1)
                y   = tf.multiply(posterior,x) # broadcasting 
                cpt = tf.add(y,cpt1)           # broadcasting
                self.tensor = cpt
            elif self.sel_type == 'threshold':        
                # print("Use threshold selection for var: {}".format(self.var))
                posterior = Op.flatten_and_order(posterior,i3.dims)
                indicator = tf.greater_equal(posterior, threshold)
                indicator = tf.cast(indicator, dtype=p.float) # remember to convert boolean values to numerical types
                indicator = tf.expand_dims(indicator, -1) # add a new dimenstion so indicator is over family
                # selected cpt = cpt1*(posterior > threshold) + cpt2*(1 - posterior > threshold)
                x = tf.multiply(cpt1, tf.subtract(1.0, indicator))
                y = tf.multiply(cpt2, indicator)
                cpt = tf.add(x, y)
                self.tensor = cpt
            elif self.sel_type == 'sigmoid':
                #print("Use sigmoid selection for var: {}".format(self.var))
                gamma = tf.constant(Op.GAMMA_VALUE, dtype=p.float)
                posterior = Op.flatten_and_order(posterior,i3.dims)
                difference = tf.subtract(posterior, threshold)
                indicator = tf.sigmoid(tf.multiply(gamma, difference))
                indicator = tf.expand_dims(indicator, -1)
                # indicator = 1 / {1 + exp(-gamma * (posterior - threshold))}
                x = tf.multiply(cpt1, tf.subtract(1.0, indicator))
                y = tf.multiply(cpt2, indicator)
                cpt = tf.add(x, y)
                self.tensor = cpt

def select_cpt_fn(cpts,indicators):
    # select cpt by looking up posteriors into intervals 
    if not indicators:
        return cpts[0]
    else:
        cpt0,indicator0 = cpts[0],indicators[0]
        cpt1 = select_cpt_fn(cpts[1:],indicators[1:])
        x = tf.multiply(cpt0,tf.subtract(1.0, indicator0))
        y = tf.multiply(cpt1,indicator0)
        sel_cpt = tf.add(x,y)
        return sel_cpt

""" constructs tensor representing the selected cpt of tbn node using multiple intervals"""
class SelectCptOpV2(Op):
    sel_types = ('linear', 'threshold', 'sigmoid')
    def __init__(self,var,vars,cpts,thresholds,posterior,sel_type):
        # cpts: a list of CptOps representing N CPTs
        # thresholds: a list of CptOps representing N-1 thresholds  
        n_intervals = len(cpts)
        u.input_check(len(thresholds)==n_intervals-1, "Number of thresholds and cpts does not match")
        u.input_check(sel_type in self.sel_types, "Select type %s not supported" % sel_type)
        inputs = [cpts,thresholds,posterior]

        Op.__init__(self,inputs,vars)
        self.var = var
        self.label  = 'sel_cpt_%s_%s_v2' % (var.name,self.strvars)
        self.dims   = d.get_dims(vars)
        self.static = posterior.static
        for cpt in cpts+thresholds:
            if not cpt.static:
                self.static = False
        self.n_intervals = n_intervals
        self.sel_type = sel_type

    def execute(self):
        i0,i1,i2 = self.inputs[0],self.inputs[1],self.inputs[2]
        cpts = [cpt.tensor for cpt in i0]
        thresholds = [thres.tensor for thres in i1]
        posterior = i2.tensor

        with tf.name_scope(self.label):
            posterior = Op.flatten_and_order(posterior,i2.dims)
            indicators = []
            if self.sel_type == 'threshold':
                for thres in thresholds:
                    ind = tf.cast(tf.greater_equal(posterior,thres), dtype=p.float)
                    ind = tf.expand_dims(ind,-1)
                    indicators.append(ind)

                self.tensor = select_cpt_fn(cpts,indicators)

            elif self.sel_type == 'sigmoid':
                gamma = tf.constant(Op.GAMMA_VALUE, dtype=p.float)
                for thres in thresholds:
                    diff = tf.subtract(posterior,thres)
                    ind = tf.sigmoid(tf.multiply(gamma,diff))
                    ind = tf.expand_dims(ind,-1)
                    indicators.append(ind)

                self.tensor = select_cpt_fn(cpts,indicators)
                
            elif self.sel_type == 'linear':
                '''
                # for each interval [T_i, T_i+1]
                thresholds[0] = tf.zeros(shape=thresholds[0].shape,dtype=p.float)
                for i in range(len(thresholds)):
                    if i == len(thresholds)-1:
                        width = tf.subtract(1.0,thresholds[i])
                    else:
                        width = tf.subtract(thresholds[i+1],thresholds[i])
                        # get interval length
                    diff = tf.subtract(posterior,thresholds[i])
                    ind = tf.divide(diff,width)
                    ind = tf.clip_by_value(ind,0.0,1.0)
                    ind = tf.expand_dims(ind,axis=-1)
                    indicators.append(ind)
                '''
                raise NotImplementedError("Linear selection v2 is not ready")
                self.tensor = select_cpt_fn(cpts,indicators)

            #print("cpt shape: {}".format(self.tensor.shape))
                

""" constructs tensor representing the selected cpt of tbn node using multiple intervals"""
class SelectCptOpV3(Op):
    sel_types = ('nearest')
    def __init__(self,var,vars,cpts,thresholds,posterior,sel_type):
        # cpts: a list of CptOps representing N CPTs
        # thresholds: a list of CptOps representing N-1 thresholds  
        n_intervals = len(cpts)
        u.input_check(sel_type in self.sel_types, "Select type %s not supported" % sel_type)
        u.input_check(len(thresholds)==n_intervals, "Number of thresholds and cpts does not match")
        inputs = [cpts,thresholds,posterior]

        Op.__init__(self,inputs,vars)
        self.var = var
        self.label  = 'sel_cpt_%s_%s_v3' % (var.name,self.strvars)
        self.dims   = d.get_dims(vars)
        self.static = posterior.static
        for cpt in cpts+thresholds:
            if not cpt.static:
                self.static = False
        self.n_intervals = n_intervals
        self.sel_type = sel_type

    def execute(self):
        i0,i1,i2 = self.inputs[0],self.inputs[1],self.inputs[2]
        cpts = [cpt.tensor for cpt in i0]
        thresholds = [thres.tensor for thres in i1]
        posterior = i2.tensor

        with tf.name_scope(self.label):
            posterior = Op.flatten_and_order(posterior,i2.dims)

            if self.sel_type == 'nearest':
                # temp workaround for threshold 0
                print("Use nearest neighbor selection.")
                distances = []
                for thres in thresholds:
                    distance = tf.subtract(posterior,thres)
                    distance = tf.abs(distance)
                    distances.append(distance)
                    # -|P_u - T_i|
                
                gamma = tf.constant(Op.GAMMA_VALUE, dtype=p.float)
                distances = tf.stack(distances,axis=-1)
                weights = tf.math.softmax(tf.negative(tf.multiply(gamma,distances)),axis=-1) 
                # soft nearest neighbor
                cpts = tf.stack(cpts,axis=-1)
                weights = tf.expand_dims(weights,axis=-2)
                self.tensor = tf.reduce_sum(tf.multiply(weights,cpts),axis=-1)

            print("cpt shape: {}".format(self.tensor.shape))
    



 
""" constructs tensor representing a scalar """
class ScalarOp(Op):

    def __init__(self,scalar,vars):
        Op.__init__(self,None,vars)
        self.static = True
        self.scalar = scalar
        self.label  = 'scalar'
        self.dims   = d.get_dims(vars)
        
    def execute(self):
        with tf.name_scope(self.label):
            self.tensor = tf.constant(self.scalar,dtype=p.float)
                              
""" reference, trainable and fixed cpt ops are subclasses of this class """
class CptOp(Op):

    def __init__(self,var,cpt_type,vars):
        Op.__init__(self,None,vars)
        self.var   = var
        self.type  = cpt_type    
        self.label = '%s_%s_%s' % (cpt_type,var.name,self.strvars)  
        self.dims  = d.get_dims(vars) 
        
""" reference to cpt tensor """
class RefCptOp(CptOp):

    def __init__(self,var,cpt_type,tied_cpt_op,vars):
        CptOp.__init__(self,var,cpt_type,vars)
        self.static      = tied_cpt_op.static
        self.tied_cpt_op = tied_cpt_op
    
    def execute(self):
        assert self.tied_cpt_op.tensor is not None # op already executed
        with tf.name_scope(self.label):
            self.tensor = self.tied_cpt_op.tensor  # use tensor of tied-cpt op

''' reference to one individual threshold from multiple thresholds'''
class RefThresholdOp(CptOp):

    def __init__(self,var,cpt_type,vars,thresholds_op):
        CptOp.__init__(self,var,cpt_type,vars)
        self.static = False
        assert cpt_type.startswith('thres')
        #self.type = cpt_type
        self.index = int(cpt_type[len('thres'):])
        self.thresholds_op = thresholds_op

    def execute(self):
        # retrieve one set of thresholds
        assert self.thresholds_op is not None # op already executed
        with tf.name_scope(self.label):
            self.tensor = tf.gather(self.thresholds_op.tensor,indices=self.index-1,axis=-1)
                              
""" constructs tensor representing a non-trainable cpt of a tbn node """
class FixedCptOp(CptOp):

    def __init__(self,var,cpt,cpt_type,vars):
        assert type(cpt) is np.ndarray
        CptOp.__init__(self,var,cpt_type,vars)
        self.static = True
        self.cpt    = cpt # np.ndarray
                
    def execute(self):
        with tf.name_scope(self.label):
            self.tensor = Op.get_cpt_tensor(self.cpt) # returns unique tensor per cpt

""" constructs tensor representing a trainable cpt of a tbn node """
class TrainCptOp(CptOp):
    
    def __init__(self,var,cpt,cpt_type,fix_zeros,vars):
        assert type(cpt) is np.ndarray
        CptOp.__init__(self,var,cpt_type,vars)
        self.static            = False
        self.cpt               = cpt       # np.ndarray
        self.fix_zeros         = fix_zeros # whether to fix cpt zeros when learning cpt
        self.fixed_zeros_count = 0         # number of fixed zeros in weights
        self.weight_id         = count()   # for naming weight variables
        self.weight_variables  = []        # list of trainable variables (weights)
        self.cpt_spec          = None      # information to construct cpt tensor
        
        if fix_zeros: self.fixed_zeros_count = np.count_nonzero(cpt==0)
               
    # returns trainable variable (weight) with length = len(distribution) - zero_count
    def trainable_weight(self,distribution,zero_count=0): # distribution is np array
        id     = next(self.weight_id)
        name   = f'w{id}'
        dtype  = p.float
        length = len(distribution)-zero_count
        shape  = (length,)   # same as distribution.shape if zero_count=0  
        value  = [1.]*length # uniform distribution (try before random distribution)
        
        # the only trainable variables in tac
        weight = tf.Variable(initial_value=value,trainable=True,
                    shape=shape,dtype=dtype,name=name)    

        self.weight_variables.append(weight)
        return weight

    # return trainable variable (weight) with same size as distribution
    def trainable_weight_nd(self,distribution):
        id     = next(self.weight_id)
        name   = f'w{id}_thres'
        dtype  = p.float
        shape = distribution.shape
        value  = np.zeros(shape)
        
        # the only trainable variables in tac
        weight = tf.Variable(initial_value=value,trainable=True,
                    shape=shape,dtype=dtype,name=name)    
                    # initialized with uniform distribution
        self.weight_variables.append(weight)
        return weight

     
    # -returns a nested tuple with the same shape as cpt
    # -entries of the nested tuple are of three types:
    #   -trainable variable to be normalized (trainable distribution) 
    #   -constant (untrainable deterministic distribution)
    #   -trainable variable to be augmented with some zeros and then normalized
    #    (corresponds to a distribution with some fixed zeros but not deterministic)
    def spec(self,cpt):
        if self.type == 'thres':
            return self.trainable_weight_nd(cpt)
            # for threshold
        if cpt.ndim == 1: # cpt is a distribution
            zero_count = np.count_nonzero(cpt==0)
            if self.fix_zeros and zero_count > 0:
                deterministic = np.all((cpt==0) | (cpt==1))
                if deterministic:
                    # deterministic distribution: fixing zeros fixes the distribution
                    fixed_distribution = tf.constant(cpt,dtype=p.float)
                    return [fixed_distribution] # list to distinguish from fully trainable
                # distribution with some fixed zeros but not deterministic
                indices = [(0 if p==0 else i+1) for i,p in enumerate(cpt)]
                weight  = self.trainable_weight(cpt,zero_count)
                return [weight,indices]       # list to distinguish from fully trainable
            return self.trainable_weight(cpt) # fully trainable distribution, no zeros
        return tuple(self.spec(cond_cpt) for cond_cpt in cpt)
        
    # returns a tensor representing a trainable cpt (built from a cpt spec)
    def trainable_cpt(self,spec):
        if self.type == 'thres':
            # trainable variable for thresholds, between 0 and 1
            thres = tf.math.sigmoid(spec)
            #thres = tf.add(tf.multiply(Op.MAX_THRES_VALUE-Op.MIN_THRES_VALUE, thres), Op.MIN_THRES_VALUE)
            return thres
        if type(spec) is tuple: # spec for a conditional cpt
            return tf.stack([self.trainable_cpt(cond_spec) for cond_spec in spec])
        if type(spec) is list:  # distribution with fixed zeros
            if len(spec)==1:    # deterministic distribution (untrainable)
                return spec[0]
            assert len(spec)==2 # distribution with zeros (non-zeros are trainable)
            weight, indices = spec                   # weight for nonzero probabilities
            zero   = tf.constant([-np.inf],dtype=p.float)
            params = tf.concat([zero,weight],axis=0) # weight with leading -inf
            weight = tf.gather(params,indices)       # weight with -inf inserted
            return tf.math.softmax(weight)           # normalized distribution with zeros
        # trainable variable (weight), normalize it so it becomes a distribution
        return tf.math.softmax(spec) # normalize
        
    # defines a spec for constructing the cpt tensor, creating variables in the process
    def execute(self):
        with tf.name_scope(self.label):
            self.cpt_spec = self.spec(self.cpt)
        
    def build_cpt_tensor(self):
        with tf.name_scope(self.label):
            self.tensor = self.trainable_cpt(self.cpt_spec)
        
""" constructs tensor representing evidence on tbn node """
class EvidenceOp(Op):
    
    def __init__(self,var,vars):
        Op.__init__(self,None,vars)
        self.static = False
        self.var    = var
        self.dims   = d.get_dims(vars)
        self.label  = 'evd_%s_%s' % (var.name,self.strvars)
        
    def execute(self):
        name  = self.label
        card  = self.var.card
        shape = (None,card)
        dtype = p.float
        value = [[1.]*card]
        # tf uses -1 for the batch when reshaping, but None when constructing variables
        
        with tf.name_scope(self.label):
            self.tensor = tf.Variable(value,trainable=False,shape=shape,dtype=dtype)

''' construct a tensor representing N-1 trainable thresholds for a testing node
    For each parent state, the N-1 thresholds must be increasing'''

class TrainThresholdsOp(CptOp):
    
    def __init__(self,var,thresholds,vars):
        CptOp.__init__(self,var,"thresholds",vars)
        self.static = False
        self.num_thresholds = len(thresholds)
        self.thresholds = np.stack(tuple(thresholds),axis=-1) # numpy arrays
        self.weight_id = count()
        self.weight_variables = []
        self.thresholds_spec = None # spec to build N-1 increaing thresholds

    # returns trainable variables for N-1 threshold
    def trainable_weight(self,distribution): # distribution is np array
        id     = next(self.weight_id)
        name   = f'w{id}'
        dtype  = p.float
        assert len(distribution) == self.num_thresholds
        if self.num_thresholds == 1:
            length = 2
            # use two weights for single p
        else:
            length = len(distribution)
        shape  = (length,)   # same as distribution.shape if zero_count=0  
        value = [1.0]*length
        
        # the only trainable variables in tac
        weight = tf.Variable(initial_value=value,trainable=True,
                    shape=shape,dtype=dtype,name=name)    

        self.weight_variables.append(weight)
        return weight

    # -returns a nested tuple of the same shape as thresholds
    # each entry of the tuple are weights for N-1 thresholds for each parent state
    def spec(self,thresholds):
        if thresholds.ndim == 1: # thres is a distribution
            return self.trainable_weight(thresholds) # fully trainable distribution, no zeros
        return tuple(self.spec(thres) for thres in thresholds)

    '''
    # returns a tensor representing trainable thresholds from thres spec
    def trainable_thresholds(self,spec):
        if type(spec) is tuple: # spec for a conditional cpt
            return tf.stack([self.trainable_thresholds(cond_spec) for cond_spec in spec])
        else:
            # for one parent state
            thres = tf.math.cumsum(tf.math.abs(spec)) # increasing thresholds
            return tf.math.tanh(thres)
    '''

    # returns a tensor representing increasing trainable thresholds
    def trainable_thresholds(self,spec):
        # normalize 
        def __thresholds(dists):
            if type(dists) is tuple:
                return tf.stack([__thresholds(dist) for dist in dists])
            else:
                # for each parent state
                return tf.math.softmax(dists)

        thresholds = __thresholds(spec) # shape (pcards,N-1)
        arrays = []
        for i in range(self.num_thresholds):
            # for each threshold
            if i == 0:
                thres = tf.gather(thresholds,indices=i,axis=-1)
                arrays.append(thres)
                # T_0 = p_0
            else:
                prev = arrays[-1]
                thres = tf.gather(thresholds,indices=i,axis=-1)
                thres = tf.add(prev, tf.multiply(tf.subtract(1.0,prev),thres))
                # T_i = T_i-1 + (1-T_i-1)*p_i
                arrays.append(thres)
                # obtain increasing thresholds

        result = tf.stack(arrays,axis=-1)
        return result

    # defines a spec for constructing the thresholds tensor, initializing weight variables
    def execute(self):
        with tf.name_scope(self.label):
            self.thresholds_spec = self.spec(self.thresholds)

    def build_thresholds_tensor(self):
        with tf.name_scope(self.label):
            self.tensor = self.trainable_thresholds(self.thresholds_spec)
