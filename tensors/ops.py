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
    gamma             = None  # hold gamma variable used for sigmoid cpt selection
    
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
                op.execute()
                # the weights of each cpt are grouped together
                weight_variables.append(op.weight_variables)
                fixed_zeros_count += op.fixed_zeros_count
            elif type(op) is GammaOp:
                if op.trainable: # need to learn gamma 
                    op.build_gamma_variable()
                    weight_variables.append(op.weight_variables)
#            elif op.static:
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
            if type(op) is TrainCptOp: # already executed and created weight variables
                op.build_cpt_tensor()  # we now need to assemble weights into a cpt
            else: 
                op.execute()           # will not create variables
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
            if not type(op) is TrainCptOp: continue
            op.build_cpt_tensor() # will not create variables
            train_cpt_tensors.append(op.tensor)
        return tuple(train_cpt_tensors)

    # we also build trainable gamma twice: once as part of the ta graph, and then
    # to be able to get their values for saving into file
    @staticmethod
    def trainable_gammas(ops_graph):
        trainable_gamma_tensors = []
        for op in ops_graph.ops: # parents before children
            if type(op) is GammaOp and op.trainable:
                # need to learn gamma
                op.execute() # will not create variable
                trainable_gamma_tensors.append(op.tensor)
        return tuple(trainable_gamma_tensors)
        
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
    def __init__(self,var,cpt1,cpt2,posterior,vars,threshold=None, sel_type="linear", gamma=None):
        inputs = [cpt1,cpt2,posterior]
        if threshold is not None:
            # if threshold included in cpt selection
            inputs.append(threshold)
            if gamma is not None:
                # if sigmoid selection is used
                inputs.append(gamma)
        Op.__init__(self,inputs,vars)
        self.static = posterior.static and cpt1.static and cpt2.static and (threshold is None or threshold.static) 
        self.var    = var
        self.label  = 'sel_cpt_%s_%s' % (var.name,self.strvars)
        self.dims   = d.get_dims(vars)
        self.sel_type = sel_type
        
    def execute(self):
        i1, i2, i3            = self.inputs[:3]
        cpt1, cpt2, posterior = i1.tensor, i2.tensor, i3.tensor
        if len(self.inputs) == 4:
            threshold = self.inputs[3].tensor
        elif len(self.inputs) == 5:
            threshold = self.inputs[3].tensor
            gamma = self.inputs[4].tensor
        with tf.name_scope(self.label):
            if self.sel_type == 'linear':
                #print("Use linear selection for var: {}".format(self.var))
                posterior = Op.flatten_and_order(posterior,i3.dims)
                # add a new dimension so posterior is over family not just parents
                posterior = tf.expand_dims(posterior,-1) # add trivial dimension at end (card 1)
                # selected cpt = (cpt1-cpt2)*posterior + cpt2 (linear combination)
                # selected cpt = cpt1 for posterior=1 and cpt2 for posterior=0
                x   = tf.subtract(cpt1,cpt2)
                y   = tf.multiply(posterior,x) # broadcasting 
                cpt = tf.add(y,cpt2)           # broadcasting
                self.tensor = cpt
            elif self.sel_type == 'threshold':        
                # print("Use threshold selection for var: {}".format(self.var))
                posterior = Op.flatten_and_order(posterior,i3.dims)
                indicator = tf.greater_equal(posterior, threshold)
                indicator = tf.cast(indicator, dtype=p.float) # remember to convert boolean values to numerical types
                indicator = tf.expand_dims(indicator, -1) # add a new dimenstion so indicator is over parents and child
                # selected cpt = cpt1*(posterior > threshold) + cpt2*(1 - posterior > threshold)
                x = tf.multiply(cpt1, indicator)
                y = tf.multiply(cpt2, tf.subtract(1.0, indicator))
                cpt = tf.add(x, y)
                self.tensor = cpt
            elif self.sel_type == 'sigmoid':
                print("Use sigmoid selection for var: {}".format(self.var))
                posterior = Op.flatten_and_order(posterior,i3.dims)
                difference = tf.subtract(posterior, threshold)
                indicator = tf.sigmoid(tf.multiply(gamma, difference))
                indicator = tf.expand_dims(indicator, -1)
                # indicator = 1 / {1 + exp(-y * (posterior - threshold))}
                x = tf.multiply(cpt1, indicator)
                y = tf.multiply(cpt2, tf.subtract(1.0, indicator))
                cpt = tf.add(x, y)
                self.tensor = cpt


 
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
        name   = f'w{id}'
        dtype  = p.float
        shape = distribution.shape
        value  = np.ones(shape)
        
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
            # threshold is a joint distribution
            return self.trainable_weight_nd(cpt)
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
        return tf.math.softmax(spec)
        
    # defines a spec for constructing the cpt tensor, creating variables in the process
    def execute(self):
        with tf.name_scope(self.label):
            self.cpt_spec = self.spec(self.cpt)
        
    def build_cpt_tensor(self):
        with tf.name_scope(self.label):
            self.tensor = self.trainable_cpt(self.cpt_spec)

class GammaOp(Op):
    # construct tensor representing a fixed/trainable gamma parameter of a tbn node
    def __init__(self,var,value,trainable=False):
        Op.__init__(self,inputs=None,vars=())
        self.static = True
        self.dims=None
        self.label = 'gamma_%s' %var.name
        self.value = value
        self.trainable = trainable
        self.weight_variables = []

    def build_gamma_variable(self):
        name = self.label
        with tf.name_scope(self.label):
            weight = tf.Variable(initial_value=[self.value], shape=(1,), dtype=p.float, name=name)
            self.weight_variables.append(weight)
    
    def execute(self):
        with tf.name_scope(self.label):
            if not self.trainable:
                self.tensor = tf.constant(self.value, dtype=p.float)
            else:
                self.tensor = self.weight_variables[0]
                # already build tf variables for gamma

class RefGammaOp(Op):
    # reference to gamma tensor
    def __init__(self,var,tied_gamma_op):
        Op.__init__(self,inputs=None,vars=())
        self.static = True
        self.label = 'gamma_%s' % var.name
        self.tied_gamma_op = tied_gamma_op

    def execute(self):
        assert self.tied_gamma_op.tensor is not None # gamma op already created
        with tf.name_scope(self.label):
            self.tensor = self.tied_gamma_op.tensor
        
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