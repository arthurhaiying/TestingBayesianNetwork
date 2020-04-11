import time
import numpy as np
import tensorflow as tf

import tensors.ops as ops
import train.data as d
import utils.utils as u
  
   
"""
A class that represents an AC, which is extracted from a opsgraph.

The AC is represented in the classical way in terms of individual arithmetic operations.

Used for benchmarking evaluation time against the tensor-based implementation of the AC.

Used also to verify correctness of the tensor-graph implementation of AC, which
is quite complicated.

Assumes opsgraph not trainable, and corresponds to a classical BN (no testing).

The extraction of an AC from an opsgraph is based on representing factors as numpy
arrays whose entries are AC nodes. By overloading operators +, *, /, we can then
use array operations (aka factor operations) to construct an AC (a trace of the
factor operations embedded in the ops graph).
"""

""" Node of an AC """
class Node:
    
    # list of created add/mul/div nodes when executing ops graph
    # topologically sorted
    instances = None
    
    def __init__(self,c1=None,c2=None):
        self.c1     = c1   # AC node
        self.c2     = c2   # AC node
        self.value  = None
        self.type   = None # evidence indicator, parameter, add, mul, div
        self.tensor = None # used when constructing tf graph for AC
                
    # overloading operators +, *, /
    # allows us to use numpy array operations on arrays that contain AC nodes
    def __add__(self,other):
        n = Add(self,other)
        Node.instances.append(n)
        return n
        
    def __mul__(self,other):
        n = Mul(self,other)
        Node.instances.append(n)
        return n
        
    def __truediv__(self,other):
        n = Div(self,other)
        Node.instances.append(n)
        return n

""" Node types """       
class Add(Node):

    def __init__(self,c1,c2):
        Node.__init__(self,c1,c2)
        self.type = 0
        
class Mul(Node):

    def __init__(self,c1,c2):
        Node.__init__(self,c1,c2)
        self.type = 1
        
class Div(Node):

    def __init__(self,c1,c2):
        Node.__init__(self,c1,c2)
        self.type = 2

class Par(Node):

    def __init__(self,value):
        Node.__init__(self)
        self.type = 3
        self.value = np.float32(value) # CPT entry
        
class Evd(Node):

    # var is opsgraph.Var, index is for the value of var corresponding to evidence
    def __init__(self,var,index):
        Node.__init__(self)
        self.type  = 4
        self.var   = var
        self.index = index
          

""" Scalar Arithmetic Circuit """
class ScalarAC:
    
    def __init__(self,opsgraph):
        assert not opsgraph.trainable and not opsgraph.testing
        
        u.show(f'\nConstructing classical AC...')
        start_compile_time = time.perf_counter()
        
        # list of add/mul/div nodes, topologically sorted (bottom up)
        self.nodes = None
        # list of AC nodes representing evidence (one per var/value)
        self.evd_nodes = []
        # list of AC nodes representing parameters (cpt entries)
        self.parameter_nodes = []
        # list of evidence lambdas
        self.lambdas   = [] # each lambda is a tuple of evidence nodes (lambda per var)
        # factor that contains output AC nodes (marginal)
        self.output_factor = None
        # size of AC (number of nodes)
        self.size = None
        
        # maps ops.op into its factor (result of executing operation)
        op2factor = {}
 
        # execution will populate the nodes, lambdas and roots fields
        Node.instances = [] # created add/mul/div will be added to this list
        for op in opsgraph.ops: # bottom up
            factor = self.execute(op,op2factor)
            if type(op) == ops.EvidenceOp:
                self.lambdas.append(factor)
                self.evd_nodes.extend(factor.nodes())
            elif type(op) == ops.FixedCptOp:
                self.parameter_nodes.extend(factor.nodes())
        self.nodes = Node.instances # add/mul/div nodes
        
        # order of lambdas should match order of opsgraph inputs    
        assert opsgraph.evidence_vars == tuple(f.vars[0] for f in self.lambdas)
        self.lambdas = tuple(f.nodes() for f in self.lambdas)
        
        # saving output nodes of AC
        output_op          = opsgraph.ops[-1]
        self.output_factor = op2factor[output_op]
        
        # computing AC size
        self.size = len(self.nodes) + len(self.parameter_nodes) + len(self.evd_nodes)
        
        compile_time = time.perf_counter() - start_compile_time
        
        u.show(f'  AC size {self.size:,}')
        u.show(f'Compile Time: {compile_time:.3f} sec')
    
    # implements op by adding corresponding nodes to the AC   
    # the ac nodes constructed by executing op are gathered into a factor (numpy array)
    # which facilitates the execution of an op     
    def execute(self,op,op2factor):
        remove_batch = lambda vars: tuple(var for var in vars if not var.is_batch)
        
        if type(op) == ops.EvidenceOp:
            var    = op.var
            vars   = (var,) # evidence on this var
            table  = np.array([Evd(var,i) for i in range(var.card)],dtype=Node)
            factor = Factor(table,vars)
        elif type(op) == ops.FixedCptOp:
            vars   = op.vars
            table  = Factor.cpt2table(op.cpt)
            factor = Factor(table,vars)  
        elif type(op) == ops.RefCptOp:
            vars   = op.vars # has different vars than tied cpt
            table  = op2factor[op.tied_cpt_op].table
            factor = Factor(table,vars)
        elif type(op) == ops.MultiplyOp:
            f1, f2 = op2factor[op.inputs[0]], op2factor[op.inputs[1]]
            factor = f1.multiply(f2)
        elif type(op) == ops.MulProOp:
            f1, f2 = op2factor[op.inputs[0]], op2factor[op.inputs[1]]
            vars   = remove_batch(op.vars) # remove batch var
            factor = f1.mulpro(f2,vars)
        elif type(op) == ops.ProjectOp:
            vars   = remove_batch(op.vars) # remove batch var
            f      = op2factor[op.inputs[0]]
            factor = f.project(vars)
        elif type(op) == ops.NormalizeOp:
            f      = op2factor[op.inputs[0]]
            factor = f.normalize()
        elif type(op) == ops.ScaleOp:
            f      = op2factor[op.inputs[0]]
            factor = f.scale()
        else: assert(False) # other operations not relevant to classical ACs
        
        assert factor.vars == remove_batch(op.vars)
        op2factor[op] = factor # cache result of executing operation
        return factor

    "Verification based on AC represented as an array (no batch)"
    
    # evaluates AC on given evidence and checks result against given marginals
    def verify_array(self,evidence,marginals1):
        u.show(f'\nVerifying against classical AC (array)...')
        
        size       = d.evd_size(evidence)
        rows       = d.evd_col2row(evidence)
        marginals2 = []
        
        # evaluation time excludes assertion of evidence
        eval_time = 0 # pure evaluation time (add/mul/div)
        for lambdas in rows:
            self.assert_evidence_array(lambdas)
            marginal, et = self.evaluate_array() # np array
            marginals2.append(marginal)
            eval_time += et
        
        marginals2 = np.array(marginals2,dtype=np.float32)
        u.equal(marginals1,marginals2,tolerance=True)
        
        u.show(f'Evaluation Time: {eval_time:.3f} sec ({1000*eval_time/size:.0f} ms per example)')
        return eval_time, 1
        
    # sets the value of lambda indicators in AC
    # lambdas is a list of lists
    def assert_evidence_array(self,lambdas):
        for lambda_numbers, lambda_nodes in zip(lambdas,self.lambdas):
            for number, node in zip(lambda_numbers, lambda_nodes):
                node.value = np.float32(number)
    
    # evaluates the AC by performing arithmetic operations
    # assumes assert_evidence_array has already been called
    def evaluate_array(self):
        start_eval_time = time.perf_counter()
        for n in self.nodes: # bottom up
            v1, v2 = n.c1.value, n.c2.value
            if   n.type==0: n.value = v1+v2
            elif n.type==1: n.value = v1*v2
            else:           n.value = v1/v2
        eval_time = time.perf_counter() - start_eval_time
        return self.output_factor.value(), eval_time
            
    
    """ Verification based on AC represented as a scalar tensor graph (with batch) """
    
    def compile_tf_graph(self):
        u.show('  compiling tf graph...',end='')
        self.tf_ac = self.tf_graph()
        u.show('done')
        
    def verify_tf_graph(self,evidence,marginals1):
        assert self.tf_ac is not None
        
        size = batch_size = d.evd_size(evidence)
        
        u.show(f'\nVerifying against classical AC (tf graph, batch_size {batch_size}))')
    
        # split lambdas into scalars (with batch)
        evidence = self.split_evidence(evidence)
        
        # tf graph accepts only tensors as input
        evidence = tuple(tf.constant(e,dtype=tf.float32) for e in evidence)
            
        start_eval_time = time.perf_counter()
        for start in range(0,size,batch_size):
            u.show(f'{int(100*start/size):4d}%\r',end='',flush=True)
            stop            = start + batch_size
            evidence_batch  = d.evd_slice(evidence,start,stop)
            marginals_batch = self.tf_ac(*evidence_batch) # evaluating tf graph
            marginals_batch = self.gather_marginals(marginals_batch)
            if start==0:
                marginals2 = marginals_batch
            else:
                marginals2 = np.concatenate((marginals2,marginals_batch),axis=0)
        eval_time = time.perf_counter() - start_eval_time
        
        u.equal(marginals1,marginals2,tolerance=True)
        
        size = d.evd_size(evidence)
        u.show(f'Evaluation Time: {eval_time:.3f} sec ({1000*eval_time/size:.0f} ms per example)')
        return eval_time, batch_size
        
    def tf_graph(self):
        # create variables for AC evidence nodes (with batch)
        # an AC evidence node is a scalar
        # variables must be created outside tf.function
        espec = []
        for n in self.evd_nodes:
            shape    = (None,1)
            dtype    = tf.float32
            var      = tf.Variable([[1.]],trainable=False,shape=shape,dtype=dtype)
            n.tensor = var
            espec.append(tf.TensorSpec(shape=shape,dtype=dtype))
           
        # return callable tf graph that evaluates AC
        return self.tf_marginals.get_concrete_function(*espec)
                
        
    # this function creates all tensors beyond variables
    # it will be compiled into the callable tf graph (ac_func)
    @tf.function
    def tf_marginals(self,*evidence):
        # assert evidence
        for n, e in zip(self.evd_nodes,evidence): 
            n.tensor.assign(e) # n.tensor is a tf variable
        # parameters
        for n in self.parameter_nodes: # n is an AC node
            n.tensor = tf.constant(n.value,dtype=tf.float32) # scalar
        # add/mul/div
        for n in self.nodes: # bottom up
            # v1/v2 are either scalars, or scalars with batch
            t1, t2 = n.c1.tensor, n.c2.tensor
            if   n.type==0: n.tensor = t1+t2
            elif n.type==1: n.tensor = t1*t2
            else:           n.tensor = t1/t2
        # tf_graph outputs
        return [n.tensor for n in self.output_factor.nodes()]   
        
        
    """ Verification based on AC represented as a scalar tf tensors (with batch) """
    
    def verify_tf(self,evidence,marginals1):

        size = batch_size = d.evd_size(evidence)
        
        u.show(f'\nVerifying against classical AC (tf tensors, batch_size {batch_size}))')
    
        # split lambdas into scalars (with batch)
        evidence = self.split_evidence(evidence)
                   
        eval_time = 0 # pure evaluation time (add/mul/div)
        for start in range(0,size,batch_size):
            u.show(f'{int(100*start/size):4d}%\r',end='',flush=True)
            stop            = start + batch_size
            evidence_batch  = d.evd_slice(evidence,start,stop)
            marginals_batch, et = self.evaluate_tf(evidence_batch)
            marginals_batch = self.gather_marginals(marginals_batch)
            if start==0:
                marginals2 = marginals_batch
            else:
                marginals2 = np.concatenate((marginals2,marginals_batch),axis=0)
            eval_time += et
        
        u.equal(marginals1,marginals2,tolerance=True)
        
        size = d.evd_size(evidence)
        u.show(f'Evaluation Time: {eval_time:.3f} sec ({1000*eval_time/size:.0f} ms per example)')
        return eval_time, batch_size
        
    def evaluate_tf(self,evidence):
        # assert evidence
        for n, e in zip(self.evd_nodes,evidence): 
            n.tensor = tf.constant(e,dtype=tf.float32)
        # parameters
        for n in self.parameter_nodes: # n is an AC node
            n.tensor = tf.constant(n.value,dtype=tf.float32) # scalar
        start_eval_time = time.perf_counter()
        # add/mul/div
        for n in self.nodes: # bottom up
            # v1/v2 are either scalars, or scalars with batch
            t1, t2 = n.c1.tensor, n.c2.tensor
            if   n.type==0: n.tensor = t1+t2
            elif n.type==1: n.tensor = t1*t2
            else:           n.tensor = t1/t2
        eval_time = time.perf_counter() - start_eval_time
        # tf_graph outputs
        return [n.tensor for n in self.output_factor.nodes()], eval_time 
        
    """ Verification based on AC represented as a scalar numpy arrays (with batch) """
    
    def verify_numpy(self,evidence,marginals1):
        
        size = batch_size = d.evd_size(evidence)
        
        u.show(f'\nVerifying against classical AC (numpy arrays, batch_size {batch_size})')
        
        # split lambdas into scalars (with batch)
        evidence = self.split_evidence(evidence)

        eval_time = 0 # pure evaluation time (add/mul/div)
        for start in range(0,size,batch_size):
            u.show(f'{int(100*start/size):4d}%\r',end='',flush=True)
            stop            = start + batch_size
            evidence_batch  = d.evd_slice(evidence,start,stop)
            marginals_batch, et = self.evaluate_numpy(evidence_batch)
            marginals_batch = self.gather_marginals(marginals_batch)
            if start==0:
                marginals2 = marginals_batch
            else:
                marginals2 = np.concatenate((marginals2,marginals_batch),axis=0)
        eval_time += et
        
        u.equal(marginals1,marginals2,tolerance=True)
        
        size = d.evd_size(evidence)
        u.show(f'Evaluation Time: {eval_time:.3f} sec ({1000*eval_time/size:.0f} ms per example)')
        return eval_time, batch_size
        
    def evaluate_numpy(self,evidence):
        # assert evidence
        for n, e in zip(self.evd_nodes,evidence): 
            n.tensor = e.astype(np.float32) # scalar with batch
        # parameters
        for n in self.parameter_nodes: # n is an AC node
            n.tensor = n.value # scalar, np.float32
        start_eval_time = time.perf_counter()
        # add/mul/div
        for n in self.nodes: # bottom up
            # v1/v2 are either scalars, or scalars with batch
            t1, t2 = n.c1.tensor, n.c2.tensor
            if   n.type==0: n.tensor = t1+t2
            elif n.type==1: n.tensor = t1*t2
            else:           n.tensor = t1/t2
        eval_time = time.perf_counter() - start_eval_time
        return [n.tensor for n in self.output_factor.nodes()], eval_time
        
        
    # split evidence into scalars (with batch)
    def split_evidence(self,evidence):
        evd = []
        for col in evidence:
            for i in range(col.shape[1]):
                evd.append(col[:,i])
        return evd
        
    # put together scalar marginals (with batch) into a distribution (with batch)
    def gather_marginals(self,marginals):
        return np.stack(marginals,axis=1)
        
        
""" Factors """
class Factor:

    get_value = np.vectorize(lambda n: n.value) # n is an AC node
    cpt2table = np.vectorize(lambda p: Par(p))  # p is a probability
    
    # table is a numpy array of Nodes
    def __init__(self,table,vars):
        assert type(table) == np.ndarray and type(vars) == tuple
        assert all(not var.is_batch for var in vars)
        assert u.sorted(tuple(v.id for v in vars))
        assert table.shape == tuple(var.card for var in vars)
        self.vars  = vars
        self.table = table 
        self.rank  = len(self.vars)
        
    # returns a list of the nodes in a factor
    def nodes(self):
        linear = self.table.reshape((self.table.size,))
        return list(linear)
        
    # returns the values of nodes in a factor (as np array of the same shape)
    def value(self):
        return Factor.get_value(self.table)

    # project self on vars
    def project(self,vars):
        vset  = set(vars)
        assert vset <= set(self.vars)
        axes  = tuple(i for i, var in enumerate(self.vars) if var not in vset)
        table = np.sum(self.table,axis=axes)
        return Factor(table,vars)
        
    # multiplies self with factor (uses broadcasting)
    def multiply(self,factor):
        table1, vars1 = self.table, self.vars
        table2, vars2 = factor.table, factor.vars
        if vars1==vars2: return Factor(table1*table2,vars1)
        varset1, varset2 = set(vars1), set(vars2)
        vars  = list(varset1 | varset2)
        vars.sort()
        vars = tuple(vars)
        shape1 = tuple((v.card if v in varset1 else 1) for v in vars)
        shape2 = tuple((v.card if v in varset2 else 1) for v in vars)
        table1 = np.reshape(table1,shape1) # adds trivial dimensions
        table2 = np.reshape(table2,shape2) # adds trivial dimensions
        table  = table1 * table2
        return Factor(table,vars)
        
    # multiplies self with factor then projects result on var
    def mulpro(self,factor,vars):
        assert set(vars) <= set(self.vars) | set(factor.vars)
        f = self.multiply(factor)
        return f.project(vars)
        
    # normalizes factor
    def normalize(self):
        axes  = tuple(i for i in range(self.rank)) # sumout all axes
        norm  = np.sum(self.table,axis=axes,keepdims=True)
        table = self.table/norm
        return Factor(table,self.vars)
        
    # not exactly like tf graphs (which is more selective when to scale)
    def scale(self):
        return self.normalize()