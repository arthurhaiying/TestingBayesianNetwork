import numpy as np
from numbers import Number

import train.data as data
import utils.utils as u
  
        
"""
A basic variable elimination algorithm for sanity checks.
Implements factors as numpy arrays and factor operations using numpy operations.

Interface: posteriors(bn,inputs,output,evidence).

Evidence is a list of ndarrays which correspond to inputs (each is a batch of lambdas)

Works only for non-testing tbns.
"""

""" Var is an abstraction of a tbn node, or a batch. """
class Var:
    def __init__(self,bn_node=None,batch_size=None):
        assert bn_node or batch_size
        assert not bn_node or not batch_size
        if batch_size:
            self.id   = -1
            self.name = 'batch'
            self.card = batch_size
        else:
            self.id   = bn_node.id
            self.name = bn_node.name
            self.card = bn_node.card
        self.batch = self.id == -1
        
    # for sorting
    def __lt__(self,other):
        return self.id < other.id
        
    # for printing
    def __str__(self):
        return self.name
        
"""   
A table is a numpy array or a scalar.
A factor is a pair (table,vars), where vars are ordered and correspond to table axes.
"""
class Factor:
    
    def __init__(self,table,vars,sort=False):
    
        if sort and not u.sorted(tuple(var.id for var in vars)): 
            # used only for cpts as they may not be sorted
            table, vars = self.__sort(table,vars)
            
        assert type(table) is np.ndarray or (isinstance(table,Number) and not vars)
        assert u.sorted(tuple(var.id for var in vars))
        assert isinstance(table,Number) or table.shape == tuple(var.card for var in vars)
        
        self.table = table # ndarray
        self.vars  = vars  # sequence of ordered vars
        self.rank  = len(vars)
        
        # tvars are variables without batch
        if vars and vars[0].batch:
            self.tvars     = vars[1:]
            self.has_batch = True
        else:
            self.tvars     = vars
            self.has_bacth = False
            
        self.is_scalar = not self.vars or not self.tvars # no tbn nodes
        
    # for printing
    def __str__(self):
        vars  = 'f(' + ','.join([f'{var.name}' for var in self.vars]) + ')'
        return vars + '\n' + str(self.table)

    # sort vars and transpose table accordingly
    def __sort(self,table,vars):
        svars = list(vars)
        svars.sort()
        axis  = tuple(vars.index(v) for v in svars)
        table = np.transpose(table,axis)
        return table, tuple(svars)
        
    @staticmethod
    def one():
        return Factor(1.,tuple())
        
    # normalizes factor that has batch
    def normalize(self):
        assert self.has_batch
        axes  = tuple(a for a in range(1,self.rank)) # sumout all but first axis (batch)
        norm  = np.sum(self.table,axis=axes,keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            table = self.table/norm
        table[np.isnan(table)] = 0 # converts nan to 0 to match TAC
        return Factor(table,self.vars)
    
    # sums out var from factor
    def sumout(self,var):
        axis  = self.vars.index(var)
        table = np.sum(self.table,axis=axis)
        vars  = tuple(v for v in self.vars if v != var)
        return Factor(table,vars)
        
    # project factor on vars
    def project(self,vars):
        vars1, vars2 = set(vars), set(self.vars)
        assert vars1 <= vars2
        f = self
        for var in vars2-vars1:
            f = f.sumout(var)
        return f
    
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
        return Factor(table, vars)
    

""" 
Returns posteriors on output given evidence on inputs: evidence has a batch.
inputs: list of of var names.
output: var name.
evidence: list of ndarrays that correspond to inputs.
Assumes output is connected to at least one input
"""
def posteriors(bn,inputs,output,evidence):
    u.show('\nRunning VE...',end='',flush=True)
    assert not bn.testing and len(inputs)==len(evidence)

    # we will perform elimination only on nodes that are connected to output
    qnode = bn.node(output) # query node
    nodes = qnode.connected_nodes() # set
    assert qnode in nodes
    
    # identify inputs and evidence connected to query node
    evidence_        = evidence
    enodes, evidence = [], []
    for i, e in zip(inputs,evidence_):
        n = bn.node(i) # evidence node
        if n in nodes: # connected to query
            enodes.append(n)
            evidence.append(e) # e is a batch of lambdas for node n
    assert enodes and evidence # output must be connected to some input
    
    # maps bn node to Var
    node2var   = {n:Var(bn_node=n) for n in nodes} 
    nodes2vars = lambda nodes_: tuple(node2var[n] for n in nodes_)
    
    # construct batch Var
    batch_size = data.evd_size(evidence)
    batch_var  = Var(batch_size=batch_size)
    
    # get elimination order
    order,_ ,_,_ = bn.elm_order('minfill')
    elm_order  = tuple(node2var[n] for n in order if n != qnode and n in nodes)
    
    # bn factors
    evd_factor = lambda evd, node: Factor(evd,(batch_var,node2var[node]))
    cpt_factor = lambda cpt, node: Factor(cpt,nodes2vars(node.family),sort=True)
    indicators = tuple(evd_factor(evd,n) for evd,n in zip(evidence,enodes))
    cpts       = tuple(cpt_factor(n.tabular_cpt(),n) for n in nodes)
    query_var  = node2var[qnode]
    
    # indexing factors for lookup during elimination
    # factor.tvars exclude the batch var
    scalars     = set()   # scalar factors (have no vars)
    var2factors = {var:set() for var in elm_order} # maps var to factors containing var
    var2factors[query_var] = set()
    
    def index(factor):    # add factor to pool
        if factor.is_scalar:
            scalars.add(factor)
        else:
            for var in factor.tvars: var2factors[var].add(factor)
            
    def remove(factors):  # remove factors from pool
        for f in set(factors): # copy since factors may be equal to some var2factors[var]
            assert not f.is_scalar
            for var in f.tvars: var2factors[var].remove(f)
            
    def get_factors(var): # returns factors that contain var
        factors = var2factors[var]
        assert factors
        return factors
        
    def verify_elm(f):    # verify pool at end of elimination
        assert all(not factors for var,factors in var2factors.items() if var != query_var)
        assert not scalars == bn.is_connected()
    
    # we are about to start eliminating vars: index them first
    for factor in indicators: index(factor)
    for factor in cpts:       index(factor)
            
    # eliminate vars
    one = Factor.one() # identity factor for multiplication
    for var in elm_order:
        factors = get_factors(var) # factors that contain var
        factor = one
        for f in factors: factor = factor.multiply(f)
        factor = factor.sumout(var)
        remove(factors)
        index(factor)
    
    verify_elm(factor)
    
    factor = one
    for f in get_factors(query_var): factor = factor.multiply(f)
    for f in scalars:                factor = factor.multiply(f)
    assert factor.has_batch and factor.tvars == (query_var,)
        
    factor = factor.normalize()
    u.show('done.')
    return factor.table # ndarray
    
