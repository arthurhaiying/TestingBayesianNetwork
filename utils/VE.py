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
            self.has_batch = False
            
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

    # maximize out var from factor
    def maxout(self,var):
        axis  = self.vars.index(var)
        table = np.max(self.table, axis=axis)
        vars  = tuple(v for v in self.vars if v != var)
        return Factor(table, vars)

        
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




################################################################################################
#  MAP 
################################################################################################


def VE_MAP(bn, map_vars, evidence_vars, evidence):
    """ 
    Computes the probability of the most likely instantiation of map variables given evidence using VE
    map_vars: list of map variable names
    evidence_vars: list of evidence variable names
    evidence: list of numpy arrays that corresponds to evidence instantiations
    returns: the MAP probability max_m Pr(m, e)
    assumes map_vars do not overlap with evidence_vars
    """
    print("\nRunning VE-MAP...")
    assert not bn.testing and len(evidence_vars) == len(evidence)
    assert not set(map_vars) & set(evidence_vars)

    # prune network nodes with respect to map nodes
    map_nodes = [bn.node(m) for m in map_vars]
    connected = bn.connected_nodes(map_nodes)
    assert all(n in connected for n in map_nodes)
    evidence_vars_, evidence_ = [], []
    for e,evd in zip(evidence_vars, evidence):
        node = bn.node(e)
        if node in connected:
            evidence_vars_.append(e)
            evidence_.append(evd)

    # make Vars for nodes
    node2var = {n:Var(bn_node=n) for n in connected}
    batch_size = data.evd_size(evidence_)
    batch_var = Var(batch_size=batch_size)
    # make factors
    cpts = []
    for node in connected:
        cpt = node.tabular_cpt()
        family = node.family
        vars = tuple(node2var[n] for n in family)
        cpt_factor = Factor(cpt, vars, sort=True)
        cpts.append(cpt_factor)
    indicators = []
    for e,evd in zip(evidence_vars_, evidence_):
        node = bn.node(e)
        var = node2var[node]
        vars = (batch_var, var)
        evd_factor = Factor(evd, vars)
        indicators.append(evd_factor)

    # get elimination order
    # TODO: constrained minfill order
    order,_,_,_ = bn.elm_order('minfill')
    order = list(filter(lambda n: n in connected, order))  # pruned bn
    sum_order = [node2var[n] for n in order if n not in map_nodes]
    max_order = [node2var[n] for n in order if n in map_nodes]
    elm_order = sum_order + max_order

    # index factors
    var2factors = {var:set() for var in elm_order}  # map var to factors containing this var
    def index(factor):
        vars = factor.tvars
        for var in elm_order:
            if var in vars:
                var2factors[var].add(factor) 
                return
        raise RuntimeError(f"Cannot index factor {vars}.")
    # before eliminating, add CPTs and indicators to var2factors
    for factor in cpts+indicators: 
        index(factor)

    # eliminate variables
    # first sum out non-map variables and then max out map variables
    one = Factor.one()
    result = None
    for i,var in enumerate(elm_order):
        factors = var2factors[var]
        prod = one
        for f in factors:
            prod = prod.multiply(f)
        del var2factors[var]
        if var in sum_order:
            factor = prod.sumout(var)
        else: # var in max_order
            factor = prod.maxout(var)
        if i == len(elm_order)-1:
            result = factor
        else:
            index(factor)

    assert result.vars == (batch_var,)
    print("done.")
    return result.table # numpy array
        


""" 
An ExtFactor is an extended factor which assigns to each instantiation both a number and an instantiation
An ExtFactor has (table, vars, inst_table, inst_vars) where
- table: a numpy array of numbers, where the axes corresponds to vars
- inst_table: a numpy array of instantiations, where the axes corresponds to vars
- inst_vars: each instantiation corresponds to inst_vars
"""
class ExtFactor(Factor):

    def __init__(self, table, vars, *, sort=True, inst_table=None, inst_vars=None):

        super().__init__(table, vars, sort)
        if inst_table is None:  # empty instantiation
            assert inst_vars is None 
            self.inst_vars = ()
            shape = tuple(v.card for v in vars) + (0,)
            self.inst_table = np.empty(shape=shape, dtype=int)
            return

        assert inst_vars is not None 
        assert isinstance(inst_table, np.ndarray)    # ndarray
        if inst_table.shape[-1] != len(inst_vars):
            raise ValueError("The last axis of inst_table does not match the number of inst_vars.")
        if sort:
            inst_table, inst_vars = self.__sort_inst(inst_table, inst_vars)
        assert u.sorted(tuple(v.id for v in vars))
        self.inst_table = inst_table
        self.inst_vars = inst_vars
        self.num_inst_vars = len(inst_vars)



    # for printing
    def __str__(self):
        vars  = 'f(' + ','.join([f'{v.name}' for v in self.vars]) + ')'
        inst_vars = '(' + ','.join([f'{v.name}' for v in self.vars]) + ')'
        str = ( f"vars: {vars}\n"
                f"inst vars: {inst_vars}\n"
                f"table: {self.table}\n"
                f"inst table: {self.inst_table}\n"
              )
        return str

        
    # sort inst_vars and the last axis of inst_table accordingly
    def __sort_inst(self, inst_table, inst_vars):
        inst_vars_sorted = sorted(inst_vars)
        indices = [inst_vars.index(v) for v in inst_vars_sorted]
        inst_table = np.take(inst_table, indices, axis=-1)
        return inst_table, tuple(inst_vars_sorted)

    @staticmethod
    def one():
        return ExtFactor(1.,tuple())

    # normalizes factor that has batch
    def normalize(self):
        factor = super().normalize()
        table, vars = factor.table, factor.vars
        result = ExtFactor(table, vars, inst_table=self.inst_table, 
                           inst_vars=self.inst_vars)
        return result

    # sums out var from factor
    def sumout(self, var):
        factor = super().sumout(var)
        table, vars = factor.table, factor.vars
        result = ExtFactor(table, vars, inst_table=self.inst_table,
                           inst_vars=self.inst_vars)
        return result

    # maximize out var from factor
    # for each instantiation of the remaining vars, record the value of var that attains 
    # the maximal probability 
    def maxout(self, var):
        axis = self.vars.index(var)
        vars = tuple(v for v in self.vars if v != var) 
        table = self.table
        table = np.swapaxes(table, 0, axis)     # var is now at the first axis 
        aindex = np.argmax(table, axis=0)       # store max indices of var
        table = np.max(table, axis=0)  
        # update instantiations 
        inst_vars = self.inst_vars + (var,)
        inst_table = self.inst_table
        inst_table = np.swapaxes(inst_table, 0, axis) # var is now at the first axis
        dims = [range(v.card) for v in vars]  
        index_arrays = np.ix_(*dims)
        index_arrays = (aindex, *index_arrays)
        inst_table = inst_table[index_arrays]   # take instantiations along max indices of var
        inst_table = np.concatenate([inst_table, aindex[...,np.newaxis]], axis=-1)
        inst_table, inst_vars = self.__sort_inst(inst_table, inst_vars)
        result = ExtFactor(table, vars, inst_table=inst_table, 
                           inst_vars=inst_vars)
        return result


    # multiplies self with factor
    # assume self and extfactor do not have overlapping var in their inst_vars
    def multiply(self, extfactor):
        table1, vars1 = self.table, self.vars
        table2, vars2 = extfactor.table, extfactor.vars
        inst_table1, inst_vars1 = self.inst_table, self.inst_vars
        inst_table2, inst_vars2 = self.inst_table, self.inst_vars
        if vars1==vars2:
            table = table1 * table2
            vars = vars1
            inst_vars = inst_vars1 + inst_vars2
            inst_table = np.concatenate([table1,table2], axis=-1)
            inst_table, inst_vars = self.__sort_inst(inst_table, inst_vars)
            result = ExtFactor(table, vars, inst_table=inst_table, 
                               inst_vars=inst_vars)
            return result

        # if self and extfactor have different shape
        varset1, varset2 = set(vars1), set(vars2)
        vars = list(varset1 | varset2)
        vars.sort()
        vars = tuple(vars)
        shape1 = tuple((v.card if v in varset1 else 1) for v in vars)
        shape2 = tuple((v.card if v in varset2 else 1) for v in vars)
        shape  = tuple(v.card for v in vars)
        table1 = np.reshape(table1,shape1) # adds trivial dimensions
        table2 = np.reshape(table2,shape2) # adds trivial dimensions
        table  = table1 * table2
        # update instantiations
        # TODO: concatenate ndarrays with broadcasting
        assert not set(inst_vars1) & set(inst_vars2)
        inst_vars = inst_vars1 + inst_vars2
        # broadcast inst_table to targeted shape
        inst_size1 = len(inst_vars1)
        inst_size2 = len(inst_vars2)
        inst_table1 = np.reshape(inst_table1, newshape=(*shape1, -1)) # adds trivial dimensions
        inst_table2 = np.reshape(inst_table2, newshape=(*shape2, -1)) # adds trivial dimensions
        inst_table1 = np.broadcast_to(inst_table1, shape=(*shape, inst_size1)) 
        inst_table2 = np.broadcast_to(inst_table2, shape=(*shape, inst_size2))
        inst_table = np.concatenate([inst_table1, inst_table2], axis=-1)
        inst_table, inst_vars = self.__sort_inst(inst_table, inst_vars)
        result = ExtFactor(table, vars, inst_table=inst_table, 
                           inst_vars=inst_vars)
        return result




def VE_MAP2(bn, map_vars, evidence_vars, evidence, return_inst=True):
    """ 
    Computes the probability of the most likely instantiation of map variables given evidence using VE.
    Assumes map_vars do not overlap with evidence_vars.
    Args:
        map_vars: list of map variable names
        evidence_vars: list of evidence variable names
        evidence: list of numpy arrays that corresponds to evidence instantiations
        return_inst: if true, also returns the map instantiations
    Returns: 
        table: the MAP probability max_m Pr(m, e)
        instantiations: the MAP instantiations argmax_m Pr(m, e). Only provided if return_inst is true
    """
    print("\nRunning VE-MAP...")
    assert not bn.testing and len(evidence_vars) == len(evidence)
    assert not set(map_vars) & set(evidence_vars)


    # prune network nodes with respect to map nodes
    map_nodes = [bn.node(m) for m in map_vars]
    connected = bn.connected_nodes(map_nodes)
    assert all(n in connected for n in map_nodes)
    evidence_vars_, evidence_ = [], []
    for e,evd in zip(evidence_vars, evidence):
        node = bn.node(e)
        if node in connected:
            evidence_vars_.append(e)
            evidence_.append(evd)

    # make Vars for nodes
    node2var = {n:Var(bn_node=n) for n in connected}
    batch_size = data.evd_size(evidence_)
    batch_var = Var(batch_size=batch_size)

    # choose factor type 
    FactorType = ExtFactor if return_inst else Factor
    # make factors
    cpts = []
    for node in connected:
        cpt = node.tabular_cpt()
        family = node.family
        vars = tuple(node2var[n] for n in family)
        cpt_factor = FactorType(cpt, vars, sort=True)
        cpts.append(cpt_factor)
    indicators = []
    for e,evd in zip(evidence_vars_, evidence_):
        node = bn.node(e)
        var = node2var[node]
        vars = (batch_var, var)
        evd_factor = FactorType(evd, vars)
        indicators.append(evd_factor)

    # get elimination order
    # TODO: constrained minfill order
    order,_,_,_ = bn.elm_order('minfill')
    order = list(filter(lambda n: n in connected, order))  # pruned bn
    sum_order = [node2var[n] for n in order if n not in map_nodes]
    max_order = [node2var[n] for n in order if n in map_nodes]
    elm_order = sum_order + max_order

    # index factors
    var2factors = {var:set() for var in elm_order}  # map var to extfactors containing this var
    def index(factor):
        vars = factor.tvars
        for var in elm_order:
            if var in vars:
                var2factors[var].add(factor) 
                return
        raise RuntimeError(f"Cannot index extfactor {vars}.")
    # before eliminating, add CPTs and indicators to var2extfactors
    for factor in cpts+indicators: 
        index(factor)

    # eliminate variables
    # first sum out non-map variables and then max out map variables
    one = ExtFactor.one()
    result = None
    for i,var in enumerate(elm_order):
        extfactors = var2factors[var]
        prod = one
        for f in extfactors:
            prod = prod.multiply(f)
        del var2factors[var]
        if var in sum_order:
            factor = prod.sumout(var)
        else: # var in max_order
            factor = prod.maxout(var)
        if i == len(elm_order)-1:
            result = factor
        else:
            index(factor)

    # recover map solutions
    assert result.vars == (batch_var,)
    if not return_inst:
        print("done.")
        return result.table 
    else:
        assert len(result.inst_vars) == len(map_nodes)
        table = result.table
        mvars = [node2var[n] for n in map_nodes]
        indices = tuple(result.inst_vars.index(v) for v in mvars)
        instantiations = np.take(result.inst_table, indices=indices, axis=-1)
        print("done.")
        return table, instantiations 



    



    


        


        
        











    














    











    

    

    
