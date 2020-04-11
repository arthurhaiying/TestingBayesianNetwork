import numpy as np
from random import uniform
from itertools import product
from types import FunctionType
from inspect import signature

import utils.utils as u

"""
Utilities for manipulating cpts. 

Exposes the following functions, used to prepare tbn nodes for inference (see node.py):
  # random():
    Returns a random cpt as np array.
  # expand(): 
    Expands various cpt forms, including python functions, into cpts as np arrays.
  # prune(): 
    Prunes node values in addition to expanding and pruning its cpts.
"""

# returns a random cpt for a node with cardinality card 
# and whose parants have cardinalities cards
def random(card,cards):
    if not cards:
        cpt = [uniform(0,1) for _ in range(card)]
        return np.array(cpt)/sum(cpt) # normalize
    else:
        cpt = [random(card,cards[1:]) for _ in range(cards[0])]
        return np.array(cpt)
      
      
# -expands a cpt specified by a list or python function into np array 
# -returns the cpt as is if already an np array   
def expand(node,cpt,cpt_type):
    normalized = lambda cpt: np.allclose(1.,np.sum(cpt,axis=-1)) 
    tabular    = True # whether original cpt is a list or np array
    
    if type(cpt) is list:
        cpt = np.array(cpt)
    elif type(cpt) is FunctionType:
        tabular = False
        fn_type = __function_type(node,cpt)
        if fn_type == 'function': 
            cpt = __expand_fcpt(node,cpt,node.parents)
        else:
            assert fn_type == 'constraint'
            cpt = __expand_ccpt(node,cpt,node.parents)
    else:
        u.check(type(cpt) is np.array,
            f'{cpt_type} of node {node.name} is not a list, np array or python function:\n  {cpt}',
            f'specifying TBN cpt')
            
    assert type(cpt) is np.ndarray
    assert cpt.shape == node.shape()
    u.check(normalized(cpt),
            f'{cpt_type} of node {node.name} is not normalized:\n  {cpt}',
            f'specifying TBN cpt')
            
    cpt.flags.writeable = False # read only
    return cpt, tabular # np array


# convenience function for applying various transformations to node cpts                
def __apply(node,fn):
    if node.testing:
        node._cpt1 = fn(node.cpt1,'cpt1')
        node._cpt2 = fn(node.cpt2,'cpt2')
    else:
        node._cpt  = fn(node.cpt,'cpt')
        
# -expands node cpts into np.ndarray: from nested list, np arrays or python functions
# -python function may specify a functional cpt or a constraint cpt
# -prunes impossible values of node
# -prunes cpt rows that correspond to impossible states of parents
# -prunes cpt distributions to remove values that always have zero probability
# -disconnects node its from single-value parents
# -disconnects node from its parents if it has a single value
def set_cpts(node):

    # cpt transformations
    def fn1(cpt,cpt_type):
        cpt, tabular = expand(node,cpt,cpt_type)
        if tabular:
            return __prune_rows(node,cpt)
        return cpt
    def fn2(cpt,cpt_type):
        return __prune_distributions(node,cpt)
    def fn3(cpt,cpt_type):
        axes = tuple(i for i,p in enumerate(node.parents) if p.card==1)
        return np.squeeze(cpt,axis=axes)
    def fn4(cpt,cpt_type):
        return np.array([1.])
    
    # STEP 1: expand cpt and prune its rows
    # -expand cpts into np arrays
    # -prune cpt rows which correspond to infeasible instantiations of parents
    # -infeasible rows of cpts expanded from python functions already pruned
    __apply(node,fn1)

    # STEP 2: prune node values and distributions
    # -prune node values: ones that are guaranteed to always have zero probability
    # -zeros of a cpt need to be fixed if used for inferring impossible values
    if node.fixed_cpt or node.fixed_zeros:
        did_prune = __prune_values(node)
        # if node lost values, prune the distributions of its cpts
        if did_prune: __apply(node,fn2)
       
    # STEP 3: disconnect node from some parents
    # -if we have single-value parents, remove corresponding (trivial) axes from cpts
    # -this corresponds to deleting network edges outgoing from these parents
    if any(p.card==1 for p in node.parents):
        __apply(node,fn3) # before pruning parents
        node._parents = [p for p in node.parents if p.card > 1]
        node._family  = [*node.parents,node]

    # STEP 4: disconnect node from all parents
    # if node has single value, disconnect it from parents (node independent of parents)
    if node.card==1 and node.parents: 
        __apply(node,fn4)
        node._parents = []     
        node._family  = [node]
        
    # children of parents need not be updated in STEPS 3 & 4 since the node has not
    # been added yet to the network (children of parents are set when adding the node)
        
    # sanity checks
    check = lambda cpt: type(cpt) is np.ndarray and cpt.shape == node.shape() and \
                np.allclose(1.,np.sum(cpt,axis=-1)) # normalized
    if node.testing:
        assert check(node.cpt1) and check(node.cpt2)
    else:
        assert check(node.cpt)
            
# prunes values of node that are guaranteed to have a zero probability under
# any feasible instantiation of its parents
def __prune_values(node):
    # find out which values can be pruned
    if node.testing:
        pvalues1 = __infeasible_values(node,node.cpt1)
        pvalues2 = __infeasible_values(node,node.cpt2)
        pvalues  = pvalues1 & pvalues2
    else:
        pvalues  = __infeasible_values(node,node.cpt)
    # pvalues is a set
    
    # -update node values and cardinality
    # -keep track of indices of unpruned values if some values are pruned
    if pvalues: # some values have been pruned
        # order of active values must be preserved
        node._values     = tuple(v for v in node.values if v not in pvalues)
        node._card       = len(node.values)
        # save indices of unpruned values into the set of original values
        node._values_idx = tuple(i for i,v in enumerate(node._values_org) if v not in pvalues)
        return True
    return False
    
# -returns infeasible values of node according to given cpt
# -a node value is infeasible if the cpt assigns it a zero probability under 
#  every 'feasible' instantiation of the node parents
# -values may be infeasible due to two reasons: 
#  (a) the user may have provided loose node values, which can happen when
#  the cpt is specified using a python function as this may preclude careful
#  analysis of the possible values of a node.
#  (b) all node values may be feasible when considering all parent instantiations,
#  but some may be infeasible when considering only feasible states of the parents
# -input and output cpts of the following function are np arrays
def __infeasible_values(node,cpt):
    assert node.fixed_cpt or node.fixed_zeros
    # axes: all but last axis of cpt
    # max:  pointwise max of distributions in the cpt (across parent instantiations)
    # max:  boolean vector with True for values that have zero probability in
    #       every distribution in the cpt (infeasible values)
    axes = tuple(range(len(node.parents)))
    max  = np.max(cpt,axis=axes)
    mask = (max == 0)
    return set(v for v,m in zip(node.values,mask) if m) # could be empty

# -removes cpt rows that correspond to infeasible parent instantiations
# -this is called only on cpts specified using lists and np arrays
# -cpts specified using python functions have no such rows as they are
#  excluded when the python function is expanded into an np array
#  (see __expand_fcpt() and __expand_ccpt())
# -input and output cpts in the following function are np arrays
def __prune_rows(node,cpt):
    for i, p in enumerate(node.parents):
        if cpt.shape[i] > p.card: 
            assert p._values_idx # parent p lost values
            # some values of axis i correspond to infeasible states of p
            # p._values_idx contains the indices of feasible states of p
            cpt = np.take(cpt,p._values_idx,axis=i)
    assert cpt.shape == node.shape()
    return cpt
   
# -removes entries that are zero across all cpt rows (parent instantiations)
# -input and output cpts in the following function are np arrays
def __prune_distributions(node,cpt):
    assert node._values_idx # node lost values
    cpt = np.take(cpt,node._values_idx,axis=-1)
    assert cpt.shape == node.shape()
    return cpt
        
# -constructs a tabular, 'functional cpt' from python function fn
# -node parents and the arguments of function fn must match    
# -returns cpt as np array
def __expand_fcpt(node,fn,parents,inputs=[]):
    if not parents: # all parents have been instantiated
        value = fn(*inputs) # unique value of node when parents=inputs
        cpt   = [(1. if v==value else 0.) for v in node.values] 
        return np.array(cpt)
    else: 
        parent  = parents[0]  # values of parent may have been pruned
        parents = parents[1:] # could be empty
        cpt     = [__expand_fcpt(node,fn,parents,[*inputs,v]) \
                    for v in parent.values]
        return np.array(cpt)
        
# -constructs a tabular, 'constraint cpt' from a python function ct
# -node family and the arguments of function ct must match    
# -returns cpt as np array
# -note: ct(.)=True leads to a uniform cpt
def __expand_ccpt(node,ct,parents,inputs=[]):
    if not parents: # all parents have been instantiated
        compatible = lambda v: ct(*inputs,v) # v compatible with parents instantiation
        cpt = [(1. if compatible(v) else 0.) for v in node.values]
        return np.array(cpt)/sum(cpt) # normalize
    else: 
        parent  = parents[0]  # values of node may have been pruned
        parents = parents[1:] # could be empty
        cpt     = [__expand_ccpt(node,ct,parents,[*inputs,v]) \
                    for v in parent.values]
        return np.array(cpt)
        
# -determines the type of cpt that is specified using a python function:
#   -'function:' a python function that computes the value of node given
#     values of its parents (arguments correspond to node parents)
#   -'constraint:' a python function that returns True/False for every
#     instantiation of the node family (arguments correspond to node family)
# -validates the python function:
#   -allowed for functional cpts : *args   (varying number of parents)
#   -allowed for constraint cpts:  *args,i (varying number of parents)
#   -free variables are not allowed
#   -**kwargs not allowed, keyword arguments are ignored
def __function_type(node,cpt):
    type(cpt) is FunctionType
    # functions that specify cpts cannot have free variables
    freevars = cpt.__code__.co_freevars
    u.check(not freevars,
        f'cpt function for node {node.name} has free variables, {u.unpack(freevars)}',
        f'specifying TBN cpt using python code')
    # let's find out whether this is a functional or constrained cpt
    parameters = signature(cpt).parameters
    free_parameter_count = 0
    has_star_args        = False
    for p in parameters.values():
        sp = str(p)
        assert '**' not in sp
        if '*' in sp: 
            assert free_parameter_count == 0 # *args must be first
            has_star_args = True
        elif '=' not in sp: free_parameter_count += 1
    parent_count = len(node.parents)
    if (not has_star_args and free_parameter_count == parent_count) or \
       (has_star_args and free_parameter_count==0):
        return 'function'
    assert (not has_star_args and free_parameter_count == parent_count + 1) or \
           (has_star_args and free_parameter_count == 1)
    return 'constraint'