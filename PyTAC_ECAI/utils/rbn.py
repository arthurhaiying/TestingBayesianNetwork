import numpy as np

from tbn.tbn import TBN
from tbn.node import Node

"""
A generator for random bayesian networks.

# vcount: number of variables
# scount: maximum number of values a node can have
# fcount: number of functional variables
# pcount: maximum number of parents per node
# back:   maximum distance between node and its parents in natural var order
"""

def get(vcount,scount,pcount,fcount,back,testing):
    assert vcount >= 2 and scount >= 2
    assert fcount >= 0 and fcount <= vcount and pcount < vcount
    assert back < vcount and pcount <= back
    
    # decide values and parents first
    i2values     = {}
    i2parents    = {}
    parents_pool = []
    
    for i in range(vcount):
        sc      = np.random.randint(2,scount+1)      # number of values
        pc      = np.random.randint(1+min(i,pcount)) # number of parents
        assert sc >= 2 and sc <= scount
        assert pc <= pcount
        
        i2values[i]  = range(sc)
        i2parents[i] = list(np.random.choice(parents_pool,pc,replace=False))
            
        parents_pool.append(i)
        if len(parents_pool) > back:
            parents_pool.pop(0)
            
    # choose functional variables randomly
    candidates = [i for i, parents in i2parents.items() if parents] # exclude roots
    fcount     = min(fcount,len(candidates))
    functional_indices = set(np.random.choice(candidates,fcount,replace=False))
    
    # construct bn
    bn  = TBN(f'random-{vcount}-{scount}-{pcount}-{fcount}-{back}')
    if testing: # construct a tbn equivalent to bn
        bn2     = TBN(f'random-{vcount}-{scount}-{pcount}-{fcount}-{back}-Testing')
        i2node2 = {}
     
    i2node = {} # maps node number to node object
    for i in range(vcount):
        values     = i2values[i]
        parents    = [i2node[k] for k in i2parents[i]]
        functional = i in functional_indices
        assert not functional or parents # roots cannot be functional
        
        cpt_shape  = [p.card for p in parents]
        cpt_shape.append(len(values))
        cpt        = __random_cpt(cpt_shape,functional)
        
        # tbn
        if testing:
            parents2 = [i2node2[k] for k in i2parents[i]]
            if parents2 and not functional and np.random.random() >= .5: # testing
                node = Node(f'v{i}',values=values,parents=parents2,cpt1=cpt,cpt2=cpt)
            else: # regular
                node = Node(f'v{i}',values=values,parents=parents2,cpt=cpt)
            bn2.add(node)
            i2node2[i] = node    
        # bn
        node = Node(f'v{i}',values=values,parents=parents,cpt=cpt)     
        bn.add(node)
        i2node[i] = node

    roots  = [node.name for node in bn.nodes if not node.parents]
    leaves = [node.name for node in bn.nodes if not node.children]
        
    if testing:
        return bn, roots, leaves, bn2
    return bn, roots, leaves
    
        
# constructs random cpt      
def __random_cpt(shape,functional,index=0):
    if index == len(shape)-1:
        return __random_distribution(shape[index],functional)
    else:
        return [__random_cpt(shape,functional,index+1) for _ in range(shape[index])]
        
# constructs random distribution
def __random_distribution(card,functional):
    if functional: # functional distribution
        distribution = [0.]*card
        index        = np.random.randint(card)
        distribution[index] = 1.
    else:
        distribution  = np.random.rand(card)
        norm_constant = sum(distribution)
        distribution  = [n/norm_constant for n in distribution] 
    return distribution