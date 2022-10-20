from functools import lru_cache
from collections import deque

from utils.mset import mset
import utils.utils as u
        
"""
Setting the clusters and separators of a jointree view (exploits functional cpts)

The only exposed function is set_separators_and_clusters().

This function sets the view's separators and clusters and is based on replicating
functional cpts in the view, which creates opportunities for dropping functional
variables from separators (and clusters), therefore reducing the view's rank.

The method is based on two heuristics for:

1. replicating functional cpts in the view.
2. dropping functional variables from separators.
"""

# trainable: whether the tac is trainable (needed to determine functional cpts)
def set_separators_and_clusters(view,trainable,verbose):
   
    has_replicas = lambda: any(len(nodes) > 1 for nodes in view.fcpts.values())
    
    # identify functional cpts (fcpts) and associate them with functional vars
    __set_fcpts(view,trainable)
    
    # replicate fcpts if not already replicated by decouple.py
    if view.fcpts and not has_replicas(): 
        __replicate_fcpts(view,verbose)
    
    # compute separators
    __set_classical_cls_and_sep(view) 

    # remove functional vars from separators, then remove dead fcpts
    if has_replicas():
        __shrink_separators(view)
        __remove_dead_fcpts(view,verbose)
          
    # remove clamped vars from separators (if any)
    for i,p,_,_,_ in view.bottom_up(): 
        sep  = view.sep(i)
        sep -= set(var for var in sep if var._clamped)
    
    # compute clusters (cluster of host computed earlier in view.py)
    for i,_,c1,c2,_ in view.bottom_up():
        if not c1: view.cls_set(i,set(i.var.family)) # leaf node i
        else:      view.cls_set(i,view.sep(c1) | view.sep(c2))
        
    if verbose:
        u.show('  View ranks : ' + view.ranks_str())


""" identify vars and functional cpts """ 

# functional var has a functional cpt
def __set_fcpts(view,trainable):
    for i in view.nodes:
        if view.leaf(i) and i.var.is_functional(trainable):
            var = i.var
            if var not in view.fcpts: # first fcpt encountered for var
                view.fcpts[var] = [i]
            else: # var has replicated fcpts
                assert not var._clamped # we don't replicate fcpts of clamped vars
                view.fcpts[var].append(i)

""" compute classical separators (before dropping functional vars) """

def __set_classical_cls_and_sep(view):

    vars = {} # maps view node i to vars at/below node i
    
    # compute vars
    for i,_,c1,c2,_ in view.bottom_up():
        if not c1: vars[i] = set(i.var.family) # leaf i
        else:      vars[i] = vars[c1] | vars[c2]
            
    # compute separators
    for i,p,_,_,s in view.top_down():
        if not s: sep = vars[i] & set(p.var.family) # root i, host p
        else:     sep = vars[i] & (vars[s] | view.sep(p))
        view.sep_set(i,sep)

    # compute clusters
    # cluster of host set earlier in view.py
    for i,_,c1,c2,_ in view.bottom_up():
        if not c1: view.cls_set(i,set(i.var.family))
        else:      view.cls_set(i,view.sep(c1) | view.sep(c2))    
                   
               
""" HEURISTIC: drop functional variables from separators """

# used in ECAI-2020 paper
def __shrink_separators(view):

    # compute vars that have fcpt at/below each view node
    fvars = {} # maps view node i to vars with at/below node i
    for i, _, c1, c2, _ in view.bottom_up():
        if not c1: fvars[i] = set([i.var]) if view.has_fcpt(i) else set()
        else:      fvars[i] = fvars[c1] | fvars[c2]
        
    # heuristic for deciding whether to drop a functional variable from
    # a node or from its sibling when both contain fcpt for the variable
    size = {n:0 for n in view.nodes} # maps node to size of separators at/below node
    for i, _, c1, c2, _ in view.bottom_up():
        if not c1: continue # leaf i
        sep_size = view.sep_size(c1) + view.sep_size(c2)
        size[i]  = sep_size + size[c1] + size[c2]

    # drop functional vars from separators in a top-down pass        
    r, h    = view.root, view.host
    h_fvars = set([h.var]) if view.has_fcpt(h) else set()
    view.sep_minus(r,fvars[r] & h_fvars) 
    
    # MUST process siblings (c1,c2) simultaneously before processing their children
    # this is important to enforce the running intersection property
    for i,_,c1,c2,_ in view.top_down():
        if not c1: continue # leaf node i
        sum = fvars[c1] & fvars[c2]
        if sum: # size heuristic
            if size[c1] > size[c2] or (size[c1] == size[c2] and c1.id < c2.id):
                view.sep_minus(c1,sum)
            else: 
                view.sep_minus(c2,sum)
        view.sep_intersect(c1,view.sep(c2)|view.sep(i))
        view.sep_intersect(c2,view.sep(c1)|view.sep(i))
        
def __shrink_separators_new(view):
    
    # amount of reduction in separator/cluster sizes if we drop fvar from sep(i)
    # this is not efficient as we propagate effect of dropping fvar from sep(i)
    def reduction(i,fvar):
        assert fvar in view.sep(i)
        fraction = (fvar.card-1)/fvar.card # fraction of sep size that will be reduced
        sep = view.sep_size(i)*fraction
        cls = 0
        while not view.leaf(i):
            c1, c2 = view.children(i)
            if fvar not in view.sep(c1):
                i, p = c2, i
            elif fvar not in view.sep(c2):
                i, p = c1, i
            else:
                break
            assert fvar in view.sep(i) and fvar in view.cls(p)
            sep += view.sep_size(i)*fraction
            cls += view.cls_size(p)*fraction
        return (cls,sep) # time vs space
        
    leaf_fvars = lambda i: set([i.var]) if view.has_fcpt(i) else set()
    
    # compute vars that have fcpt at/below each view node
    fvars = {} # maps view node i to vars with at/below node i
    for i,_,c1,c2,_ in view.bottom_up():
        if not c1: fvars[i] = leaf_fvars(i) # leaf node i
        else:      fvars[i] = fvars[c1] | fvars[c2]

    # drop functional vars from separators in a top-down pass        
    r, h = view.root, view.host
    view.sep_minus(r,fvars[r] & leaf_fvars(h))  
    
    # MUST process siblings (c1,c2) simultaneously before processing their children
    # this is important to enforce the running intersection property
    for i,_,c1,c2,_ in view.top_down():
        if not c1: continue # leaf node i
        for fvar in fvars[c1] & fvars[c2]:
#            print(f'\n{fvar.name}')
#            print(c1.id,reduction(c1,fvar))
#            print(c2.id,reduction(c2,fvar))
            if reduction(c1,fvar) > reduction(c2,fvar): # expensive test
                view.sep_minus(c1,set([fvar]))
            else:
                view.sep_minus(c2,set([fvar]))
        view.sep_intersect(c1,view.sep(c2)|view.sep(i))
        view.sep_intersect(c2,view.sep(c1)|view.sep(i))
        view.cls_set(i,view.sep(c1)|view.sep(c2))


""" HEURISTIC: replicate functional cpts in view """

def __replicate_fcpts(view,verbose):

    __set_classical_cls_and_sep(view)
    
    if verbose:
        replica_count  = sum(len(leaves)-1 for leaves in view.fcpts.values())
        distinct_count = sum(1 for leaves in view.fcpts.values() if len(leaves) >= 2)
        u.show(f'  added fcpts: {replica_count}, distinct {distinct_count}')
        u.show('  View ranks : ' + view.ranks_str())
       
    # compute vars that have fcpt at/below each view node
    vars   = {} # maps view node i to vars at/below node i
    ovars  = {} # maps view node i to vars outside node i
    fvars  = {} # maps view node i to vars with functional cpt at/below node i
    ofvars = {} # maps view node i to vars with functional cpt outside node i
    def set_vars():
        nonlocal vars, ovars, fvars, ofvars
        vars, ovars, fvars, ofvars = {}, {}, {}, {}
        for i,_,c1,c2,_ in view.bottom_up():
            if not c1: # leaf node i
                fvars[i] = set([i.var]) if view.has_fcpt(i) else set()
                vars[i]  = set(i.var.family)
            else:       
                fvars[i] = fvars[c1] | fvars[c2]
                vars[i]  =  vars[c1] |  vars[c2]
        for i,p,_,_,s in view.top_down():
            if not s: # root node i
                ofvars[i] = set([p.var]) if view.has_fcpt(p) else set()
                ovars[i]  = set(p.var.family)
            else:      
                ofvars[i] = fvars[s] | ofvars[p]
                ovars[i]  =  vars[s] |  ovars[p]
    set_vars()
    
    fvar_order = list(view.fcpts)
    fvar_order.sort(key = lambda var: len(var.parents))
    
    additions = []
    for i,p,c1,c2,s in view.bottom_up():
        if not s: continue
        if c1:
            fvars[i]  = fvars[c1] | fvars[c2]
            vars[i]   =  vars[c1] |  vars[c2]
        ovars[i]  =  vars[s] |  ovars[p]
        ofvars[i] = fvars[s] | ofvars[p]
#        sepi = view.sep(i)
        sepi  = vars[i] & ovars[i] # outside loop
        for fvar in fvar_order: # all functional variables
            parents = set(fvar.parents)
            cond1 = len(parents-sepi) <= 1
            cond2 = parents <= fvars[i]
            cond3 = fvar in fvars[s] and fvar not in fvars[i]
            if (cond1 or cond2) and cond3:
                additions.append((i,fvar))
                fvars[i].add(fvar)
                vars[i] |= set(fvar.family)
                 
    # replicate fcpts in view
    for i, fvar in additions:
        leaf = __add_leaf(view,fvar,i)
        view.fcpts[fvar].append(leaf)     
    __reconstruct(view)
    __set_classical_cls_and_sep(view)
    
    if verbose: 
        replica_count  = sum(len(leaves)-1 for leaves in view.fcpts.values())
        distinct_count = sum(1 for leaves in view.fcpts.values() if len(leaves) >= 2)
        u.show(f'  added fcpts: {replica_count}, distinct {distinct_count}')
        u.show('  View ranks : ' + view.ranks_str())
        
        
""" remove dead functional cpts from view """
        
# assumes separators are up-to-date (have been shrunk due to functional cpts)
def __remove_dead_fcpts(view,verbose):  
    
    pre_replica_count  = sum(len(leaves)-1 for leaves in view.fcpts.values())
    
    #view.dot('pre.gv')
    #u.pause()
    
    # dead basically means: not contributing to shrinking separators
    # leaf node i is dead if it has a functional cpt whose var is summed out
    # host cannot be dead (contains query var which is never summed out)
    # root cannot be dead (would have been pruned if it was dead)  
    def dead(i):
        return i != view.host and view.has_fcpt(i) and i.var not in view.sep(i)
             
    # add dead fcpts to dropped and clear their separators
    dropped = set() 
    def drop_dead_fcpts():
        key = lambda i: (dead(i),not i.is_host,len(i.var.parents))
        for fvar, leaves in view.fcpts.items():
            leaves.sort(key=key) # we prefer to keep hosts
            new_leaves = []
            for index, i in enumerate(leaves):
                if index != 0 and dead(i): # we need to keep one fcpt per fvar
                    view.sep_set(i,set())  # clear separator
                    dropped.add(i)
                else:
                    new_leaves.append(i)
            view.fcpts[fvar] = new_leaves
    
    # shrink separators further due to removing dead fcpts
    def shrink_separators():
        for i,_,c1,c2,_ in view.bottom_up():
            if c1: 
                assert i not in dropped
                view.sep_intersect(i,view.sep(c1)|view.sep(c2))
        for i,p,_,_,s in view.top_down():
            if s and i not in dropped: 
                view.sep_intersect(i,view.sep(s)|view.sep(p))
                view.cls_set(p,view.sep(i) | view.sep(p))
    
    # whether view still have dead fcpts
    def more_dead():
        for leaves in view.fcpts.values():
            count = sum(dead(i) for i in leaves)
            if count > 1 or (count==1 and len(leaves) > 1):
                return True
        return False
    
    # identify dead fcpts
    while True:
        drop_dead_fcpts()
        shrink_separators()
        if not more_dead(): break
    assert all(view.fcpts.values()) # at least one fcpt for each fvar
                   
    # remove dead fcpts from view
    for i in dropped: 
        __remove_leaf(view,i)
    __reconstruct(view)
            
    if verbose:
        replica_count  = sum(len(leaves)-1 for leaves in view.fcpts.values())
        distinct_count = sum(1 for leaves in view.fcpts.values() if len(leaves) >= 2)
        u.show(f'   kept fcpts: {replica_count}/{pre_replica_count}, distinct {distinct_count}')
        
    #view.dot('post.gv')
    #u.pause()

""" add/remove leaf nodes in view """

# inserts a leaf node l for var between node i and its parent p: adds internal 
# node k too where k is the new child of p and (i,l) are its children
def __add_leaf(view,var,i):
    assert i != view.host
    k = view.Ghost()        # internal node with parent p and children i,l
    l = view.Ghost(var=var) # leaf node with parent k and no children
    p = view.parent(i)
    if i == view.root:
        assert p == view.host # children node defined
        view.root = k
    else:        
        s = view.sibling(i)
        view.children_[p] = [s,k]
    view.parent_[k] = p
    view.parent_[i] = k
    view.parent_[l] = k
    view.children_[k] = [i,l]
    view.children_[l] = []
    return l
    
# removes leaf node i from view: removes its parent p too and makes 
# its sibling s a child of its grandparent g
def __remove_leaf(view,i):
    assert view.leaf(i) and i != view.root and i != view.host
    p, s = view.parent(i), view.sibling(i)
    g    = view.parent(p)
    if p != view.root:
        ss = view.sibling(p)
        view.parent_[s] = g
        view.children_[g] = [ss,s]
    else: # p is root so has no sibling
        ss = None
        view.parent_[s] = g
        assert g == view.host # children of host are not defined
        view.root = s         # s is now the new root
        
# reconstructs topological order of view nodes after having added or removed
# nodes from the view: breadth-first traversal starting from the view root
def __reconstruct(view):
    view.nodes = deque([])
    queue      = deque([view.root])
    while queue: # breadth first
        i = queue.popleft()
        queue.extend(view.children(i))
        view.nodes.appendleft(i)
        p = view.parent(i)
        view.depth_[i] = 1+view.depth_[p]
    view.nodes_set = set(view.nodes)