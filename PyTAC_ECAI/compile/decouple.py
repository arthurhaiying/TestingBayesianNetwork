import copy

from tbn.tbn import TBN
from tbn.node import Node
import utils.utils as u

"""
Transforms a tbn into another with the goal of reducing treewidth.

  ### First transformation.
  
  For each node X with "hard evidence," we cut its outgoing edges and replace X with 
  root-node replicas in the cpts of its children. After computing an elimination order
  for the transformed network, the replicas are removed and edges are restored.
  
  Impact on compilation:
    # The elimination order is computed based on the transformed network
    # X is removed from separators and clusters (except leaf clusters) in the jointree
    
  This technique can lead to much smaller TACs, depending on the network topology
  and location of hard evidence nodes in the network.
  
  Implementation details.

  Clamped nodes are handled specially in a few places during compilation:
  
    # The ._clamped attribute of hard evidence nodes is set to true.
    # When entering evidence into a cpt, evidence on clamped parents is multiplied 
      by the cpt as well (see __add_evd_to_cpt() in inference.py). 
    # This amounts to conditioning the cpt on clamped parents as clamped nodes are 
      summed out from the cpt cluster when the cluster sends messages to its neighbor 
      (see __set_cls_and_sep() in jointree.py).
    # In a classical jointree, a leaf cpt cluster will have the cpt parents in the
      attached separator. This is no longer the case if the parents are clamped. 
      This require special treatment when computing a posterior over parents during
      the selection of cpts (see __get_parents_posterior() in inference.py).
      

  ### Second transformation.
  
  Some terminology first.
  
  A "fixed" CPT is one that is carried into the TAC during compilation (not learned).
  A "deterministic" CPT has only zeros and ones.
  A "functional" CPT is both fixed and deterministic.
  
  For each node X with a functional CPT and children C1...Cn, n >= 2:
    # replace X with clones X1...Xn (each Xi has the same values, parents and CPT as X)
    # replace each edge X --> Ci with edge Xi --> Ci for i = 1...n
  Note: Clones have a single child each. Clone will be pruned if this child is pruned.

  This transformation duplicates the functional cpt of X: one copy for each child of X.
  
  After a jointree is constructed, clones are replaced by originals in leaf
  jointree nodes.
  
  The transformation is sound: original and transformed network have the same 
  distribution over common nodes (functional cpts can be repeated in a set of factors 
  without changing the factors product).

  Impact on compilation:
    # The elimination order is computed based on the transformed network
    # Nodes with functional cpts are removed from some separators and clusters
  
  This technique can lead to much smaller TACs, depending on the network topology
  and location of functional nodes in the network.
  
  Implementation details.
  
  Functional cpts are handled specially in a few places during compilation:
    # Clones are replaced with originals after jointree is constructed, while
      keeping track of the single child associated with a clone. This leads to
      duplicating functional cpts in the jointree.
      (see __recover_original_vars() in jointree.py) 
    # A duplicated var/cpt is pruned if the associated child is pruned.
      (see __prune() in jointree.py)
    # Functional nodes are removed from some clusters and separators.
      (see __set_cls_and_sep() in view.py)
"""

# returns transformed network and its elimination order
def get(net1,hard_evd_nodes,trainable_tbn,elm_method,elm_wait):
    assert net1._for_inference
    
    #net1.dot(fname='tbn_pre_decouple.gv', view=True)
    u.show(f'  Decoupling tbn:')
    elm_order, cliques1, max_binary_rank1, stats = net1.elm_order(elm_method,elm_wait)
    u.show('   ',stats)

    # cutting edges outgoing form hard evidence nodes
    cut_edges = lambda n: len(n.children) >= 1 and n in hard_evd_nodes and \
                              (n.parents or len(n.children) >= 2)
    cut_edges_set = set(n for n in net1.nodes if cut_edges(n))
    
    # replicating functional cpts
    # if both duplicate and cut_edges trigger, use cut_edges as it is more effective
    duplicate = lambda n: n not in cut_edges_set and len(n.children) >= 2 \
                            and n.is_functional(trainable_tbn)
                          
    duplicate_set = set(n for n in net1.nodes if duplicate(n))
    
    # perhaps decoupling does nothing
    if not duplicate_set and not cut_edges_set: 
        u.show('    nothing to decouple')
        return net1, elm_order, (max_binary_rank1,max_binary_rank1) # no decoupling possible

    # we will decouple
    net2 = TBN(f'{net1.name}__decoupled')
    net2._decoupling_of = net1
    
    # -when creating a clone c(n) in net2 for node n in net1, we need to look up the
    #  parents of c(n) in net2. 
    # -this is done by calling get_image(p) on each parent p of node n
    # -the length of images[p] equals the number of times get_image(p) will be called 
    # -members of images[p] may not be distinct depending on the replication strategey
    images = {} 
    def get_image(n):
        return images[n].pop()
    
    # -when we have hard evidence on node n (net1), we create a replica r (net2) of n
    # for each child of n, which copies evidence on n into the cpts of its children.
    # -maps node r (net2) to node n (net1) that it is copying evidence from
    evidence_from = {} 
    
    # maps node n (net1) to a tuple (c_1,...,c_k) where k is the number of clones that
    # node n will have in nets2, and c_i is the number of children for clone i in net2
    ccounts = {} 
    
    # fully replicated(i): one i-replica for each c-replica, where c is child of i in net1
    # partial replicated(i): one i-replica for each child c of i in net1
    fully_replicated = lambda i: all(ccount==1 for ccount in ccounts[i])
    replicas_count   = lambda i: len(ccounts[i]) # number of replicas node i has in net2
    
    # compute the number of replicas in net2 for each node in net1 (fill ccounts)
    for n in reversed(net1.nodes): # bottom up
        ccounts[n] = []
        cparents   = set()
        for c in n.children: 
            cparents |= set(c.parents)
        #replicate_node = any(cparents <= clique for clique in cliques1)
        #replicate_node = all(p in duplicate_set for p in n.parents)
        replicate_node  = True
        if n in duplicate_set and replicate_node:
            # replicate node n
            for c in n.children:
                if True: #not fully_replicated(c): 
                    # replicate node n for each replica of child c
                    ccounts[n].extend([1]*replicas_count(c))
                else: 
                    # replicate node n for each child c
                    ccounts[n].append(replicas_count(c))
        else: # do not replicate node n
            # n could be in cut_edges_set, but ccounts will not be used in that case
            duplicate_set.discard(n)
            children_replicas_count = sum(replicas_count(c) for c in n.children)
            ccounts[n].append(children_replicas_count)
              
              
    # cutting edges takes priority over decoupling as it is more effective
    for n in net1.nodes: # visiting parents before children
        if n in cut_edges_set: # disconnect n from its children
            assert n not in duplicate_set
            n._clamped = True  # flag set in original network (net1)
            parents    = [get_image(p) for p in n.parents]
            master     = clone_node(n,n.name,parents)
            net2.add(master)
            images[n] = []
            # master not added to images as it will not be a parent of any node in net2
            for i, c in enumerate(n.children):
                for j in range(replicas_count(c)): # j iterates over replicas of child c
                    # these clones will be removed after elimination order is computed
                    # clones are not testing even if master is testing
                    clone = Node(f'{n.name}_evd{i}_{j}',values=master.values,parents=[])
                    net2.add(clone)
                    evidence_from[clone] = master
                    images[n].append(clone) # children of n will reference clones, not master
        elif n in duplicate_set:        # duplicate node n and its functional cpt
            images[n] = []
            for i, ccount in enumerate(ccounts[n]):
                assert ccount > 0 # number of children each clone will have in net2
                parents = [get_image(p) for p in n.parents] 
                clone   = clone_node(n,f'{n.name}_fcpt{i}',parents)
                if i > 0: clone._master = False # clone() sets this to True
                net2.add(clone)
                images[n].extend([clone]*ccount)
        else: # just copy node n from net1 to net2
            (ccount,) = ccounts[n] # number of children clone will have in net2
            parents   = [get_image(p) for p in n.parents]
            clone     = clone_node(n,n.name,parents)
            net2.add(clone)
            images[n] = [clone]*ccount           

    assert not net2._for_inference
    assert len(images) == len(net1.nodes) 
    assert len(images) <= len(net2.nodes)
    assert all(v==[] for v in images.values())

    #net2.dot(fname='tbn_post_decouple.gv', view=True)
    elm_order, _, max_binary_rank2, stats = net2.elm_order(elm_method,elm_wait)
    u.show('   ',stats)
                
    if not duplicate_set: 
        elm_order  = [n._original for n in elm_order  if n not in evidence_from]
        # only clamping took place, so we only care about elimination order
        # return original network with _clamped flag set for some nodes
        return net1, elm_order, (max_binary_rank1,max_binary_rank2)
    
    if cut_edges_set: # some variables were clamped
        # get rid of auxiliary evidence nodes from elimination order
        elm_order  = [n for n in elm_order  if n not in evidence_from]
        # need to restore net2 by getting rid of auxiliary evidence nodes
        # and restoring children of clamped nodes
        net2.nodes = [n for n in net2.nodes if n not in evidence_from]
        replace = lambda n: evidence_from[n] if n in evidence_from else n
        for n in net2.nodes:
            n._parents = tuple(replace(p) for p in n.parents)
            n._family  = tuple(replace(f) for f in n.family)
    
    u.show(f'    node growth: {len(net2.nodes)/len(net1.nodes):.1f}')
            
    return net2, elm_order, (max_binary_rank1,max_binary_rank2)
        
# duplicates node except for id, name and neighbors
def clone_node(original,name,parents):
    assert original._for_inference
    node           = copy.copy(original) # shallow copy
    node._original = original # points to node in net1 that is being cloned in net2
    node._master   = True     # master nodes of net2 form a network that mirrors net1
    node._name     = name
    node._id       = next(Node.ID)
    node._parents  = tuple(parents)
    node._family   = tuple([*parents,node])
    node._children = []
    node._tbn      = None
    
    # cpts of original have already been processed
    node._for_inference = False
    return node
        