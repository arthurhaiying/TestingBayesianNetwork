import numpy as np
from copy import copy
from itertools import count
from collections.abc import Sequence
    
import tbn.cpt
import utils.utils as u

"""
TBN nodes. 
A node must be constructed after its parents have been constructed.
"""
class Node:
    
    ID        = count() # for generating node IDs
    cpt_types = ('cpt','cpt1','cpt2')
    
    # user attributes are ones that can be specified by the user when constructing node
    user_attributes = ('name','values','parents','testing','fixed_cpt','fixed_zeros',
                        'cpt_tie','functional','cpt','cpt1','cpt2')
            
    # only node name is position, everything else is keyword only (* as third argument)
    def __init__(self, name, *, values=(True,False), parents=[], functional=None, 
                    fixed_cpt=False, fixed_zeros=False, testing=None, cpt_tie=None,
                    cpt=None, cpt1=None, cpt2=None):

        # copy potentially mutable arguments in case they get changed by the user
        values, parents, cpt, cpt1, cpt2 = \
            copy(values), copy(parents), copy(cpt), copy(cpt1), copy(cpt2)
        # other arguments are immutable so no need to copy them
        
        # check integrity of arguments
        u.input_check(type(name) is str and str is not '',
            f'node name must be a nonempty string')
        u.input_check(isinstance(values,Sequence),
            f'node values must be a python sequence')
        u.input_check(len(values) >= 1,
            f'node must have at least one value')
        u.input_check(len(values) == len(set(values)),
            f'node values must be unique')
        u.input_check(type(parents) is list,
            f'node parents must be a list')
        u.input_check(len(parents) == len(set(parents)),
            f'node parents must be unique')
        u.input_check(all(type(p) is Node for p in parents),
            f'node parents must be TBN nodes')
        u.input_check(functional in (True,False,None),
            f'functional flag must be True or False')
        u.input_check(fixed_cpt in (True,False),
            f'fixed_cpt flag must be True or False')
        u.input_check(fixed_zeros in (True,False),
            f'fixed_zeros flag must be True or False')
        u.input_check(testing in (True,False,None),
            f'testing flag must be True or False')
        u.input_check(testing != False or (cpt1 is None and cpt2 is None),
            f'node cannot have cpt1/cpt2 if it is not testing')
        u.input_check(cpt_tie is None or (type(cpt_tie) is str and str is not ''),
            f'node flag cpt_tie must be a non-empty string')
        u.input_check(not (fixed_cpt and fixed_zeros),
            f'node flags fixed_cpt and fixed_zeros cannot be both True')
        u.input_check(not (fixed_cpt and cpt_tie),
            f'node cpt cannot be tied if it is also fixed')
        u.input_check(not (fixed_zeros and cpt_tie),
            f'node cpt cannot be tied if it has fixed zeros')
        u.input_check(cpt is None or (cpt1 is None and cpt2 is None),
            f'node cannot have both cpt and cpt1/cpt2')
        u.input_check((cpt1 is None) == (cpt2 is None),
            f'node cpt1 and cpt2 must both be specified if the node is testing')
        
        # shortcut for specifying equal cpt1/cpt2
        if testing and cpt is not None: 
            assert cpt1 is None and cpt2 is None
            cpt1 = cpt2 = cpt
            cpt  = None
            
        # infer testing flag if needed (flag is optional)
        if testing is None: 
            testing = cpt1 is not None and cpt2 is not None
        
        u.input_check(not testing or parents,
            f'testing node must have parents')
            
        # use random cpts if not specified (used usually for testing)
        assert testing in (True,False)
        card  = len(values)
        cards = tuple(p.card for p in parents)
        
        if testing and cpt1 is None:
            cpt1 = tbn.cpt.random(card,cards)
            cpt2 = tbn.cpt.random(card,cards)
        if not testing and cpt is None:
            cpt  = tbn.cpt.random(card,cards)
            
        # populate node attributes
        self._id          = next(Node.ID) # need not be unique (clones have same id)
        self._name        = name          # a unique string identifier of node
        self._testing     = testing       # whether node is testing
        self._fixed_cpt   = fixed_cpt     # cpt cannot be trained
        self._fixed_zeros = fixed_zeros   # zero probabilities in cpt will not be trained
        self._functional  = functional    # whether node is functional
        
        # -the following attributes may change when preparing network for inference
        # -node values may be pruned, network edges may be pruned and cpts may 
        #  may be expanded to tabular form and/or pruned due to edge/value pruning
        self._values      = values        # becomes a tuple if values are pruned
        self._parents     = parents       # becomes a tuple (must match cpt order)
        self._cpt         = cpt           # becomes np array
        self._cpt1        = cpt1          # becomes np array
        self._cpt2        = cpt2          # becomes np array
        self._cpt_tie     = cpt_tie       # tied cpts may have different shapes after pruning
        
        # derived attributes that may also change when preparing for inference
        family            = [*parents,self]
        self._card        = card
        self._family      = family        # becomes a tuple (must match cpt order)
        self._children    = []            # updated when children added to network
        
        # further attributes that are set later
        self._for_inference = False       # set when preparing for inference
        self._tbn         = None          # set when node added to a tbn
    
    # for sorting
    def __lt__(self,other):
        return self.id < other.id
        
    # for printing
    def __str__(self):
        parents = u.unpack(self.parents,'name')
        return "%s. Node %s %s: children %s, parents %s" % \
            (self.id, self.name, self.values,len(self.children),parents)            
            
    # read only attributes (exposed to user)
    @property
    def id(self):          return self._id
    @property
    def name(self):        return self._name
    @property
    def values(self):      return self._values
    @property
    def card(self):        return self._card
    @property
    def testing(self):     return self._testing
    @property
    def fixed_cpt(self):   return self._fixed_cpt
    @property
    def fixed_zeros(self): return self._fixed_zeros
    @property
    def parents(self):     return self._parents
    @property
    def children(self):    return self._children
    @property
    def family(self):      return self._family
    @property
    def cpt_tie(self):     return self._cpt_tie
    @property
    def cpt(self):         return self._cpt
    @property
    def cpt1(self):        return self._cpt1
    @property
    def cpt2(self):        return self._cpt2
    @property
    def cpt_label(self):   return self._cpt_label
    @property
    def tbn(self):         return self._tbn
    
    
    """ public functions """
    
    # the shape of cpt for self
    def shape(self):
        return tuple(n.card for n in self.family)
        
    # whether self is leaf
    def leaf(self):
        return not self.children
        
    # whether self is tied to node (have the same cpt)
    def cpt_tied_to(self,node):
        return node.cpt_tie is not None and node.cpt_tie == self.cpt_tie
        
    # set of tbn nodes with the same cpt tie as self
    # None if self is not tied
    def tied_nodes(self):
        return self.tbn.tied_nodes(self.cpt_tie)
   
    # whether node lost values due to pruning
    def has_pruned_values(self):
        assert self._for_inference # cpts must have been processed
        return self.card < self._card_org
        
    # returns nodes connected to self (as a set)
    def connected_nodes(self):
        visited = set()
        def visit(n):
            if n in visited: return
            visited.add(n)
            for p in n.parents:  visit(p)
            for c in n.children: visit(c)
        visit(self)     
        return visited
    
    # -whether node has a functional cpt
    # -non-trainable tbns: this is decided by examining the node's cpt
    # -trainable tbns: this is decided by examining the cpt if the cpt is fixed,
    # -otherwise it has to be declared by the user using the 'functional' flag
    def is_functional(self,trainable):
        assert self._for_inference # cpts must have been processed
        cond1 = not trainable and self._all01_cpt
        cond2 = trainable and (self._functional or (self.fixed_cpt and self._all01_cpt))
        return cond1 or cond2
    
    # -returns the cpt for node as a numpy array
    # -does not prune values or edges (which is done when preparing cpts for inference)
    # -cpts are expanded into np arrays on demand
    def tabular_cpt(self):
        assert not self.testing and not self._for_inference
        cpt,_ = tbn.cpt.expand(self,self._cpt,'cpt')
        return cpt
        
    def tabular_cpt1(self):
        assert self.testing and not self._for_inference
        cpt,_ = tbn.cpt.expand(self,self._cpt1,'cpt1')
        return cpt
        
    def tabular_cpt2(self):
        assert self.testing and not self._for_inference
        cpt,_ = tbn.cpt.expand(self,self._cpt2,'cpt2')
        return cpt
    
    """ 
    Preparing node for inference by processing its values and cpts:
    
    -cpts specified using python code are expanded into np arrays
    -cpts specified using nested lists are converted to np arrays
    -missing cpts are filled in randomly (np arrays)
    -if a node value is guaranteed to have a zero probability, it is pruned
    -if a parent ends up having a single value, it is removed from the node
     parents & family and the node cpt is reduced accordingly
    -parents & family are sorted (by id) and the cpts are adjusted accordingly
    """
    
    # -copies node and processes it so it is ready for inference
    # -this includes pruning node values and expanding/pruning cpts
    def copy_for_inference(self,tbn):
        assert not self._for_inference
        kwargs = {}
        dict   = self.__dict__
        for attr in Node.user_attributes:
            _attr = f'_{attr}'
            assert _attr in dict
            value = dict[_attr]
            if attr=='parents': 
                value = [tbn.node(n.name) for n in value]
            kwargs[attr] = value 
        # node has the same user attribues as self except that parents of self
        # are replaced by corresponding nodes in tbn
        node = Node(**kwargs)  
        node.__prepare_for_inference()
        node._for_inference = True
        return node
    
    # -prunes node values and single-value parents
    # -expands cpts into np arrays
    # -identifies 0/1 cpts
    # -sets cpt labels (for saving into file)
    # -sorts parents, family and cpts
    def __prepare_for_inference(self):
    
        # the following attributes are updated in decouple.py, which replicates
        # functional cpts and handles nodes with hard evidence, creating clones
        # of nodes in the process (clones are added to another 'decoupled' network)
        self._original   = None  # tbn node cloned by this one
        self._master     = None  # exactly one clone is declared as master
        self._clamped    = False # whether tbn node has hard evidence
        
        # the following attributes with _cpt, _cpt1, _cpt2 are updated in cpt.y
        self._values_org = self.values # original node values before pruning
        self._card_org   = self.card   # original node cardinality before pruning
        self._values_idx = None        # indices of unpruned values, if pruning happens
        
        # -process node and its cpts
        # -prune node values & parents and expand/prune cpts into tabular form  
        tbn.cpt.set_cpts(self)
                
        # the following attributes will be updated next
        self._all01_cpt  = None  # whether cpt is 0/1 (not applicable for testing nodes)
        self._cpt_label  = None  # for saving to file (updated when processing cpts)
        
        # identify 0/1 cpts
        if self.testing:
            # selected cpt is not necessarily all zero-one even if cpt1 and cpt2 are
            self._all01_cpt = False
        else:
            self._all01_cpt = np.all(np.logical_or(self.cpt==0,self.cpt==1))
            u.check(not (self.fixed_cpt and self._functional) or self._all01_cpt,
                f'node {self.name} is declared functional but its fixed cpt is not functional',
                f'specifying TBN node')
        
        # -pruning node values or parents changes the shape of cpt for node
        # -a set of tied cpts may end up having different shapes due to pruning
        # -we create refined ties between groups that continue to have the same shape
        """ this is not really proper and needs to be updated """
        if self.cpt_tie is not None:
            # s = '.'.join([str(hash(n.values)) for n in self.family])
            self._cpt_tie = f'{self.cpt_tie}__{self.shape()}'
            
        self.__set_cpt_labels()
        
        # we need to sort parents & family and also adjust the cpt accordingly
        # this must be done after processing cpts which may prune parents
        self.__sort()
        assert u.sorted(u.map('id',self.parents))
        assert u.sorted(u.map('id',self.family))
        
    
    # sort family and reshape cpt accordingly (important for ops_graph)
    def __sort(self):
        assert type(self.parents) is list and type(self.family) is list
        
        if u.sorted(u.map('id',self.family)): # already sorted
            self._parents = tuple(self.parents)
            self._family  = tuple(self.family)
            return
        
        self._parents.sort()
        self._parents = tuple(self.parents)
        
        # save original order of nodes in family (needed for transposing cpt)
        original_order = [(n.id,i) for i,n in enumerate(self.family)]
        self.family.sort()
        self._family = tuple(self.family)
        
        # sort cpt to match sorted family
        original_order.sort() # by node id to match sorted family
        sorted_axes = [i for (_,i) in original_order] # new order of axes
        if self.testing:
            self._cpt1 = np.transpose(self.cpt1,sorted_axes)
            self._cpt2 = np.transpose(self.cpt2,sorted_axes)
        else:
            self._cpt  = np.transpose(self.cpt,sorted_axes)


    # sets cpt labels used for saving cpts to file
    def __set_cpt_labels(self):
        
        # maps cpt type to label
        self._cpt_label = {}
        
        def set_label(cpt,cpt_type):
            assert cpt_type not in self.cpt_label
            type_str    = cpt_type + (f' (tie_id {self.cpt_tie})' if self.cpt_tie else '')
            parents_str = u.unpack(self.parents,'name')
            self._cpt_label[cpt_type] = f'{type_str}: {self.name} | {parents_str}'
        
        if self.testing:
            set_label(self.cpt1,'cpt1')
            set_label(self.cpt2,'cpt2')
        else:
            set_label(self.cpt,'cpt')