import numpy as np
from copy import copy,deepcopy
from itertools import count
from collections.abc import Sequence
from tbn.node import Node

import tbn.cpt
import utils.utils as u


"""
TBN nodes. 
A node must be constructed after its parents have been constructed.
"""
class NodeV2(Node):
    
    # user attributes are ones that can be specified by the user when constructing node
    user_attributes = ('name','values','parents','testing','fixed_cpt','fixed_zeros',
                        'cpt_tie','functional','cpt','cpts','thresholds','num_intervals')
            
    # only node name is position, everything else is keyword only (* as third argument)
    def __init__(self, name, *, values=(True,False), parents=[], functional=None, 
                    fixed_cpt=False, fixed_zeros=False, testing=False, cpt_tie=None,
                    cpt=None, cpts=None, thresholds=None,num_intervals=None):

        # copy potentially mutable arguments in case they get changed by the user
        values, parents, cpt, cpts, thresholds = \
            copy(values), copy(parents), copy(cpt), deepcopy(cpts), deepcopy(thresholds)
        # other arguments are immutable so no need to copy them

        super().__init__(name,values=values,parents=parents,functional=functional,
            fixed_cpt=fixed_cpt, fixed_zeros=fixed_zeros, testing=testing, cpt_tie=cpt_tie,
            cpt=cpt,cpt1=None,cpt2=None,threshold=None)
        
        u.input_check(testing == False or (num_intervals is not None), 
            f'num of intervals must be specified if testing')
        u.input_check(testing != False or (cpts is None),
            f'node cannot have cpts if it is not testing')
        u.input_check(cpt is None or (cpts is None),
            f'node cannot have both cpt and cpts')
            
        # infer testing flag if needed (flag is optional)
        if testing is None: 
            testing = cpts is not None
            
        # use random cpts if not specified (used usually for testing)
        assert testing in (True,False)
        card  = len(values)
        cards = tuple(p.card for p in parents)

        # shortcut for specifying equal cpts
        if testing and cpt is not None: 
            assert cpts is None
            cpts = [cpt for _ in range(num_intervals)]

        if testing and cpts is None:
            cpts = [tbn.cpt.random(card,cards) for _ in range(num_intervals)]
        if testing and thresholds is None:
            thresholds = [tbn.cpt.random2(cards) for _ in range(num_intervals-1)]

        if not testing and cpt is None:
            cpt  = tbn.cpt.random(card,cards)
            
        # populate node attributes
        self._cpt         = cpt           # becomes np array
        self._cpts        = cpts          # becomes list of np array
        self._thresholds  = thresholds     # becomes np array
        self._num_intervals = num_intervals

        # unused node attributes from super
        self._cpt1 = None
        self._cpt2 = None
        self._threshold = None

    # override read only attributes (exposed to user)
    @property
    def cpt1(self):        return AttributeError("node2 does not have cpt1")
    @property
    def cpt2(self):        return AttributeError("node2 does not have cpt2")
    @property
    def threshold(self):   return AttributeError("node2 does not have threshold")
    @property
    def cpts(self):        return self._cpts
    @property
    def thresholds(self):  return self._thresholds
    @property
    def n_intervals(self): return self._num_intervals
    
    """ public functions override"""

    def is_node_v2(self):
        return True
        
    def tabular_cpt1(self):
        return AttributeError("node2 does not have cpt1")
        
    def tabular_cpt2(self):
        return AttributeError("node2 does not have cpt1")

    # -copies node and processes it so it is ready for inference
    # -this includes pruning node values and expanding/pruning cpts
    def copy_for_inference(self,tbn):
        assert not self._for_inference
        kwargs = {}
        dict   = self.__dict__
        for attr in NodeV2.user_attributes:
            _attr = f'_{attr}'
            assert _attr in dict
            value = dict[_attr]
            if attr=='parents': 
                value = [tbn.node(n.name) for n in value]
            kwargs[attr] = value 
        # node has the same user attribues as self except that parents of self
        # are replaced by corresponding nodes in tbn
        node = NodeV2(**kwargs)  
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
#            s = '.'.join([str(hash(n.values)) for n in self.family])
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

            self._cpts = [np.transpose(cpt,sorted_axes) for cpt in self._cpts]
            self._thresholds = [np.transpose(thres, sorted_axes[:-1]) for thres in self._thresholds]
            #print("Sort node {}".format(self.name))
            #print("cpt size {}".format(self._cpt1.shape))
            #print("threshold size {}".format(self.threshold.shape))
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
            for i,cpt in enumerate(self._cpts):
                set_label(cpt,"cpt{}".format(i+1))
            for i,thres in enumerate(self._thresholds):
                set_label(thres, "thres{}".format(i+1))
        else:
            set_label(self.cpt,'cpt')