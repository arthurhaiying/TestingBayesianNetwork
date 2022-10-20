import math
from itertools import count
from graphviz import Digraph

import compile.view as vw
import utils.paths as paths
import utils.utils as u

""" binary jointree node """
class Node:
    
    id = count()
    
    # tbn nodes are called variables (vars)
    def __init__(self,jointree,var=None,children=None):
        # var is a tbn node
        self.id         = next(Node.id)
        self.var        = var       # tbn node whose cpt is hosted at this jt node (leaves only)
        self.is_host    = var!=None # None if node contains a replicated fcpt
        self.is_pruned  = False     # whether node is pruned
        self.jointree   = jointree  # jointree containing this node
            
        # neighbors and hosts
        if children: # internal node: three neighbors [child1,child2,parent]
            c1, c2 = children
            self.neighbors = [c1,c2]  # parent will be added later
            c1.neighbors.append(self) # add self as parent of c1
            c2.neighbors.append(self) # add self as parent of c2
        else: # leaf node: single neighbor [parent]
            self.neighbors = [] # parent will be added later
            assert var not in jointree.hosts
            jointree.hosts[var] = self
    
    def __lt__(self,other):
        return self.id < other.id
        
    def __str__(self):
        return str(self.id) + (f' ({self.var.name})' if self.var else '')
        
    # whether a jointree node is leaf
    # leaf has only one neighbor, internal has three neighbors
    def leaf(self):
        return self.var is not None
        
    # returns single neighbor of a leaf jointree node
    def neighbor(self):
        assert self.leaf() and len(self.neighbors) == 1
        return self.neighbors[0] # single neighbor for leaf nodes
        
    # whether the node contains a replicated cpt
    def is_replica(self):
        assert self.leaf()
        return self.jointree.host[self.var] != self # node does not host its var
    
            
""" binary jointree: each node has one or three neighbors """
class Jointree:

    """
    Constructs a binary jointree from a total variable order.
    Constructs a dtree first, then converts to binary jointree.
    TBN nodes will be hosted at leaf jointree nodes (nodes with one neighbor).
    When computing the marginal of a node or its parents, its host will pull messages.
    """
    def __init__(self,tbn,elm_order,hard_evd_vars,trainable):
        assert len(tbn.nodes) > 1 # need two or more tbn nodes
        self.tbn           = tbn
        self.trainable     = trainable
        self.hard_evd_vars = hard_evd_vars
        self.nodes         = [] # jointree nodes
        self.hosts         = {} # maps tbn node to leaf jointree nodes hosting its cpt
        self.evd_ops       = {} # maps tbn node to evidence op
        self.sel_cpt_ops   = {} # maps tbn node to selected cpt op
        self.cpt_evd_ops   = {} # maps tbn node to cpt+evidence op (cpt can be selected)
        self.message_ops   = {} # maps signature to message
        self.signatures    = {} # maps (i,j) to signature for message i->j
        self.lookups       = 0
        self.hits          = 0
        
        # construct jointree
        root = self.__construct_dtree(elm_order)
        self.__convert_dtree_to_jointree(root)   
        if tbn._decoupling_of: 
            # tbn is the result of decoupling another tbn
            original_tbn = tbn._decoupling_of
            assert original_tbn._for_inference
            self.__recover_original_vars(original_tbn) 
        else:
            assert tbn._for_inference
    
    def __str__(self):
        return f'jointree: {len(self.nodes)} nodes'
        
        
    """ constructing jointree in three steps:
        1. construct dtree
        2. convert dtree to binary jointree
        3. replace replicated vars by originals (if any) """
    
    """ step 1: construct dtree based on elimination order """
    def __construct_dtree(self,elm_order):
        # variables (vars) are tbn nodes
        var2dts    = {} # maps var to dtree nodes that contain var
        dt2vars    = {} # maps dtree node to variables at/below node
            
        # adds leaf dtree node that hosts var
        # node will have one neighbor, neighbors=[parent]
        # will update dt2vars[] map
        def add_leaf_dt(var):
            dt             = Node(self,var)  # leaf dtree node
            dt2vars[dt]    = set(var.family) # vars appearing in dtree node
            var2dts[var]   = set()           # dtrees that contain var (set in next pass)
            self.nodes.append(dt)            # add to jointree
            
        # constructs an internal dtree node with dt1 and dt2 as its children
        # node will have three neighbors, neighbors=[dt1,dt2,parent]
        # will update var2dts[] and dt2vars[] maps
        def add_internal_dt(dt1,dt2):
            dt = Node(self,children=(dt1,dt2)) # internal dtree node
            # update maps
            vars1 = dt2vars[dt1]
            vars2 = dt2vars[dt2]
            vars  = vars1 | vars2
            # dt2vars map
            del dt2vars[dt1] # no longer needed
            del dt2vars[dt2] # no longer needed
            dt2vars[dt] = vars
            # var2dts map
            for v in vars1: var2dts[v].remove(dt1)
            for v in vars2: var2dts[v].remove(dt2)
            for v in vars : var2dts[v].add(dt)
            # add to jointree
            self.nodes.append(dt) 
            return dt
        
        # compose dtrees into one by constructing internal nodes
        def compose(dts,roots):
            dts.sort() # needed for deterministic behavior and width    
            dt = dts[0]
            for i in range(1,len(dts)):
                dt = add_internal_dt(dt,dts[i])
            roots.difference_update(dts) # dts are no longer roots
            roots.add(dt) # dt is a root
            return dt
        
        ### a. construct leaf dtree nodes
        for var in elm_order: 
            add_leaf_dt(var)
        for var in elm_order: # this needs to be done after constructing all leaf nodes
            dt = self.hosts[var]
            for v in var.family: var2dts[v].add(dt)
       
        ### b. construct internal dtree nodes
        roots = set(self.nodes) # dtree nodes with no parents
        for var in elm_order:   # eliminate vars by composing dtrees that contain it
            dts = var2dts[var]  # dtrees that contain var
            dt = compose(list(dts),roots)
            # dt is highest dtree node that contains var
            assert dt in var2dts[var] and len(var2dts[var]) == 1 
            dt2vars[dt].remove(var) # var does not appear in any other dtree
            del var2dts[var]        # no longer needed

        ### c. finalize dtree construction
        compose(list(roots),roots)  # roots is not a singleton if tbn is disconnected
        root = roots.pop()          # we have a single root now
        root.neighbors.append(None) # root has no parents
        assert not dt2vars[root]    # all variables have been eliminated
        del dt2vars[root]           # no longer needed
        
        # sanity checks
        assert not roots   # empty set
        assert not var2dts # empty map 
        assert not dt2vars # empty map
        return root        
        
    """ step 2: remove root of dtree """
    def __convert_dtree_to_jointree(self,root):
        # using dtree interpretation of neighbors
        child1 = lambda dt: dt.neighbors[0]
        child2 = lambda dt: dt.neighbors[1]
        parent = lambda dt: dt.neighbors[-1] # index is 0 (leaf dt) or 2 (internal dt)
        
        def replace_parent(dt,new_parent):
            old_parent = parent(dt)
            assert old_parent == root
            dt.neighbors[-1] = new_parent
            
        # connect children of root
        c1, c2 = child1(root), child2(root)
        replace_parent(c1,new_parent=c2) 
        replace_parent(c2,new_parent=c1)
        
        # remove root from jointree
        assert root == self.nodes[-1] # root is last node in self.nodes
        self.nodes.pop() 
        
    """ step 3: replace replicated variables with their originals """
    def __recover_original_vars(self,original_tbn):
        self.tbn = original_tbn
        for i in self.nodes:
            if i.leaf():
                original_var = i.var._original
                assert original_var and original_var.tbn == original_tbn
                i.var     = original_var
                i.is_host = False # updated later
        # -an original var and its cpt may now appear in multiple jointree
        #  leaves, but only one jointree leaf will be the host of var
        # -when decoupling a network, some nodes are marked as 'master'
        # -master nodes form a subnetwork that mirrors the original network
        # -the strategy below implies that hosts will also form a network 
        #  that mirrors the original one
        # -this is not strictly needed but is meant for systematically
        #  assigning hosts (which may actually improve message caching)
        # -the strategy is also used when dropping dead fcpts later
        #  (we prefer not to drop master fcpt, which hosts its var)
        self.hosts = {var._original:i for var,i in self.hosts.items() if var._master}
        assert len(self.hosts) == len(original_tbn.nodes) 
        # leaf nodes with replicated fcpts do not host variables
        for i in self.hosts.values(): 
            i.is_host = True
        
      
    """ hosts """      
    # returns a jointree leaf that hosts var and its cpt
    # var is a tbn node and may have multiple leaf nodes that contains its cpt,
    # but only one such node can be the host
    def host(self,var):
        assert var in self.hosts
        return self.hosts[var]
        
    """ evidence ops """
    def declare_evidence(self,vars,ops): # vars are tbn nodes
        for var, op in zip(vars,ops):
            assert not var in self.evd_ops
            self.evd_ops[var] = op
        
    def get_evd_op(self,var): # var is a tbn node
        assert var in self.evd_ops
        return self.evd_ops[var]
    
    """ selected cpt ops """
    def lookup_sel_cpt_op(self,var):  # var is a tbn node
        assert var.testing
        if var in self.sel_cpt_ops:
            return self.sel_cpt_ops[var]
        return None
        
    def save_sel_cpt_op(self,var,op): # var is a tbn node
        assert var.testing
        self.sel_cpt_ops[var] = op
        
    """ cpt+evd ops """
    def lookup_cpt_evd_op(self,var):  # var is a tbn node
        if var in self.cpt_evd_ops:
            return self.cpt_evd_ops[var]
        return None
        
    def save_cpt_evd_op(self,var,op): # var is a tbn node
        self.cpt_evd_ops[var] = op
    
    """ message ops """
    def lookup_message_op(self,signature):
        self.lookups += 1
        if signature in self.message_ops:
            self.hits += 1
            return self.message_ops[signature]
        return None
    
    def save_message_op(self,signature,op):
        self.message_ops[signature] = op
        
    """ return a jointree view for computing posterior of var or its parents """
    
    # context captures the pruned status of tbn nodes
    # var is a tbn node
    def view_for_query(self,var,context,verbose=True): 
        host = self.host(var)       # jointree node hosting cpt of var
        self.__prune(host,context)  # goes over all jointree nodes
        view = vw.View(host,self)   # view for unpruned jointree nodes
        view.prepare_for_inference(context,verbose)         
        return view

    # sets the is_pruned flag for jointree nodes
    # context provides the is_pruned flag for tbn nodes
    # no assumptions about the is_pruned flag of jointree nodes
    def __prune(self,host,context):
        view = vw.View(host,self,pruned=False) # includes all jointree nodes
    
        for i,_,c1,c2,_ in view.bottom_up():    
            if not c1: i.is_pruned = context.is_pruned(i.var)
            else:      i.is_pruned = c1.is_pruned and c2.is_pruned        
        
        host = view.host
        assert not context.is_pruned(host.var) # query var cannot be pruned
        host.is_pruned = False # contains query variable


    """ visualize jointree using graphviz """
    def dot(self,host,fname='jointree.gv',view=False,pruned=False):
        assert host.leaf()
        fname = paths.dot / fname
        g     = Digraph('jointree', filename=fname)
        g.attr(rankdir='TD')
        g.attr('node', fontsize='20')
        g.attr('node', style='filled')

        def fam2str(i):
            s = i.var.family
            l = list(s)
            l.sort()
            v = l[-1]
            label = ' '.join([n.name for n in l if n != v]) + ' &#8594; ' + v.name 
            return label  
        
        # clusters
        exclude = lambda i: pruned and i.is_pruned
        evd     = lambda i: i.leaf() and i.var in self.jointree.evd_ops
        name    = lambda i: f'{i.id}'
        label   = lambda i: f'{i.id}\n' + (fam2str(i) if i.leaf() else '')
        bcolor  = lambda i: 'grey' if exclude(i) else 'red' \
                            if (i.leaf() and i.is_replica()) else 'black'
        fcolor  = lambda i: 'transparent' if exclude(i) else 'lightyellow' \
                            if evd(i) else 'transparent'
        tcolor = lambda i: 'grey' if exclude(i) else 'black'
        shape  = lambda i: 'ellipse' 
        
        def edge(i,j):
            if j==None:
                g.node(name(i),label=label(i),shape=shape(i),color=bcolor(i),fillcolor=fcolor(i),fontcolor=tcolor(i))
            for k in i.neighbors:
                if k == j: continue
                edge(k,i) # recurse
                g.node(name(k),label=label(k),shape=shape(k),color=bcolor(k),fillcolor=fcolor(k),fontcolor=tcolor(k)) 
                g.edge(name(i),name(k),color='lightgrey',arrowhead='none')               

        edge(host,None)
            
        g.render(fname,view=view)
