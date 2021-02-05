from math import log2
from functools import reduce
from graphviz import Digraph
from collections import deque

#import compile.jointree as jointree
import compile.separators as separators
import utils.paths as paths
import utils.utils as u
        
# utilities for measuring size of clusters and separators
log2x = lambda s: log2(s) if s > 0 else 0
        
"""
A jointree VIEW based on a 'host:' a leaf jointree node.
A view is useful for traversing the jointree top-down or bottom-up, in a manner 
akin to the pull and push passes on classical jointrees. A view is particularly
useful for skipping pruned jointree nodes during a traversal, and skipping jointree
nodes that become redundant due to pruning (nodes with two unpruned neighbors).
The view root is the single neighbor of host (host is not part of the view).
If a view is constructed with the 'pruned' flag set to True, pruned jointree nodes
and ones with two unpruned neighbors will be excluded from the view. It is possible
that only the host is unpruned, in which case the view is empty (has no root).
Every node in a view has:
    # an unpruned parent: its neighbor closest to host.
      (the host is the root's parent)
    # zero or two unpruned children: its neighbors furthest from host.
    
Nodes of a view are listed deeper before shallower.
Visiting nodes left-to-right (normal order)  amounts to a breadth-first bottom-up pass.
Visiting nodes right-to-left (reverse order) amounts to a breadth-first top-down pass.
"""
    
class View:
    def __init__(self,host,jointree,pruned=True):
        # if pruned=False, we include all jointree nodes in the view
        # otherwise, we include only unpruned nodes
        include = lambda i: not pruned or not i.is_pruned
        
        # host must be a leaf node and cannot be pruned
        assert host.leaf() and include(host)
        
        neighbor      = host.neighbor()
        self.host     = host
        self.jointree = jointree
        self.nodes    = []  # children before parents, deeper before shallower
        self.empty    = not include(neighbor)
        self.cls_     = {}  # maps a view node to its cluster
        self.fcpts    = {}  # maps fvar to leaf nodes containing its fcpt
        self.evd_at   = {}  # maps var to view node that includes evidence on var
        
        if self.empty: # view has no nodes
            self.nodes_set = set()
            return
            
        self.root       = neighbor       
        self.children_  = {} # maps a view node to its children (zero or two)
        self.parent_    = {} # maps a view node to its parent
        self.sep_       = {} # maps a view node to its separator with its parent
        self.signature_ = {} # maps a view edge to its signature
        self.depth_     = {} # maps a view node to its depth
        
        fringe = [(self.root,host,host)] # (node,parent,ancestor)
        self.depth_[host] = 0
        while fringe:
            # look for a node with 0 or 2 children
            i, p, a  = fringe.pop()
            children = tuple(k for k in i.neighbors if k != p and include(k))
            count    = len(children)
            if count == 1: # skip i
                assert pruned
                c = children[0] # single child of i, other child pruned
                fringe.append((c,i,a))
                if i == self.root: self.root = c
                continue
            assert count in (0,2)
            # add node to view
            self.depth_[i]    = self.depth_[a] + 1 
            self.parent_[i]   = a
            self.children_[i] = []
            if a != host: 
                self.children_[a].append(i)
            self.nodes.append(i)
            # queue children
            for c in children: 
                fringe.append((c,i,i))
  
        # host not included in nodes
        self.nodes.sort(key=lambda i: self.depth_[i], reverse=True) # deeper nodes first
        self.nodes = deque(self.nodes)   # more efficient for traversing backward
        self.nodes_set = set(self.nodes) # for fast lookup
      
    """ view structure """
    
    # leaf, parent, chilren are not defined for host
    # sibling not defined for root
    
    def leaf(self,i):
        assert i != self.host
        return not self.children_[i]
        
    def parent(self,i):
        assert i != self.host
        return self.parent_[i]

    def children(self,i):
        assert i != self.host
        return self.children_[i]
        
    def sibling(self,i):
        assert i != self.root and i != self.host
        p = self.parent_[i]
        c = self.children_[p]
        return c[0] if i == c[1] else c[1]
        
    def depth(self,i):
        return self.depth_[i]
        
    # returns children and sibling of i (None if they do not exist)
    # used extensively when traversing views
    def edge_neighbors(self,i,p):
        c1  = c2 = s = None
        c12 = self.children_[i]
        if c12:
            c1, c2 = c12
        if i != self.root:
            c12 = self.children_[p]
            s = c12[0] if i == c12[1] else c12[1]
        return c1, c2, s
    
    """ signatures """
    
    def signature(self,i,j):
        return self.signature_[(i,j)]
        
    def signature_set(self,i,j,signature):
        self.signature_[(i,j)] = signature
        
    """ clusters and separators """
    
    def cls(self,i):
        assert i == self.host or i in self.nodes_set
        if i in self.cls_: return self.cls_[i]
        return None
        
    def sep(self,i):
        assert i != self.host and i in self.nodes_set
        if i in self.sep_: return self.sep_[i]
        return None
        
    def cls_set(self,i,cls):
        self.cls_[i] = cls
        
    def cls_union(self,i,vars):
        self.cls_[i] |= vars
        
    def sep_set(self,i,sep):
        self.sep_[i] = sep
        
    def sep_intersect(self,i,vars):
        self.sep_[i] &= vars
        
    def sep_minus(self,i,vars):
        self.sep_[i] -= vars
        
    def sep_union(self,i,vars):
        self.sep_[i] |= vars
        
    # size is the number of instantiations for cluster/separator
    
    def cls_size(self,i):
        cls = self.cls(i)
        return reduce(lambda x,y: x*y, u.map('card',cls),1)

    def sep_size(self,i):
        sep = self.sep(i)
        return reduce(lambda x,y: x*y, u.map('card',sep),1) 
        
    # rank is the number of variables in a cluster/separator
    # binary rank (brank) is log2 the number of instantiations of a cluster/separator
    
    def cls_brank(self,i): 
        return log2x(self.cls_size(i))
        
    def cls_rank(self,i): 
        return len(self.cls(i))
        
    def sep_brank(self,i): 
        return log2x(self.sep_size(i))
        
    def sep_rank(self,i): 
        return len(self.sep(i))
            
    """ functional vars and evidence """
    
    # whether var is functional    
    def is_fvar(self,var):
        return var in self.fcpts
    
    # whether view node i has a functional cpt
    def has_fcpt(self,i):
        assert i==self.host or self.leaf(i)
        return i.var in self.fcpts
            
    # whether var has evidence that is assigned to view node i
    def has_evidence_at(self,var,i):
        # var is redundant but added for clarity
        # i may be host
        assert var==i.var
        return var in self.evd_at and i in self.evd_at[var]
        
    # whether there is evidence on var
    def has_evidence(self,var):
        return var in self.evd_at
        
    """ travering a view """
    
    # generates edges (i p) bottom-up while also returning neighbors of edge nodes (if any)
    def bottom_up(self):
        for i in self.nodes:
            p = self.parent(i)
            c1, c2, s = self.edge_neighbors(i,p) # some neighbors may be none
            yield i, p, c1, c2, s
                
    # generates edges (i p) top-down while also returning neighbors of edge nodes (if any)
    def top_down(self):
        for i in reversed(self.nodes):
            p = self.parent(i)
            c1, c2, s = self.edge_neighbors(i,p) # some neighbors may be none
            yield i, p, c1, c2, s
    
    """ view ranks """
    
    # rank is the number of variables in a cluster/separator
    # binary rank (brank) is log2 the number of instantiations of a cluster/separator
    def ranks(self):
        cbrank = max(self.cls_brank(i) for i in self.nodes)
        crank  = max(self.cls_rank(i)  for i in self.nodes)
        sbrank = max(self.sep_brank(i) for i in self.nodes)
        srank  = max(self.sep_rank(i)  for i in self.nodes)
        
        cbrank = max(cbrank,self.cls_brank(self.host))
        crank  = max(crank,self.cls_rank(self.host))
        
        cbsize  = log2x(sum(self.cls_size(i) for i in self.nodes) + self.cls_size(self.host))
        sbsize  = log2x(sum(self.sep_size(i) for i in self.nodes))
        
        return cbrank, crank, sbrank, srank, cbsize, sbsize
    
    # returns ranks as a string
    def ranks_str(self):
        cbrank, crank, sbrank, srank, cbsize, sbsize = self.ranks()
        return (f'cls ({cbrank:.1f} {crank} {cbsize:.2f}), '
                f'sep ({sbrank:.1f} {srank} {sbsize:.2f})')
    
    
    """ ghost jointree node used to extend view when replicating functional cpts """

    class Ghost:
        def __init__(self,var=None):
            import compile.jointree as jointree
            self.id      = next(jointree.Node.id)
            self.var     = var
            self.is_host = False # ghosts do not host variables
        def __str__(self):
            if self.var: return f'Ghost {self.id}: {self.var.name}'
            return f'Ghost {self.id}'
            
            
    """ compute separators, clusters, signatures and assign evidence to view nodes """
    
    def prepare_for_inference(self,context,verbose): 
        jt = self.jointree
        h  = self.host
        # view host is special: (1) does not appear in view.nodes, (2) always 
        # jointree node (never a ghost node), (3) only properties defined for
        # host are its cluster, whether it has fcpt, and whether it has evidence.
        assert h not in self.nodes_set and type(h) != self.Ghost and \
            h not in self.parent_ and h not in self.children_ 
       
        # only three properties are defined for host
        QVAR = h.var # query variable
        self.cls_set(h,set(QVAR.family))
        if context.has_evidence(QVAR):       self.evd_at[QVAR] = set([h])
        if QVAR.is_functional(jt.trainable): self.fcpts[QVAR]  = [h]

        # an empty view has a host but no nodes
        if self.empty: 
            assert not self.nodes_set
            return # code below will do nothing
        
        # compute the view's separators and clusters
        separators.set_separators_and_clusters(self,jt.trainable,verbose)

        # decide which view nodes will integrate evidence
        self.__assign_evidence(context,jt)
        
        # set the signatures of messages in view (signatures are keys used for caching)
        self.__set_signatures()
        
        # verify various properties of separators and clusters
        self.__verify(verbose)
               
        #self.dot(view=True)
        #input('continue?')
            
            
    """ computing signatures for caching messages """
    
    # A signature for message i->j is a pair (t1,t2):
    # sig1: tbn nodes whose cpts are involved in computing message i->j
    # sig2: separator on edge i->j  
    # Signatures are cache keys used to store/lookup messages during inference
    
    def __set_signatures(self): 
        # maps (i,p) to vars whose cpts are involved in computing message i->p
        # a functional var may have multiple cpts, but their number does not matter
        cpts = {}
        for i, p, c1, c2, _ in self.bottom_up():
            if not c1: cpts[(i,p)] = frozenset([i.var])
            else: cpts[(i,p)] = cpts[(c1,i)] | cpts[(c2,i)]   
            sig1 = cpts[(i,p)] # frozenset
            sig2 = frozenset(self.sep(i))
            signature = (sig1,sig2) # tuple
            self.signature_set(i,p,signature)   
            
            
    """ assign evidence vars to view nodes """
    
    # assign evidence to leaf nodes (evidence on view host was assigned earlier)
    # -soft evidence on var is assigned to _one_ node that contains the cpt of var
    # -hard evidence on var is assigned to _all_ nodes that contain the cpt of var
    #  (fvars may have multiple nodes that contain their cpt)
    # -after assigning evidence, we can start using view.has_evidence(var) and 
    #  view.has_evidence_at(var,i)
    
    def __assign_evidence(self,context,jt):
        hard_evd_vars = jt.hard_evd_vars
        for i in self.nodes:
            if self.leaf(i) and context.has_evidence(i.var):
                var = i.var # may be the var at view host (when we have functional cpts)
                if var not in self.evd_at:
                    self.evd_at[var] = set([i])
                else:
                    self.evd_at[var].add(i)

        # need to pick only one view node to host each soft evidence
        for var, nodes in self.evd_at.items():
            if var not in hard_evd_vars:
                # var has soft evidence: enter evidence at its jointree host
                i = jt.host(var)
                assert i in nodes
                self.evd_at[var] = set([i])
        
    """ verifying clusters and separators """
    
    def __verify(self,verbose):  
        cls = lambda i: self.cls(i)
        sep = lambda i: self.sep(i)
        
        # remove functional and clamped vars from clusters/separators
        rem_f  = lambda vars: set(var for var in vars if not self.is_fvar(var))
        rem_fc = lambda vars: set(var for var in vars if not self.is_fvar(var) and not var._clamped)
             
        # host and root are special, host contains the cpt for query var (QVAR)       
        h, r, QVAR = self.host, self.root, self.host.var

        assert h not in self.nodes_set and type(h) != self.Ghost and \
            h not in self.parent_ and h not in self.children_ and \
            h in self.cls_ and not h in self.sep_ and self.parent(r)==h
        
        # clamped vars cannot be replicated    
        assert all(not var._clamped for var in self.fcpts)
        
        for i, p, c1, c2, _ in self.bottom_up():
        
            assert all(var.card > 1 for var in sep(i))
            assert all(var.card > 1 for var in cls(i))
            
            # the following is a weaker property when shrinking separators due to fvars
            assert sep(i) <= cls(i) & cls(p) # would be == for classical jointrees
            
            # a functional var may appear on both sides of an edge, but not in the
            # separator of that edge (in that case, var has fcpts on both sides of edge)
            if self.leaf(i) and p == h:
                # view contains only root and host: both clusters may contain a clamped
                # var which would not appear in the separator connecting them
                assert i == r and r in self.nodes and len(self.nodes) == 1
                assert rem_fc(sep(i)) == rem_fc(cls(i)) & rem_fc(cls(p))
            else:
                assert rem_f(sep(i)) == rem_f(cls(i)) & rem_f(cls(p))
            
            # clamped vars never appear in separators
            assert not any(var._clamped for var in sep(i))
            
            if not c1: # leaf cluster i
                assert cls(i) == set(i.var.family)
                assert sep(i) <= cls(i)
                assert not i.var._clamped or self.has_evidence_at(i.var,i)
                for var in cls(i)-sep(i): # var is summed out from leaf cluster i
                    if var==i.var:
                        assert self.has_evidence_at(var,i) # otherwise, dead cpt                                                  
                    else: # var is a parent in cpt of cluster i
                        assert var._clamped # has hard evidence
            else: # internal cluster i
                # the following implies that each separator is a subset of the
                # union of other two separators; hence, if a variable appears in
                # one separator, it must also appear in at least another one
                assert sep(i) <= sep(c1) | sep(c2)
                assert sep(c1) <= sep(i) | sep(c2)
                assert sep(c2) <= sep(i) | sep(c1)
                # redundant tests
                assert cls(i) == sep(c1) | sep(c2)
                assert cls(i) == sep(c1) | sep(i)
                assert cls(i) == sep(c2) | sep(i)
                
        # host contains a cpt for the query var (QVAR)
        # if the cpt is functional, it cannot be dead as it contains the query var
        assert all(var.card > 1 for var in cls(h))
        assert not QVAR._clamped # only TAC inputs can be clamped (query cannot be input)
        for var in cls(h)-sep(r):
            assert var==QVAR or var._clamped      
            
        
    """ visualize view using graphiz """

    def dot(self,fname='view.gv',view=True):
        fname = paths.dot / fname
        g     = Digraph('view', filename=fname)
        g.attr(rankdir='TD')
        g.attr('node', fontsize='20')
        g.attr('node', style='filled')
        
        def set2str(i,j=None):
            if j and self.sep(i):       s = self.sep(i)
            elif not j and self.cls(i): s = self.cls(i)
            else: return '' # cls/sep not set yet
            l = list(s)
            l.sort(key=lambda i: i.name)
            label = u.unpack(l,'name')
            return label
            
        def fam2str(i):
            l = list(i.var.family)
            l.sort()
            v = l[-1]
            label = ' '.join([n.name for n in l if n != v]) + ' &#8594; ' + v.name # html arrow
            return label  
       
        h = self.host
        # clusters
        leaf   = lambda i: i==h or self.leaf(i)
        evd    = lambda i: leaf(i) and self.has_evidence_at(i.var,i)
        fcpt   = lambda i: leaf(i) and self.has_fcpt(i)
        name   = lambda i: f'{i.id}'
        label  = lambda i: f'{i.id}\n' + (fam2str(i) if leaf(i) else set2str(i))
        bcolor = lambda i: 'red' if fcpt(i) else 'black'
        fcolor = lambda i: 'transparent' if evd(i) else 'transparent'
        shape  = lambda i: 'hexagon' if type(i)==self.Ghost else 'ellipse' 
        
        # separators
        label_  = lambda i, p: set2str(i,p)
        bcolor_ = 'black'
        fcolor_ = 'gray93'
        shape_  = 'rectangle'
        
        for i,p,c1,c2,_ in self.bottom_up():
            # cls(i)
            g.node(name(i),label=label(i),shape=shape(i),color=bcolor(i),fillcolor=fcolor(i)) 
            # sep(p)
            name_ = f'{name(i)}_{name(p)}'
            g.node(name_,label=label_(i,p),shape=shape_,color=bcolor_,fillcolor=fcolor_)
            # edges cls(i)--sep(i)--cls(p)
            g.edge(name(p),name_,color=bcolor_,arrowhead='none')
            g.edge(name_,name(i),color=bcolor_,arrowhead='none')
            
            # consistent layout of children
            if not c1: continue
            with g.subgraph() as s:
                s.attr(rank = 'same',rankdir='LR')
                l, r = (c1,c2) if c1.id < c2.id else (c2,c1)
                s.node(name(l))
                s.node(name(r))
    
        # host cluster
        g.node(name(h),label=label(h),shape=shape(h),color=bcolor(h),fillcolor=fcolor(h))
            
        g.render(fname,view=view)