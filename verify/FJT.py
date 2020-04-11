import numpy as np
from itertools import count as counter
from functools import reduce
from math import log2
from graphviz import Digraph

import utils.VE as VE
import utils.paths as paths
import utils.utils as u

"""
An implementation of the algorithm that exploits functional dependencies for
the purpose of verification against a crude implementation of VE. This is a
direct implementation of the algorithm that bypasses the use of tensorflow.
It is therefore a mockup of the real algorithm for generating TACs/ACs as
tensflow graphs.

Networks with functional CPTs are generated randomly, jointree views are
generated randomly too.

Interface: Net.verify() which computes all node marginals using two methods:
a crude VE and the jointree algorithm which shrinks separators/clusters
due to functional dependencies.
"""


"""
Variables and CPTs.
"""

class Var:
    names = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    def __init__(self,index,functional):
        self.id     = index
        self.name   = Var.names[index]
        self.func   = functional
        self.values = (1,2,3)[:np.random.randint(2,4)]
        self.card   = len(self.values)
        self.cpt    = None
        self.factor = None  # factor of cpt (for inference using VE)
        self.batch  = False # for compatibility with Var in VE.Var
        
    # for sorting
    def __lt__(self,other):
        return self.id < other.id
        
    # for printing
    def __str__(self):
        return self.name
        
    def set_cpt(self,cpt):
        self.cpt    = cpt
        self.factor = VE.Factor(cpt.table,cpt.vars,sort=True) 
        
    def random_distribution(self):
        if self.func: # functional distribution
            distribution = [0.]*self.card
            index        = np.random.randint(self.card)
            distribution[index] = 1.
        else:
            distribution  = np.random.rand(self.card)
            norm_constant = sum(distribution)
            distribution  = [n/norm_constant for n in distribution] 
        return distribution
        
class CPT:
    def __init__(self,var,parents):
        self.var     = var
        self.parents = parents
        self.vars    = [*parents,var]
        self.table   = np.array(self.__random_table())
        
    def __random_table(self,index=0):
        if index == len(self.parents):
            return self.var.random_distribution()
        else:
            card = self.parents[index].card
            return [self.__random_table(index+1) for _ in range(card)]
 
"""
Bayesian network.
"""

class Net:

    # constructs a random network
    # vcount: number of variables
    # fcount: number of functional variables
    # max_pcount: maximum number of parents per node
    def __init__(self,vcount,fcount,max_pcount):
        assert vcount >= 2 and vcount <= 26
        assert fcount >= 0 and fcount <= vcount and max_pcount < vcount
        
        self.vcount = vcount # number of variables
        self.fcount = fcount # number of functional variables
        indices     = range(vcount)
        findices    = set(np.random.choice(vcount,fcount,replace=False))
        self.vars   = tuple(Var(index,index in findices) for index in indices)
        
        parents_set = []
        for var in self.vars:
            pcount  = np.random.randint(1+min(var.id,max_pcount))
            parents = np.random.choice(parents_set,pcount,replace=False)
            cpt     = CPT(var,parents)
            var.set_cpt(cpt)
            parents_set.append(var)
          
    # crude VE: construct joint then sum out variables (a safe implementation)
    def marginals(self):
        joint = VE.Factor.one()
        for var in self.vars:
            joint = joint.multiply(var.factor)
 
        def sum_all_but(qvar):
            marginal = joint
            for var in self.vars:
                if var != qvar:
                    marginal = marginal.sumout(var)
            marginal = marginal.table
            assert np.allclose(1.,np.sum(marginal)) # marginal is normalized
            return marginal
            
        return tuple(sum_all_but(var) for var in self.vars)
            
    # compare crude VE with jointree (replicates functional cpts and shrinks separators)
    # max_duplicates: max number of replicas of a functional cpts in the jointree
    def verify(self,max_duplicates,seed=None):
        if seed is not None: np.random.seed(seed)
        #print(f'\n===verifying bn: {self.vcount} vars, {self.fcount} functional')
        marginals1 = self.marginals()
        for marginal1, var in zip(marginals1,self.vars):
            view      = View(self,var,max_duplicates)
            marginal2 = view.marginal()
            equal     = np.isclose(marginal1,marginal2)
            ok        = np.all(equal) 
            if not ok:
                print(f'Mismatch on variable {var.name}')
                print('  marginal1\n  ',marginal1[np.logical_not(equal)])
                print('  marginal2\n  ',marginal2[np.logical_not(equal)])
                print('***Ouch!!!\n')
                #net.dot()
                #view.dot()
                quit()
            size1, size2 = view.size()
            #print(f'var {var.name}: {size1:.1f} --> {size2:.1f}')
            assert size2 <= size1
            print('.',end='',flush=True)
        print('ok.',end='',flush=True)
        
    # visualize network
    def dot(self,fname='net.gv',view=True):
        fname = paths.dot / fname
        d     = Digraph('net', filename=fname)
        d.attr(rankdir='TD')
        d.attr('node', shape='circle')
        d.attr('node', fontsize='20')
        d.attr('node', style='filled')
        for v in self.vars:
            color = 'red' if v.func else 'black'
            d.node(v.name,color=color,fillcolor='transparent',fontcolor='black')
            for p in v.cpt.parents: d.edge(p.name,v.name,color='black') 
        d.render(fname,view=view)
        

"""
Jointree view.
"""

class Node:

    def __init__(self,left=None,right=None,var=None):
        assert var or (left and right)
        assert (var and left==None and right==None) or (var==None and left and right)

        self.id      = None
        self.left    = left
        self.right   = right
        self.var     = var 
        self.parent  = None
        self.sibling = None
        self.sep     = None
        self.cls     = None
        self.vars    = None
        self.fvars   = None
        self.osep    = None # original separator, before shrinking
        self.size    = None # size of separators below node i
        
        if not var: left.parent = right.parent = self
    
    def leaf(self):
        return self.var is not None
        
    def root(self):
        return self.parent.parent == None
        
        
class View:
    
    # fields: leaves, nodes, root, host
    # nodes are ordered children before parents
    # qvar: query variable
    # max_duplicates: max number of replicas for a functional cpt
    def __init__(self,net,qvar,max_duplicates):
        self.net    = net 
        self.leaves = set()
        # construct leaves
        for var in net.vars:
            if var.func:
                duplicates = np.random.randint(2,1+max_duplicates)
                for _ in range(duplicates):
                    self.leaves.add(Node(var=var))
            else:
                self.leaves.add(Node(var=var))
        # construct random dtree
        trees  = list(self.leaves)
        rindex = lambda: np.random.randint(len(trees))
        while len(trees) > 1:
            index = rindex()
            tree1 = trees[index]
            trees[index] = trees[-1]
            trees.pop()
            index = rindex()
            tree2 = trees[index]
            tree  = Node(left=tree1,right=tree2)
            trees[index] = tree
        # node trees[0] is root of dtree and will be discarded during orientation
        # find host and root
        host = None
        for leaf in self.leaves:
            if leaf.var == qvar:
                host = leaf
                break
        assert host
        root = host.parent
        self.host, self.root = host, root # before calling orient
        
        # construct jointree view from dtree (will bypass node trees[0])
        self.nodes = [] # filled by orient, bottom-up
        self.__orient(root,host) # may change self.root
        # host is not connected to view: its left, right, parent are undefined
        # only connection to view is being parent of root
        host.left = host.right = host.parent = None
        
        # number view nodes
        id = counter(0)
        for i in self.nodes: i.id = next(id)
        host.id = next(id)
        
        # set separators and clusters
        self.__initialize()
        self.__shrink()

    # convert dtree to binary jointree
    # sets left, right, sibling, parent of nodes
    def __orient(self,i,p):
    
        def children(n,exclude):
            assert exclude.parent
            if n.leaf(): return n, exclude, tuple()
            if not n.parent: # bypass root of dtree
                assert n == self.root and exclude == self.host
                m = n.left if exclude == n.right else n.right
                self.root = m
                return m, exclude, (m.left,m.right) 
            if not n.parent.parent: # bypass root of dtree
                c1 = n.parent.left if n == n.parent.right else n.parent.right
                c2 = n.left if exclude == n.right else n.right
                c1.parent = n
                return n, exclude, (c1,c2)
            return n, exclude, tuple(c for c in (n.left,n.right,n.parent) if c != exclude)
    
        i, p, children_ = children(i,p)
        if children_:
            left, right = children_
            self.__orient(left,i)
            self.__orient(right,i)
            i.left, i.right = left, right
            left.sibling, right.sibling = right, left
            
        i.parent = p
        self.nodes.append(i) # bottom-up
            
        
    # set sep, vars, fvars, etc
    def __initialize(self):
        vars_  = lambda i: set(i.var.cpt.vars)
        fvars_ = lambda i: set([i.var]) if i.var.func else set()
        # bottom-up
        for i in self.nodes:
            if i.leaf():
                i.vars  = vars_(i)
                i.fvars = fvars_(i)
            else:  
                c1, c2, = i.left, i.right
                i.vars  = c1.vars  | c2.vars
                i.fvars = c1.fvars | c2.fvars 
        # top-down
        for i in reversed(self.nodes):
            if i.root(): 
                i.sep  = i.vars & vars_(self.host)
                i.osep = set(i.sep) # save copy of original separator
            else:
                p, s   = i.parent, i.sibling
                i.sep  = i.vars & (s.vars | p.sep) 
                i.osep = set(i.sep) # save copy of original separator
        # clusters
        self.__set_cls()
        
    # shrink separators due to functional dependencies
    def __shrink(self):
        fvars_   = lambda i: set([i.var]) if i.var.func else set()
        sep_size = lambda i: reduce(lambda x,y: x*y, u.map('card',i.sep),1)
        # heuristic for deciding which branch to sum from
        for i in self.nodes: # bottom-up
            if i.leaf(): 
                i.size = 0
            else: 
                c1, c2 = i.left, i.right
                i.size = c1.size + c2.size + sep_size(c1) + sep_size(c2)
        # MUST process siblings simultaneously before processing their children
        # this is important to enforce the running intersection property
        def down_sum(i,p):
            if i.leaf(): return
            c1, c2 = i.left, i.right
            sum = c1.fvars & c2.fvars
            if sum:
                if c1.size < c2.size or (c1.size == c2.size and c1.id < c2.id):
                    c1.sep -= sum
                else:
                    c2.sep -= sum
            # propagate separator shrinking
            c2.sep &= c1.sep | i.sep 
            c1.sep &= c2.sep | i.sep
            down_sum(c1,i)
            down_sum(c2,i)
        r, h = self.root, self.host
        r.sep -= r.fvars & fvars_(h) 
        down_sum(r,h)
        # update clusters
        self.__set_cls()

    # set clusters
    def __set_cls(self):
        for i in self.nodes: # bottom-up
            if i.leaf(): 
                i.cls = i.vars
            else: 
                c1, c2 = i.left, i.right
                i.cls = c1.sep | c2.sep
                if not ((i.cls == c1.sep | i.sep) and (i.cls == c2.sep | i.sep)):
                    print('\ncls',i.id)
                    self.dot()
                assert (i.cls == c1.sep | i.sep) and (i.cls == c2.sep | i.sep)
                # the following condition is weaker than the one for classical jointrees
                # equality does not hold since functional cpts may be dropped from jointree
                assert (c1.sep <= c1.cls & i.cls) and (c2.sep <= c2.cls & i.cls)
                
    """
    Interface for View: size, marginal and visualization
    """
    
    # size of view: log sum of separator sizes (original and shrunk)
    def size(self):
        def setsize(vars):
            cp = 1
            for var in vars: cp *= var.card
            return cp
        size1 = size2 = 0
        for i in self.nodes:
            size1 += setsize(i.osep)
            size2 += setsize(i.sep)
        return log2(size1), log2(size2)
        
    # compute marginal based on jointree view
    def marginal(self):
        factor = {} # maps node to factor computed at node
        for i in self.nodes: # bottom-up
            if i.leaf():
                factor[i] = i.var.factor
            else:
                c1, c2 = i.left, i.right
                f1, f2 = factor[c1], factor[c2]
                factor[i] = f1.multiply(f2).project(i.sep)
        f = factor[self.root].multiply(self.host.var.factor)
        f = f.project([self.host.var])
        f = f.table
        if not np.allclose(1.,np.sum(f)):  # marginal is normalized
            print('\njointree view not normalized')
            self.net.dot()
            self.dot()
            quit()
        return f
        
    # visualize jointree view
    def dot(self,fname='view.gv',view=True):
        fname = paths.dot / fname
        g     = Digraph('view', filename=fname)
        g.attr(rankdir='TD')
        g.attr('node', fontsize='20')
        g.attr('node', style='filled')
        
        def set2str(s):
            l = list(s)
            l.sort(key=lambda i: i.name)
            label = u.unpack(l,'name')
            return label
            
        def cpt2str(i):
            l = list(i.var.cpt.vars)
            l.sort()
            v = l[-1]
            label = ' '.join([n.name for n in l if n != v]) + ' &#8594; ' + v.name # html arrow
            return label
            
        def sep2str(i):
            l = list(i.osep)
            l.sort(key=lambda i: i.name)
            summed  = lambda name: f'<font color=\'red\'><b>{name}</b></font>'
            html    = ''.join([summed(v.name) if v not in i.sep else v.name for v in l]) 
            return '<' + html + '>'
       
        h = self.host
        # clusters
        name   = lambda i: f'{i.id}'
        label  = lambda i: f'{i.id}\n' + (cpt2str(i) if i.leaf() else '') #set2str(i.cls))
        fcpt   = lambda i: i.leaf() and i.var.func
        bcolor = lambda i: 'red' if fcpt(i) else 'black'
        
        for i in reversed(self.nodes): # bottom up
            p = i.parent
            # cluster
            g.node(name(i),label=label(i),shape='ellipse',color=bcolor(i),fillcolor='transparent') 
            # separator
            name_ = f'{name(i)}_{name(p)}'
            g.node(name_,label=sep2str(i),shape='rectangle',color='black',fillcolor='transparent') #'gray93'
            # edges
            g.edge(name(p),name_,color='black',arrowhead='none')
            g.edge(name_,name(i),color='black',arrowhead='none')

        # host
        g.node(name(h),label=label(h),shape='ellipse',color=bcolor(h),fillcolor='transparent')
            
        g.render(fname,view=view)