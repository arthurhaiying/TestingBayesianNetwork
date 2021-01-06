from graphviz import Digraph
    
import decompose.graph as eo
from tbn.node import Node
import utils.paths as paths
import utils.utils as u
      
"""
TBN class
"""

class TBN:
    
    def __init__(self,name='tbn'):
        assert type(name) is str
        self.name           = name 
        self.nodes          = []    # tbn nodes, parents before children
        self.testing        = False # if any node is testing
        self.sel_type       = 'linear' # use linear selection by default
        self._n2o           = {}    # maps node name to node object
        self._cpt_ties      = {}    # maps cpt_tie to list of tied nodes with that id
        self._decoupling_of = None  # original tbn if this is a decoupled tbn
        self._for_inference = False # whether this network is prepared for inference
        self._add_order     = []    # order in which nodes were added
    
    def __str__(self):
        return 'TBN ' + self.name + ": " + str(len(self.nodes)) + ' nodes'

    # -adds node to tbn
    # -nodes must be added in topological order: parents before children  
    def add(self,node):
        u.input_check(isinstance(node, Node),
            f'{node} is not a TBN node object')
        u.input_check(node.tbn is None, '' if node.tbn is None else \
            f'node {node.name} is already in a different TBN {node.tbn.name}')
        u.input_check(not node.name in self._n2o,
            f'a node with name {node.name} already exists in TBN {self.name}')
        for p in node.parents: # parents must have already been added
            u.input_check(p.tbn is self and p.name in self._n2o,
                f'parent {p.name} of node {node.name} has not been added to TBN {self.name}')
        assert self._for_inference == node._for_inference
        
        # check if node is tied and process accordingly
        tie_id = node.cpt_tie
        if tie_id: # node is tied to another
            assert not node.fixed_cpt    # only trainable cpts can be tied
            if tie_id in self._cpt_ties: # a node tied to this one has already been added
                tied_nodes = self._cpt_ties[tie_id] 
                tied_node  = tied_nodes[0]
                assert node.shape() == tied_node.shape() # tied cpts should have same shape
                tied_nodes.append(node)
            else: # no other node tied to this one has been added yet
                self._cpt_ties[tie_id] = [node]
            
        # connect node to parents
        for p in node.parents: 
            p._children.append(node)
        
        # add node
        node._tbn            = self
        self.testing        |= node.testing
        self._n2o[node.name] = node
        self._add_order.append(node)
        self.nodes.append(node)

    # set the cpt selection function used by this tbn
    def set_select_type(self,type):
        u.input_check(type in ('linear', 'threshold', 'sigmoid'), "Invalid selection type. Must be linear, threshold, or sigmoid")
        self.sel_type = type

    def get_select_type(self):
        return self.sel_type

    # returns a node object given a node name
    def node(self,name):
        node = self._n2o.get(name,None)
        u.input_check(node,f'node {name} does not exist in TBN {self.name}')
        return node
        
    # checks if name is the name of a tbn node
    def is_node_name(self,name):
        return name in self._n2o
        
    # returns all node with tie, if any
    def tied_nodes(self,tie):
        if tie in self._cpt_ties:
            return self._cpt_ties[tie]
        return None
        
    # returns an elimination order
    def elm_order(self,method,wait=None):
        assert method=='minfill' or wait
        return eo.elm_order(self,method,wait)

    # whether tbn is connected
    def is_connected(self):
        return self.nodes[0].connected_nodes() == set(self.nodes)
        
    # returns a copy of tbn prepared for inference:
    #   cpt's processed and values/edges pruned
    def copy_for_inference(self):
        assert not self._for_inference
        name = f'{self.name}_inf'
        net  = TBN(name)
        net.sel_type = self.sel_type
        net._for_inference = True
        for n in self._add_order: # use exact order in which nodes were added
            m = n.copy_for_inference(net)
            net.add(m)
        return net
        
    # visualizes a tbn using graphiz
    def dot(self,fname='tbn.gv',view=True,context=None):
        fname = paths.dots / fname
        d     = Digraph(self.name, filename=fname)
        d.attr(rankdir='TD')
        d.attr('node', shape='circle')
        d.attr('node', fontsize='20')
        d.attr('node', style='filled')
        
        pruned   = lambda n: context and context.is_pruned(n)
        evidence = lambda n: context and context.has_evidence(n)

        for n in self.nodes:
#            if pruned(n): continue
            # node
            bcolor = 'lightgrey' if pruned(n) else 'red' if n.testing else 'black'
            fcolor = 'lightyellow' if evidence(n) else 'transparent'
            tcolor = 'red' if len(n.values)==1 else 'lightgrey' if pruned(n) else 'black'
            d.node(n.name,color=bcolor,fillcolor=fcolor,fontcolor=tcolor)
            # edges
            bcolor = 'lightgrey' if pruned(n) else 'black'
            for p in n.parents:
#                if pruned(n): continue
                d.edge(p.name, n.name,color=bcolor) 
        d.render(fname,view=view)
    