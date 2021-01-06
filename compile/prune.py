
"""
Context for pruning tbn nodes given query and evidence.

The context exposes the following:

    # pruned(): maps nodes to True/False
    # evidence(): maps nodes to True/False
    # nodes: tuple of unpruned nodes, parents before children
    # testing_nodes: tuple of unpruned testing nodes, parents before children
    # live_count: number of live testing nodes (updated by for_selection context)

Pruning is based on the following:

    # any node not connected to query is pruned
    # a node is pruned if it is not one of, or an ancestor of one the following:
        # query node
        # connected evidence node
"""

class for_node_posterior:

    def __init__(self,query_node,evidence_nodes,tbn):
        self._pruned   = None # set of pruned nodes
        self._evidence = None # set of unpruned evidence nodes
        
        # find nodes connected to query in network
        connected = self.__connected_nodes(query_node)
        evidence  = set(evidence_nodes) & connected
        
        # prune (nodes not in active are pruned)
        # initialize active to query node and connected evidence nodes
        active = set(evidence)
        active.add(query_node)

        # traverse bottom up, activating parents of active nodes
        for n in reversed(tbn.nodes): # bottom-up
            if n in active:
                for p in n.parents: active.add(p)

        # find nodes connected to query after pruning
        # prune nodes that are not connected to query in pruned network
        connected      = self.__connected_nodes(query_node,active)
        evidence      &= connected
        active        &= connected
        self._evidence = evidence
        self._pruned   = set(tbn.nodes) - active

        # exposed by context
        # nodes and testing_nodes are sorted: parents before children
        self.nodes         = tuple(n for n in tbn.nodes if n in active)
        self.testing_nodes = tuple(n for n in self.nodes if n.testing)
        self.live_count    = 0 # updated when pruning for selection
        assert not self.testing_nodes or len(self.nodes) >= 2
        
    # returns _active_ nodes connected to node (as a set)
    def __connected_nodes(self,node,active_nodes='all'):
        visited = set()
        def visit(n):
            if n in visited: return
            visited.add(n)
            for p in n.parents:  
                if active_nodes=='all' or p in active_nodes: visit(p)
            for c in n.children: 
                if active_nodes=='all' or c in active_nodes: visit(c)
        visit(node)     
        return visited
        
    """ public functions """
    
    def is_pruned(self,n):
        return n in self._pruned
        
    def has_evidence(self,n):
        return n in self._evidence


class for_nodes_posterior:

    def __init__(self,query_nodes,evidence_nodes,tbn):
        self._pruned   = None # set of pruned nodes
        self._evidence = None # set of unpruned evidence nodes
        
        # find nodes connected to query in network
        connected = set()
        for query_node in query_nodes:
            connected |= self.__connected_nodes(query_node)
        evidence  = set(evidence_nodes) & connected
        
        # prune (nodes not in active are pruned)
        # initialize active to query node and connected evidence nodes
        active = set(evidence)
        for query_node in query_nodes:
            active.add(query_node)

        # traverse bottom up, activating parents of active nodes
        for n in reversed(tbn.nodes): # bottom-up
            if n in active:
                for p in n.parents: active.add(p)

        # find nodes connected to query after pruning
        # prune nodes that are not connected to query in pruned network
        connected = set()
        for query_node in query_nodes:
            connected |= self.__connected_nodes(query_node,active)
        evidence      &= connected
        active        &= connected
        self._evidence = evidence
        self._pruned   = set(tbn.nodes) - active

        # exposed by context
        # nodes and testing_nodes are sorted: parents before children
        self.nodes         = tuple(n for n in tbn.nodes if n in active)
        self.testing_nodes = tuple(n for n in self.nodes if n.testing)
        self.live_count    = 0 # updated when pruning for selection
        assert not self.testing_nodes or len(self.nodes) >= 2
        
    # returns _active_ nodes connected to node (as a set)
    def __connected_nodes(self,node,active_nodes='all'):
        visited = set()
        def visit(n):
            if n in visited: return
            visited.add(n)
            for p in n.parents:  
                if active_nodes=='all' or p in active_nodes: visit(p)
            for c in n.children: 
                if active_nodes=='all' or c in active_nodes: visit(c)
        visit(node)     
        return visited
        
    """ public functions """
    
    def is_pruned(self,n):
        return n in self._pruned
        
    def has_evidence(self,n):
        return n in self._evidence


"""
Context for pruning tbn nodes when selecting a cpt for a testing node (T).

This context requires for_node_posterior context, and exposes the following:

    # pruned(): maps nodes to True/False
    # evidence(): maps nodes to True/False
    
Pruning is based on the following:

    # A testing node X is selected (its cpts has been selected) if
        # 'ancestral:'   X is an ancestor of T
        # 'predecessor:' X precedes T in the topological node order
    # Evidence at/below testing node X is ignored if X is not selected
    # We first identify ignored evidence and then prune as usual
"""

class for_selection:
    
    def __init__(self,testing_node,context,selection='predecessor'):
        assert testing_node.testing
        assert not context.is_pruned(testing_node)
        assert selection in ('ancestral','predecessor')
        
        self.context   = context
        self._pruned   = set()
        self._evidence = set()
        
        # identify testing nodes whose cpts have been selected
        selected = set()
        if selection == 'ancestral': self.__get_ancestors(testing_node,selected)
        else:                        self.__get_predecessors(testing_node,selected)
        
        # ignore unselected testing nodes and their descendants (including evidence)
        ignore = set()
        for n in context.nodes: # top-down
            if (n.testing and n not in selected) or \
               any(p in ignore for p in n.parents): ignore.add(n)
        assert testing_node in ignore
        
        # initialize flags
        for n in context.nodes:
            p = n in ignore or not context.has_evidence(n)
            e = n not in ignore and context.has_evidence(n)
            if p: self._pruned.add(n)
            if e: self._evidence.add(n)
        self._pruned.remove(testing_node)
        
        # prune
        for n in reversed(context.nodes): # bottom-up
            if n not in self._pruned: 
                for p in n.parents: self._pruned.discard(p)
        self._pruned |= context._pruned
        
        # updating context
        self.live = len(self._evidence) > 1 or testing_node not in self._evidence
        if self.live: context.live_count += 1
        
        # if single unpruned node, testing_node is root and dead
        # a testing node can become root when its parents have one possible instantiation
        # so they get pruned (edges from parents to testing_node removed)
        unpruned = tuple(n for n in context.nodes if n not in self._pruned)
        single_node = len(unpruned) == 1
        assert not single_node or not testing_node.parents 
        assert not single_node or not self.live
        
              
    # add to selected the testing ancestors of node
    def __get_ancestors(self,node,selected,ancestors=set()):
        for parent in node.parents:
            if parent not in ancestors:
                if parent.testing: selected.add(parent)
                ancestors.add(parent)
                __get_ancestors(parent,selected,ancestors)
            
    # add to selected the testing predecessors of node
    def __get_predecessors(self,node,selected):
        for n in self.context.nodes:
            if n == node: return
            elif n.testing: selected.add(n)
            
    """ public functions """
    
    def is_pruned(self,n):
        return n in self._pruned
        
    def has_evidence(self,n):
        return n in self._evidence
            