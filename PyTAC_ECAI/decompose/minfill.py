import time
import math
import heapq
from itertools import count

import utils.utils as u


""" returns a minfill elimination order """

# uses clique size of eliminated node as a tie-breaker
def elm_order(tbn):
    start_time = time.time()
    graph = Graph(tbn)
    order, cliques, width_l, width_s, size = graph.elm_order()
    elm_time = time.time()-start_time
    max_binary_rank = math.log2(width_s) # rank of max cluster assuming binary variables
    stats = (f'elm order: '
             f'cls ({max_binary_rank:.1f} {width_l}), '
             f'cls sum ({math.log2(size):.1f} {size:,}), '
             f'vars {len(tbn.nodes)}, time {elm_time:.3f} sec')
    return order, cliques, max_binary_rank, stats


""" node in undirected graph """

class Node:
    
    ID = count()
    
    def __init__(self,tbn_node):
        self.id           = next(Node.ID)
        self.tbn_node     = tbn_node
        self.card         = tbn_node.card
        self.parents      = set()
        self.neighbors    = set()
        self.fillin_count = None # number of edges to be added when eliminating node
        self.eliminated   = False
        
    # for sorting nodes
    def __lt__(self,other):
        return self.id < other.id
    
    # returns size of clique formed by node and its neighbors (factors cardinality)
    def clique_size(self):
        size = self.card
        for n in self.neighbors: size *= n.card
        return size
        
    # add edge self-n
    def add_edge(self,n):
        assert not self in n.neighbors and not n in self.neighbors
        self.neighbors.add(n)
        n.neighbors.add(self)
                    
    # disconnect node from its neighbors and update their fillin counts:
    #   for each self-n-m, with no edge self-m, decrement the fillin count of n
    # returns affected neighbors (whose fillin counts changed due to disconnect action)
    def disconnect(self):
        affected_nodes = set()
        for n in self.neighbors:
            n.neighbors.remove(self)
            if n.fillin_count > 0: # otherwise, every neighbor of n is connected to self
                affected_nodes.add(n)
                M = n.neighbors - self.neighbors
                # M contains neighbors of n that are not connected to self
                n.fillin_count -= len(M)                
        self.neighbors.clear()
        return affected_nodes 
    
    # pairwise connect node and parents
    def moralize(self,parents): # parents is a list
        parents1 = parents
        parents2 = parents.copy()
        for p1 in parents1:
            self.add_edge(p1)
            parents2.pop(0) # p1 is popped
            for p2 in parents2:
                assert p1 != p2
                if not p1 in p2.neighbors: # edge n1-n2 does not exist
                    p1.add_edge(p2)
    
    # returns score for eliminating nodes
    def elimination_score(self):
        return (self.fillin_count,self.clique_size())
        
    # sets fillin count of node (quadratic complexity in number of neighbors)
    # a careful implementation that reduces running time
    def set_fillin_count(self):
        self.fillin_count = 0
        N = set(self.neighbors) # copy set
        while N:
            n = N.pop()
            # N contains neighbors m != n of self for which we did not consider edge m-n
            M = N - n.neighbors
            # M contains neighbors m != n of self with no edge m-n (we did not consider m-n)
            self.fillin_count += len(M)

""" undirected graph """

class Graph:
    
    # construct an undirected graph representing the moral graph of tbn
    def __init__(self,tbn):
        self.nodes = [] # list of graph nodes (not set for deterministic behavior)
        self.count = None
        # add graph nodes and edges
        # tn: tbn node
        # gn: graph node
        dict = {} # maps tbn node to graph node
        for tn in tbn.nodes: # add graph nodes
            gn = Node(tn)
            self.nodes.append(gn)
            dict[tn] = gn
        self.count = len(self.nodes)    
        for tn in tbn.nodes: # add graph edges
            gn  = dict[tn]
            gps = [dict[tp] for tp in tn.parents]
            gn.moralize(gps)
            gn.parents = gps
        for gn in self.nodes: # set count of fillin edges
            gn.set_fillin_count()
        
    # pairwise connect neighbors of n and update fillin counts of affected nodes
    # for each added edge n1-n2: nodes n1, n2 and their common neighbors are affected
    # returns affected nodes
    def make_clique(self,n):
        affected_nodes = set() # nodes whose fillin counts may change
        N = set(n.neighbors)   # copy neighbors
        while N:
            n1 = N.pop()
            # N is neighbors n2 of self: n2 != n1 and the pair (n1, n2) have not been considered
            M = N - n1.neighbors
            # M is subset of N for which there is no edge n1-n2
            for n2 in M:
                # update minfill counts before adding edge (so .neighbors is correct)
                # n1 is acquiring a new neighbor n2, leading to new fillin edges between
                # n2 and old neighbors of n1, except ones that are now neighbors of n2
                # symmetric situation for n2
                n1.fillin_count += len(n1.neighbors - n2.neighbors)
                n2.fillin_count += len(n2.neighbors - n1.neighbors)
                # every common neighbors m of n1 and n2 will loose a fillin edge after
                # adding edge n1-n2
                common_neighbors = n1.neighbors & n2.neighbors
                for m in common_neighbors: m.fillin_count -= 1 # added n1-n2 for n1-m-n2
                # the minfill score of n1, n2 and their common neighbors change
                affected_nodes.update((n1,n2))
                affected_nodes.update(common_neighbors)
                n1.add_edge(n2)
        return affected_nodes
                
    # eliminates node from graph after pairwise connecting its neighbors
    def eliminate(self,n):
        assert not n.eliminated
        if n.fillin_count > 0: # otherwise, n.neighbors already a clique
            affected_nodes1 = self.make_clique(n)
            assert n in affected_nodes1
            affected_nodes1.remove(n)  # being eliminated
            assert n.fillin_count == 0  # validates update of fillin counts
        else:
            affected_nodes1 = set()
        affected_nodes2 = n.disconnect() # delete edges between n to its neighbors
        assert n not in affected_nodes2
        n.eliminated = True
        affected_nodes1.update(affected_nodes2)
        return affected_nodes1
        # n not removed from graph for efficiency reasons, but now disconnected from all
        
    def elm_order(self):
        pq      = PQ(self.nodes)
        pi      = []
        width_l = 0
        width_s = 0
        size    = 0
        cliques = []
        while True:
            n = pq.pop()
            if n is None: break # all nodes have been eliminated
            pi.append(n)
            clique  = set(i.tbn_node for i in n.neighbors)
            clique.add(n.tbn_node)
            cliques.append(clique)
            l       = 1+len(n.neighbors)
            s       = n.clique_size()
            width_l = max(width_l,l) # before eliminating node
            width_s = max(width_s,s) # before eliminating node
            size    += s             # before eliminating node
            affected_nodes = self.eliminate(n)    # set
            affected_nodes = list(affected_nodes) # list
            affected_nodes.sort()    # for deterministic behavior
            for n in affected_nodes: # n has new fillin count
                pq.add(n,replace=True) 
        assert pq.empty()
        assert(self.count == len(pi))
        pi = u.map('tbn_node',pi)
        return pi, cliques, width_l, width_s, size
        
        
""" priority queue that allows changing priorities """

class PQ:

    def __init__(self,nodes):
        self.pq       = [] # nodes in pq (valid and invalid)
        self.pq_dict  = {} # maps node to its valid entry in pq
        self.pq_count = count() # for breaking ties
        for n in nodes: self.add(n)       # fill priority queue
        
    # whether priority queue is empty
    def empty(self):
        return not self.pq and not self.pq_dict
        
    # adds node to priority queue
    # a node can be added multiple times (latest priority applies)
    def add(self,n,replace=False):
        assert not replace or n in self.pq_dict
        if n in self.pq_dict:  # n has a new priority
            self.invalidate(n) # invalidate current priority
        priority = n.elimination_score()
        count    = next(self.pq_count) # tie-breaker, needed for latest python
        entry    = [priority,count,n]
        self.pq_dict[n] = entry
        heapq.heappush(self.pq,entry)
        
    # invalidates current priority for n (called before n added with new priority)
    def invalidate(self,n):
        entry     = self.pq_dict.pop(n)
        assert not entry[-1] is 'overridden'
        entry[-1] = 'overridden'
        
    # returns node with smallest score from priority queue
    def pop(self):
        while self.pq:
            _, _, n = heapq.heappop(self.pq)
            if not n is 'overridden': # otherwise, ignore n
                del self.pq_dict[n]
                return n
        return None # no more elements in priority queue
        
