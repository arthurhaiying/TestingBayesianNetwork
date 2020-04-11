import signal
import subprocess
from time import sleep
from itertools import combinations

import decompose.minfill as minfill
import utils.utils as u

# returns an elimination order based on a given solver
def elm_order(tbn,solver,wait):
    # minfill and tamaki exact are not online so wait does not apply
    assert solver in ('minfill','flow cutter','tamaki exact','tamaki heuristic')
    if solver == 'minfill':
        return minfill.elm_order(tbn)
    graph = Graph(tbn) # moral graph
    return graph.elm_order(solver,wait)
    
"""
undirected graph according to the format of 
https://pacechallenge.wordpress.com/2016/12/01/announcing-pace-2017/
"""

# we use 'vertex' for integer representing vertex in graph
# we use 'node' for tbn node
# vertices start at 1
class Graph:

    # construct the moral graph of a tbn
    def __init__(self,tbn):
        self.vcount      = len(tbn.nodes) # number of vertices
        self.ecount      = None           # number of edges
        self.edges       = []             # edge is a pair (i,j)
        self.vertex2node = {v+1:n for v, n in enumerate(tbn.nodes)}
        
        node2vertex = {n:v+1 for v, n in enumerate(tbn.nodes)}
        for n in tbn.nodes:
            clique = tuple(node2vertex[m] for m in n.family)
            for edge in combinations(clique,2): 
                self.edges.append(edge)
        self.ecount = len(self.edges)
        # for deterministic behavior
        self.edges.sort()
        
    def __str__(self):
        edges = '\n'.join(f'{i} {j}' for i, j in self.edges)
        return f'p tw {self.vcount} {self.ecount}\n' + edges
        
    # map graph vertices to tbn nodes
    def vertices2nodes(self,vertices):
        return tuple(self.vertex2node[v] for v in vertices)
        
    # write graph to file
    def write(self,fname):
        with open(fname,'w') as f:
            f.write(f'c moral graph\n')
            f.write(f'p tw {self.vcount} {self.ecount}\n')
            for edge in self.edges:
                f.write(f'{edge[0]} {edge[1]}\n')
                
    # returns tree decomposition computed by flow cutter
    def elm_order(self,solver,wait):
        u.show(f'    calling {solver}...',end='')
        graph_fname = 'decompose/tmp/graph.gr'
        tree_fname  = 'decompose/tmp/tree.td'
        if solver == 'flow cutter':
            program = 'flow_cutter_pace17'
            cmd     = [f'./decompose/solvers/{program}']
            online  = True
        elif solver == 'tamaki heuristic':
            program = 'tamaki/tw-heuristic'
            cmd     = [f'./decompose/solvers/{program}']
            online  = True
        elif solver == 'tamaki exact':
            program = 'tamaki/tw-exact'
            cmd     = [f'./decompose/solvers/{program}']
            online  = False
        # write graph to file
        self.write(graph_fname)
        # call tree decomposition program
        with open(f'{graph_fname}',"r") as input, open(f'{tree_fname}',"w") as output:
            process = subprocess.Popen(cmd,stdin=input,stdout=output)
            if online:
                u.show(f'waiting {wait} sec...',end='',flush=True)
                sleep(wait)
                process.send_signal(signal.SIGTERM)
            else:
                process.wait() # blocks python until process returns
        code     = process.returncode
        _, error = process.communicate()
        process.kill()
        u.check(code != 0,
            f'failed to execute {solver} because\n  {error}',
            f'using treewidth solver')
        u.show('done')
        # read decomposition tree from file
        tree = TreeD(tree_fname)
        # convert decomposition tree to elimination order (vertices)
        vertex_order = tree.elm_order()
        # return elimination order of tbn nodes
        stats = f'elm order: cls max {tree.width}'
        return self.vertices2nodes(vertex_order), tree.width, stats
        
"""
tree decomposition according to the format of 
https://pacechallenge.wordpress.com/2016/12/01/announcing-pace-2017/
"""    
        
# bags are indexed starting from 1
class TreeD:

    # read tree decomposition from file
    def __init__(self,fname):
        self.bcount    = None # number of bags (clusters)
        self.vcount    = None # number of vertices in underlying graph
        self.width     = None # size of largest bag (cluster): treewidth+1
        self.index2bag = {}   # maps bag index to tuple of graph vertices
        self.edges     = []   # edge is pair (i,j) of bag indices
        
        def getline(f):
            line = f.readline()
            line = line.strip('\n')
            return line.split(' ')
        
        is_comment = lambda line: line[0]=='c'
        is_bag     = lambda line: line[0]=='b'
        
        with open(fname,'r') as f:
            line = getline(f)
            while is_comment(line): 
                line = getline(f)
            assert line[0]=='s' and line[1]=='td'
            self.bcount = int(line[2])
            self.width  = int(line[3])
            self.vcount = int(line[4])

            count = 2*self.bcount-1 # number of bags + number of edges
            while count > 0:
                line = getline(f)
                if is_comment(line): continue
                count -= 1
                if is_bag(line):
                    index = int(line[1])
                    bag   = set(int(s) for s in line[2:])
                    self.index2bag[index] = bag
                else:
                    edge = (int(line[0]),int(line[1]))
                    self.edges.append(edge)
                    
        # for deterministic behavior
        self.edges.sort()
    
    def __str__(self):
        bag2str = lambda bag: ' '.join([str(v) for v in bag])
        bags  = '\n'.join(f'b {i} {bag2str(bag)}' for i, bag in self.index2bag.items())
        edges = '\n'.join(f'{i} {j}' for i, j in self.edges)
        return f's td {self.bcount} {self.width} {self.vcount}\n' + bags + '\n' + edges
        
    # extract elimination order from tree decomposition                
    def elm_order(self):
        neighbors = {i+1:[] for i in range(self.bcount)}
        for i, j in self.edges:
            neighbors[i].append(j)
            neighbors[j].append(i)  
            
        # for deterministic behavior
        def eliminate(vars1,vars2):
            vars = list(vars1-vars2)
            vars.sort()
            return vars
            
        order = []       
        def message(i,j): # i to j
            for k in neighbors[i]:
                if k != j: message(k,i)
            cls_i = self.index2bag[i]
            cls_j = self.index2bag[j]
            order.extend(eliminate(cls_i,cls_j))
            
        r = 1 # root
        for i in neighbors[r]:
            message(i,r)
        eliminated = set(order)
        cls_r      = self.index2bag[r]
        order.extend(eliminate(cls_r,eliminated))
                
        assert len(order) == self.vcount
        return order
               