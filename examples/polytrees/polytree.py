from graphviz import Digraph
import numpy as np
import random
import re

from pathlib import Path
import os,sys

if __name__ == '__main__':
    #basepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    basepath = Path(__file__).resolve().parents[2]
    #print("basepath: %s" %basepath)
    sys.path.append(str(basepath))

from tbn.tbn import TBN
from tbn.node import Node
import utils.VE as VE

# experiment parameter settings
NUM_NODES = 50
NUM_ITERS = 500
MAX_NUM_ABSTRACTED_NODES = 5
MIN_NUM_EVIDENCE_NODES = 5
MAX_NUM_EVIDENCE_NODES = 10
MIN_CARDINALITY = 2
MAX_CARDINALITY = 8

MIN_RANK = 3
MAX_RANK = 10
MIN_NODES_PER_RANK = 2
MAX_NODES_PER_RANK = 5
MAX_IN_DEGREE = 3
MAX_OUT_DEGREE = 3
EDGE_PROB = 0.2
MIN_NUM_NODES = 20


# make a random cpt for a node with cardinality card 
# and whose parants have cardinalities cards
def random_cpt(card,cards):
    alpha = np.ones(card) # uniform
    if not cards:
        # no parents
        return np.random.dirichlet(alpha)
    else:
        return np.random.dirichlet(alpha, tuple(cards))

# generate a random polytree of n_nodes nodes
# run n_iters iterations
# return a list of list representing the adjacency list of the generated polytree
def get_random_polytree(n_nodes, n_iters):

    dag = [[i+1] if i < n_nodes-1 else [] for i in range(n_nodes)]
    # initialize a simple ordered tree
    adj_list = []
    for i in range(n_nodes):
        if i == 0:
            adj_list.append([i+1])
        elif i == n_nodes-1:
            adj_list.append([i-1])
        else:
            adj_list.append([i-1,i+1])

    # add/remove/inverse edge to dag
    for _ in range(n_iters):
        (node_u, node_v) = np.random.choice(n_nodes,size=2,replace=False) 
        #if node_v in dag[node_u]:
            #continue
        node_w = path(adj_list,node_u,node_v)[-2] # predecessor w of v
        # remove the edge between v and w
        if node_v in dag[node_w]:
            dag[node_w].remove(node_v)
        else:
            dag[node_v].remove(node_w)
        adj_list[node_w].remove(node_v)
        adj_list[node_v].remove(node_w)
        # add edge between u and v
        p = np.random.uniform(low=0,high=1)
        if p > 0.5:
            dag[node_u].append(node_v)
            # for half probability, connect u to v
        else:
            dag[node_v].append(node_u)
        adj_list[node_u].append(node_v)
        adj_list[node_v].append(node_u)

    return dag


def get_random_DAG():
    # repeat 
    while True:
        rank = np.random.randint(low=MIN_RANK,high=MAX_RANK+1)
        n_nodes = 0
        dag = {}
        in_degrees = {}
        for _ in range(rank):
            # for each rank
            nodes_per_rank = np.random.randint(low=MIN_NODES_PER_RANK,high=MAX_NODES_PER_RANK)
            for node_r in range(n_nodes,n_nodes+nodes_per_rank):
                dag[node_r] = []
                in_degrees[node_r] = 0
                # create new nodes at rank r
            for node in range(n_nodes):
                # for each older node
                for node_r in range(n_nodes,n_nodes+nodes_per_rank):
                    # for each new node
                    p = np.random.uniform()
                    if p <= EDGE_PROB:
                        if in_degrees[node_r] < MAX_IN_DEGREE and len(dag[node]) < MAX_OUT_DEGREE:
                            # if satisfy degrees requirement
                            dag[node].append(node_r)
                            in_degrees[node_r]+=1
                        # add a new edge from old node to new node with prob
            n_nodes += nodes_per_rank

        dag_list = [None]*n_nodes
        for node,children in dag.items():
            dag_list[node] = children
        dag = dag_list
        # check connectivity
        flag = False
        conn_nodes = None
        for node_x in range(n_nodes):
            conn_nodes = connected_nodes(node_x,dag)
            if len(conn_nodes) >= MIN_NUM_NODES:
                # if enough nodes are connected, success
                flag = True
                break

        if flag:
            conn_nodes.sort()
            conn_nodes_dict = {node:i for i,node in enumerate(conn_nodes)}
            # new id for each node
            dag2 = [None]*len(conn_nodes)
            for node,children in enumerate(dag):
                if node in conn_nodes:
                    # if connected
                    node2 = conn_nodes_dict[node]
                    children2 = [conn_nodes_dict[child] for child in children]
                    dag2[node2] = children2
            return dag2



def connected_nodes(node,dag):
    adj_list = get_adjacency_list(dag)
    visited_nodes = []
    queue = [node]
    visited_nodes.append(node)
    while queue:
        curr = queue.pop(0)
        for neighbor in adj_list[curr]:
            if neighbor not in visited_nodes:
                visited_nodes.append(neighbor)
                queue.append(neighbor)
    return visited_nodes






    
def dot(dag, fname="bn.gv"):
    d = Digraph()
    d.attr(rankdir='TD')
    for node in range(len(dag)):
        d.node('v'+str(node))
    for node in range(len(dag)):
        for child in dag[node]:
            d.edge('v'+str(node),'v'+str(child))

    d.render(fname, view=False)

# return the shortest path between src and dst in a graph
# adj_list: a list of list representing the adjacency list of graoh
def path(adj_list, src, dst):
    prev_list = [None for _ in range(len(adj_list))]
    is_node_visited = [False for _ in range(len(adj_list))]
    # mark all nodes as unvisited
    queue = [src]
    is_node_visited[src] = True
    prev_list[src] = src
    while queue:
        curr = queue.pop(0)
        if curr == dst:
            break
            # find dst
        for neighbor in adj_list[curr]:
            # for each neighbor
            if not is_node_visited[neighbor]:
                # if not visited 
                queue.append(neighbor)
                is_node_visited[neighbor] = True
                prev_list[neighbor] = curr

    path = []
    prev = dst
    while prev != src:
        path.append(prev)
        prev = prev_list[prev]
    path.append(src)
    # retrieve the path
    return path[::-1]

# return a topological ordering of nodes in dag
# dag: a list of list representing directed adjacencies of DAG
def topo_sort(dag):
    n_nodes = len(dag)
    in_degrees = [0]*n_nodes

    # compute in-degrees of each node in dag
    for node in range(n_nodes):
        for child in dag[node]:
            in_degrees[child]+=1
    
    queue = [node for node in range(n_nodes) if in_degrees[node] == 0] # roots
    order = []
    while queue:
        curr = queue.pop(0)
        order.append(curr)
        for child in dag[curr]:
            in_degrees[child]-=1
            # decrement in-degrees for each child
            if in_degrees[child] == 0:
                queue.append(child)

    return order

def get_parent_list(dag):
    # get parents of each node
    n_nodes = len(dag)
    parent_list = [[] for _ in range(n_nodes)]
    for node in range(n_nodes):
        for child in dag[node]:
            parent_list[child].append(node)
    return parent_list

# randomly generate a true BN of dag structure
# return a tbn object
def sample_random_BN(dag):
    # sample polytree DAG
    n_nodes = len(dag)
    parent_list = get_parent_list(dag)
    order = topo_sort(dag) 
    # return a topological ordering of nodes
    cards = [np.random.randint(low=MIN_CARDINALITY,high=MAX_CARDINALITY+1) for _ in range(n_nodes)]
    # random cardinality for each node 
    bnNode_cache = {}
    bn = TBN("polytree")
    # print("Start adding nodes...")
    for node in order:
        name = 'v'+str(node)
        card = cards[node]
        values = ['state'+str(i) for i in range(card)]
        parentNodes = [bnNode_cache[p] for p in parent_list[node]]
        pcards = [cards[p] for p in parent_list[node]]
        cpt = random_cpt(card, pcards)
        bnNode = Node(name,values=values,parents=parentNodes,testing=False,cpt=cpt)
        bnNode_cache[node] = bnNode
        bn.add(bnNode)
        # add bn node to bn

    return bn

def ancestors(node_x, dag):
    # get ancestors of node x in dag
    parent_list = get_parent_list(dag)
    ancestors = []
    queue = parent_list[node_x]
    while queue:
        curr = queue.pop(0)
        if curr not in ancestors:
            ancestors.append(curr)
            for p in parent_list[curr]:
                queue.append(p)
    return ancestors


# make a random query over BN
# dag - a list of list representing the adjacencies in a BN
# returns a query (Q,E,X) where Q is a query node, E is a set of evidence nodes, X is a set of abstracted nodes
def random_query(dag):
    n_nodes = len(dag)
    # choose a query node from leaf nodes
    leaves = [node for node in range(n_nodes) if len(dag[node]) == 0]
    node_q = np.random.choice(leaves)

    # ancestors_q of node q
    ancestors_q = ancestors(node_q,dag)
    # choose abstracted nodes from ancestors_q of node q
    ancestors_q = list(ancestors_q)
    num_abs_nodes = min(len(ancestors_q), MAX_NUM_ABSTRACTED_NODES)
    nodes_abs = np.random.choice(ancestors_q,size=num_abs_nodes,replace=False)

    # choose evidence nodes from the remaining nodes in dag
    nodes_remaining = [node for node in range(n_nodes) if node != node_q and 
        node not in nodes_abs]
    num_evid_nodes = np.random.randint(low=MIN_NUM_EVIDENCE_NODES,
        high=min(len(nodes_remaining), MAX_NUM_EVIDENCE_NODES)+1)
    #num_evid_nodes = min(len(nodes_remaining), MAX_NUM_EVIDENCE_NODES)
    nodes_evid = np.random.choice(nodes_remaining,size=num_evid_nodes,replace=False)

    return node_q, nodes_evid, nodes_abs


# make a random query over BN
# dag - a list of list representing the adjacencies in a BN
# returns a query (Q,E,X) where Q is a query node, E is a set of evidence nodes, X is a set of abstracted nodes
def random_query_with_ancestral_evidences(dag):
    n_nodes = len(dag)
    # choose a query node from leaf nodes
    leaves = [node for node in range(n_nodes) if len(dag[node]) == 0]
    random.shuffle(leaves)
    for node_q in leaves:
        # ancestors_q of node q
        ancestors_q = ancestors(node_q,dag)
        # choose abstracted nodes from ancestors_q of node q
        num_abs_nodes = min(len(ancestors_q), MAX_NUM_ABSTRACTED_NODES)
        #nodes_abs = np.random.choice(ancestors_q,size=num_abs_nodes,replace=False)
        nodes_abs = ancestors_q[:num_abs_nodes] # get first num_abs_nodes ancestors

        # choose evidence nodes from the ancestors of abstracted nodes
        ancestors_acc = set()
        for node_x in nodes_abs:
            ancestors_x = ancestors(node_x,dag)
            ancestors_acc = ancestors_acc.union(ancestors_x)
        # notice that evidences cannot be abstracted nodes
        ancestors_acc = list(ancestors_acc.difference(nodes_abs))
        if len(ancestors_acc) < MIN_NUM_EVIDENCE_NODES:
            continue
        num_evid_nodes = np.random.randint(low=MIN_NUM_EVIDENCE_NODES,
            high=min(len(ancestors_acc), MAX_NUM_EVIDENCE_NODES)+1)
        #num_evid_nodes = min(len(nodes_remaining), MAX_NUM_EVIDENCE_NODES)
        nodes_evid = np.random.choice(ancestors_acc,size=num_evid_nodes,replace=False)
        return node_q, nodes_evid, nodes_abs

    raise RuntimeError("No query with enough evidences")


def get_adjacency_list(dag):
    parent_list = get_parent_list(dag)
    adj_list = [parents+children for parents,children in zip(parent_list,dag)]
    return adj_list

# return evidences at or above node x in polytree dag
# dag: a list of list representing directed adjacency in polytree dag
# evidences: a set of evidences in dag
# return: a subset of evidences that are above node_x
def evidences_at_or_above(node_x,evidences,dag):
    n_nodes = len(dag)
    evid = []
    adj_list = get_adjacency_list(dag)
    is_node_visited = [False]*n_nodes
    if node_x in evidences:
        # if evidence at x
        evid.append(node_x)
    is_node_visited[node_x] = True

    queue = []
    # start with parents of x
    for node in range(n_nodes):
        if node_x in dag[node]:
            queue.append(node)
            is_node_visited[node] = True   
    while queue:
        curr = queue.pop(0)
        if curr in evidences:
            evid.append(curr)
        for neighbor in adj_list[curr]:
            if not is_node_visited[neighbor]:
                queue.append(neighbor)
                is_node_visited[neighbor] = True

    return evid




# return evidences in the y side of the edge (x, y) in polytree dag
# dag: a list of list representing directed adjacency in polytree dag
# evidences: a set of evidences in dag
# return: a subset of evidences that are in the side of y in dag
def evidences_in_the_lower_side_of(node_x,node_y,evidences,dag):
    n_nodes = len(dag)
    adj_list = get_adjacency_list(dag)

    evid = []
    is_node_visited = [False]*n_nodes
    is_node_visited[node_x] = True
    queue = [node_y]
    is_node_visited[node_y] = True
    # start with node y but cannot visit node x
    while queue:
        curr = queue.pop(0)
        if curr in evidences:
            evid.append(curr)
        for neighbor in adj_list[curr]:
            if not is_node_visited[neighbor]:
                queue.append(neighbor)
                is_node_visited[neighbor] = True

    return evid


# return evidences in the x side of the edge (x, y) in polytree dag
# dag: a list of list representing directed adjacency in polytree dag
# evidences: a set of evidences in dag
# return: a subset of evidences that are in the side of y in dag
def evidences_in_the_upper_side_of(node_x,node_y,evidences,dag):
    n_nodes = len(dag)
    adj_list = get_adjacency_list(dag)
    evid = []
    is_node_visited = [False]*n_nodes
    is_node_visited[node_y] = True
    queue = [node_x]
    is_node_visited[node_x] = True
    # start with node y but cannot visit node x
    while queue:
        curr = queue.pop(0)
        if curr in evidences:
            evid.append(curr)
        for neighbor in adj_list[curr]:
            if not is_node_visited[neighbor]:
                queue.append(neighbor)
                is_node_visited[neighbor] = True

    return evid


# allocate the set of ancestral evidences for each child of an abstracted node with reference to Q in dag
# dag: a list of list representing directed adjacency in polytree dag
# (q,e,x): query node/evidence nodes/abstracted nodes
# return a dict of list representing the alive evidence for each child of abstracted node
def alloc_ancestral_evidences(node_q,nodes_evid,nodes_abs,dag):
    ancestor_evid_for_children_dict = {}
    for node_x in nodes_abs:
        for node_y in dag[node_x]:
            # for each child of abstracted node
            if node_y not in ancestor_evid_for_children_dict:
                ancestors_x = ancestors(node_x,dag)
                ancestor_evid = [evid for evid in nodes_evid if evid in ancestors_x]
                ancestor_evid_for_children_dict[node_y] = ancestor_evid
    return ancestor_evid_for_children_dict



# allocate the set of alive evidences for each child of an abstracted node with reference to Q in dag
# dag: a list of list representing directed adjacency in polytree dag
# (q,e,x): query node/evidence nodes/abstracted nodes
# return a dict of list representing the alive evidence for each child of abstracted node
def alloc_alive_evidences(node_q,nodes_evid,nodes_abs,dag):
    parent_list = get_parent_list(dag)
    ancestors_q = list(ancestors(node_q,dag))
    ancestors_q.append(node_q) # be careful about a query node as child of abstracted nodes
    alive_evid_for_children_dict = {}
    children_of_higher_degree = [] # children of abstracted nodes that have multiple abstracted parents
    # get the number of abstracted parents of node
    def degree(node):
        count = 0
        for p in parent_list[node]:
            if p in nodes_abs:
                count+=1
        return count

    for node_x in nodes_abs:
        children = []
        child_q = None
        for child in dag[node_x]:
            if child not in ancestors_q:
                children.append(child)
            else:
                child_q = child
        # child q might have multiple abstracted parents
        assert child_q is not None
        if degree(child_q) == 1:
            children.append(child_q)
        else:
            children_of_higher_degree.append(child_q)
        # allocate evidence for children of degree one
        alive_evid_acc = evidences_at_or_above(node_x,nodes_evid,dag)
        for i,child in enumerate(children):
            if i >= 1:
                alive_evid_acc += evidences_in_the_lower_side_of(node_x,children[i-1],nodes_evid,dag)
                # alive evidence for child i is evidence above x and evidence below all children 0:i-1
            alive_evid_for_children_dict[child] = alive_evid_acc.copy() # need to copy the list

    # allocate evidences for children of multiple degree
    for child in children_of_higher_degree:
        parents_abs = [p for p in parent_list[child] if p in nodes_abs]
        alive_evid_acc = []
        for parent in parents_abs:
            alive_evid_acc += evidences_in_the_upper_side_of(parent,child,nodes_evid,dag)
        alive_evid_for_children_dict[child] = alive_evid_acc.copy()
    
    return alive_evid_for_children_dict


def get_states_map(card, scard):
    states_map = [[] for _ in range(scard)]
    for i in range(card):
        states_map[i % scard].append(i)
    return states_map

def node_id(node):
    return int(re.sub('\D','',node.name))


def get_states_map_dict(bn,nodes_abs):
    states_map_dict = {}
    for node_x in nodes_abs:
        name = 'v'+str(node_x)
        bnNode = bn.node(name)
        card = len(bnNode.values)
        scard = (card+1)//2
        states_map = get_states_map(card,scard)
        states_map_dict[node_x] = states_map
    return states_map_dict
        


# create an abstract BN in which nodes in nodes_abs have half cardinality 
def make_abstract_BN(bn,nodes_abs,states_map_dict):
    abstract_bn = TBN("abstract bn")
    abs_bnNodes_cache = {}
    for bnNode in bn._add_order:
        # visit node in topo order
        id = node_id(bnNode)
        if id not in nodes_abs:
            name,values,parents,cpt = bnNode.name,bnNode.values,bnNode.parents,bnNode.cpt
            card = len(values)
            pids = [node_id(pNode) for pNode in parents]
            sparents = [abs_bnNodes_cache[pid] for pid in pids]
            spcards = [len(sparent.values) for sparent in sparents]
            abs_bnNode = Node(name,values=values,parents=sparents,testing=False,cpt=np.zeros(tuple(spcards)+(card,)))
            abs_bnNodes_cache[id] = abs_bnNode
            abstract_bn.add(abs_bnNode)
            # copy this node
        else:
            # for abstracted node
            name = bnNode.name
            states_map = states_map_dict[id]
            scard = len(states_map)
            svalues = ['states ' + ','.join(map(str,states)) for states in states_map]
            pids = [node_id(pNode) for pNode in bnNode.parents]
            sparents = [abs_bnNodes_cache[pid] for pid in pids]
            spcards = [len(sparent.values) for sparent in sparents]
            abs_bnNode = Node(name,values=svalues,parents=sparents,testing=False,cpt=np.zeros(tuple(spcards)+(scard,)))
            abs_bnNodes_cache[id] = abs_bnNode
            abstract_bn.add(abs_bnNode)
            
    return abstract_bn


import itertools as iter

def one_hot(index,card):
    arr = np.zeros(card)
    arr[index] = 1.0
    return arr

# reparametrize the incomplete BN from BN based on given evidence
# bn, abstract_bn - tbn objects over the given polytree dag
# node_evid, node_abs - evidence nodes and abstraccted nodes
# states_map：a list of list representing the original states within superstate
# states_map_dict - a dict of states_map for each abstracted node
# alive_evid_dict: a dict of alive evidences for each abstracted node
# evidence - a dict of hard evidence 
def reparam(bn,abstract_bn,nodes_abs,states_map_dict,alive_evid_dict,evidence_values):
    n_nodes = len(bn.nodes)
    nodes_children_of_abs = alive_evid_dict.keys()
    print("Start reparam bn...")
    nodes_reparam = []
    # case 1: abstracted node that is not children of another abstracted node
    # pr(x'|u) = sum pr(x|u)
    for node_x in nodes_abs:
        if node_x not in nodes_children_of_abs:
            name = 'v'+str(node_x)
            bnNode = bn.node(name)
            card = len(bnNode.values)
            scard = len(states_map_dict[node_x])
            pcards = [len(pNode.values) for pNode in bnNode.parents]
            cpt = bnNode.cpt
            assert np.array(cpt).shape == tuple(pcards)+(card,)
            # sum along last dimension of original cpt
            arrays = []
            for states in states_map_dict[node_x]:
                array = np.sum(cpt[...,np.array(states)],axis=-1,keepdims=True)
                arrays.append(array)
            # update cpt for abstracted nodes
            super_cpt = np.concatenate(arrays,axis=-1)
            assert super_cpt.shape == tuple(pcards)+(scard,)
            abs_bnNode = abstract_bn.node(name)
            abs_bnNode._cpt = super_cpt
            nodes_reparam.append(node_x)

    # case 2: for children of abstracted node
    for node_y in nodes_children_of_abs:
        print("reparem child %s..." %(str(node_y)))
        name = 'v'+str(node_y)
        bnNode = bn.node(name)
        card = len(bnNode.values)
        pids = [node_id(pNode) for pNode in bnNode.parents]
        pcards = [len(pNode.values) for pNode in bnNode.parents]
        #print("card: %s" % card)
        #print("parents: %s" %(pids))
        #print("parents card: %s" %(pcards))
        cpt = bnNode.cpt
        assert np.array(cpt).shape == tuple(pcards)+(card,)

        # get abstracted parents and non-abstracted parents
        super_pcards = []
        for pid in pids:
            if pid not in nodes_abs:
                # non-abstracted parents
                pname = 'v'+str(pid)
                pcard = len(bn.node(pname).values)
                super_pcards.append(pcard)
            else:
                #print("find abstracted parent")
                # abstracted parents
                pcard = len(states_map_dict[pid])
                super_pcards.append(pcard)
        #print("parents super card %s" %(super_pcards))
        # get value sets for each parent in abstracted bn
        super_parent_value_sets = []
        for pid in pids:
            if pid not in nodes_abs:
                # non abstracted parents
                pname = 'v'+str(pid)
                pcard = len(bn.node(pname).values)
                value_set = [[i] for i in range(pcard)]
                # each superstate contains one original state
                super_parent_value_sets.append(value_set)        
            else:
                # abstracted parents
                value_set = states_map_dict[pid]
                super_parent_value_sets.append(value_set)

        #print("super parent value sets: %s" %(super_parent_value_sets))
        # for each super parent instantiation 
        eids = alive_evid_dict[node_y]
        ecards = []
        for eid in eids:
            ename = 'v'+str(eid)
            eNode = bn.node(ename)
            ecard = len(eNode.values)
            ecards.append(ecard)
            # cardinality of evidences
        #print("evidences: %s" % eids)
        #print("evidences card: %s" %(ecards))
        evalues = [evidence_values[eid] for eid in eids]
        evid_lambdas = []
        for ecard,evalue in zip(ecards,evalues):
            evid_lambda = one_hot(evalue,ecard)
            evid_lambdas.append(evid_lambda)
            # one hot lambdas for evidences
        total_cards = pcards + ecards
        total_input_lambdas = []
        for super_parent_values in iter.product(*super_parent_value_sets):
            #print("parent super values: %s" % (super_parent_values,))
            # for each super parent instantiation
            parent_lambdas = []
            for pcard,pvalue in zip(pcards,super_parent_values):
                array = np.zeros(pcard)
                array[np.array(pvalue)] = 1.0/len(pvalue)
                parent_lambdas.append(array)
                # soft value for each parent
            input_lambdas = parent_lambdas + evid_lambdas
            total_input_lambdas.append(input_lambdas)

        # convert lambdas to AC inputs
        total_input_size = 1
        for pcard in super_pcards:
            total_input_size*=pcard
        total_input_lambdas = list(zip(*total_input_lambdas))
        total_input_lambdas = [np.array(lambdas) for lambdas in total_input_lambdas]
        #print("num_input: %s num_input_of_lambdas: %s" %(len(pids + eids), len(total_input_lambdas)))
        assert len(total_input_lambdas) == len(pids + eids)
        for tcard,lambdas in zip(total_cards, total_input_lambdas):
            #print("card: %s total input size: %s lambdas: %s" %(tcard,total_input_size,lambdas.shape))
            assert lambdas.shape == (total_input_size, tcard)

        # computer posterior probability pr(y|x'v,e)
        input_names = []
        for pid in pids:
            input_names.append('v'+str(pid))
        for eid in eids:
            input_names.append('v'+str(eid))
        output_name = 'v'+str(node_y)
        if node_y not in nodes_abs:
            # children of abstracted itself is not an abstracted node 
            super_cpt = VE.posteriors(bn,inputs=input_names,output=output_name,evidence=total_input_lambdas)
            #print("super cpt: %s" % (super_cpt.shape,))
            shape=tuple(super_pcards)+(card,)
            super_cpt = super_cpt.reshape(shape)
            abs_bnNode = abstract_bn.node(name)
            abs_bnNode._cpt = super_cpt
        else:
            # special case that children of abstracted node itself is also an abstracted node
            scard = states_map_dict[node_y]
            super_cpt = VE.posteriors(bn,inputs=input_names,output=output_name,evidence=total_input_lambdas)
            #print("super cpt: %s" % (super_cpt.shape,))
            shape=tuple(super_pcards)+(card,)
            super_cpt = super_cpt.reshape(shape)
            arrays = []
            for states in states_map_dict[node_y]:
                array = np.sum(super_cpt[...,np.array(states)],axis=-1,keepdims=True)
                arrays.append(array)
            super_cpt = np.concatenate(arrays,axis=-1)
            abs_bnNode = abstract_bn.node(name)
            abs_bnNode._cpt = super_cpt

        nodes_reparam.append(node_y)
        print("Finish reparam child %s" %str(node_y))

    # case 3: for the remaining nodes, copy cpt from bn
    for node_i in range(n_nodes):
        if node_i not in nodes_abs and node_i not in nodes_children_of_abs:
            name = 'v'+str(node_i)
            bnNode = bn.node(name)
            cpt = bnNode.cpt
            abs_bnNode = abstract_bn.node(name)
            abs_bnNode._cpt = np.copy(cpt)
            nodes_reparam.append(node_i)

    print("num nodes: %d nodes processed:%s" %(n_nodes, nodes_reparam))
    assert len(nodes_reparam) == n_nodes 
    print("Finish reparam bn.")

            
            

# reparametrize the incomplete BN from BN based on given evidence
# bn, abstract_bn - tbn objects over the given polytree dag
# node_evid, node_abs - evidence nodes and abstraccted nodes
# states_map：a list of list representing the original states within superstate
# states_map_dict - a dict of states_map for each abstracted node
# alive_evid_dict: a dict of alive evidences for each abstracted node
# evidence - a dict of hard evidence 
# be careful about evid in non-abstracted parents
def reparam(bn,abstract_bn,nodes_abs,states_map_dict,alive_evid_dict,evidence_values):
    n_nodes = len(bn.nodes)
    nodes_children_of_abs = alive_evid_dict.keys()
    print("Start reparam bn...")
    nodes_reparam = []
    # case 1: abstracted node that is not children of another abstracted node
    # pr(x'|u) = sum pr(x|u)
    for node_x in nodes_abs:
        if node_x not in nodes_children_of_abs:
            name = 'v'+str(node_x)
            bnNode = bn.node(name)
            card = len(bnNode.values)
            scard = len(states_map_dict[node_x])
            pcards = [len(pNode.values) for pNode in bnNode.parents]
            cpt = bnNode.cpt
            assert np.array(cpt).shape == tuple(pcards)+(card,)
            # sum along last dimension of original cpt
            arrays = []
            for states in states_map_dict[node_x]:
                array = np.sum(cpt[...,np.array(states)],axis=-1,keepdims=True)
                arrays.append(array)
            # update cpt for abstracted nodes
            super_cpt = np.concatenate(arrays,axis=-1)
            assert super_cpt.shape == tuple(pcards)+(scard,)
            abs_bnNode = abstract_bn.node(name)
            abs_bnNode._cpt = super_cpt
            nodes_reparam.append(node_x)

    # case 2: for children of abstracted node
    for node_y in nodes_children_of_abs:
        print("reparem child %s..." %(str(node_y)))
        name = 'v'+str(node_y)
        bnNode = bn.node(name)
        card = len(bnNode.values)
        pids = [node_id(pNode) for pNode in bnNode.parents]
        pcards = [len(pNode.values) for pNode in bnNode.parents]
        #print("card: %s" % card)
        #print("parents: %s" %(pids))
        #print("parents card: %s" %(pcards))
        cpt = bnNode.cpt
        assert np.array(cpt).shape == tuple(pcards)+(card,)

        # get abstracted parents and non-abstracted parents
        super_pcards = []
        for pid in pids:
            if pid not in nodes_abs:
                # non-abstracted parents
                pname = 'v'+str(pid)
                pcard = len(bn.node(pname).values)
                super_pcards.append(pcard)
            else:
                #print("find abstracted parent")
                # abstracted parents
                pcard = len(states_map_dict[pid])
                super_pcards.append(pcard)
        #print("parents super card %s" %(super_pcards))
        # get value sets for each parent in abstracted bn
        super_parent_value_sets = []
        for pid in pids:
            if pid not in nodes_abs:
                # non abstracted parents
                pname = 'v'+str(pid)
                pcard = len(bn.node(pname).values)
                if pid in alive_evid_dict[node_y]:
                    # if alive evidence on this parent
                    value_set = [[evidence_values[pid]] for i in range(pcard)]
                else:
                    value_set = [[i] for i in range(pcard)]
                # each superstate contains one original state
                super_parent_value_sets.append(value_set)        
            else:
                # abstracted parents
                value_set = states_map_dict[pid]
                super_parent_value_sets.append(value_set)

        #print("super parent value sets: %s" %(super_parent_value_sets))
        # for each super parent instantiation 
        eids = []
        ecards, evalues = [],[]
        for eid in alive_evid_dict[node_y]:
            if eid in pids:
                assert eid not in nodes_abs
                print("evidence on non abstracted parents!")
            else:
                ename = 'v'+str(eid)
                eNode = bn.node(ename)
                ecard = len(eNode.values)
                eids.append(eid)
                ecards.append(ecard)
                evalues.append(evidence_values[eid])
                # cardinality of evidences
        #print("evidences: %s" % eids)
        #print("evidences card: %s" %(ecards))
        evid_lambdas = []
        for ecard,evalue in zip(ecards,evalues):
            evid_lambda = one_hot(evalue,ecard)
            evid_lambdas.append(evid_lambda)
            # one hot lambdas for evidences
        total_cards = pcards + ecards
        total_input_lambdas = []
        for super_parent_values in iter.product(*super_parent_value_sets):
            #print("parent super values: %s" % (super_parent_values,))
            # for each super parent instantiation
            parent_lambdas = []
            for pcard,pvalue in zip(pcards,super_parent_values):
                array = np.zeros(pcard)
                array[np.array(pvalue)] = 1.0/len(pvalue)
                parent_lambdas.append(array)
                # soft value for each parent
            input_lambdas = parent_lambdas + evid_lambdas
            total_input_lambdas.append(input_lambdas)

        # convert lambdas to AC inputs
        total_input_size = 1
        for pcard in super_pcards:
            total_input_size*=pcard
        total_input_lambdas = list(zip(*total_input_lambdas))
        total_input_lambdas = [np.array(lambdas) for lambdas in total_input_lambdas]
        #print("num_input: %s num_input_of_lambdas: %s" %(len(pids + eids), len(total_input_lambdas)))
        assert len(total_input_lambdas) == len(pids + eids)
        for tcard,lambdas in zip(total_cards, total_input_lambdas):
            #print("card: %s total input size: %s lambdas: %s" %(tcard,total_input_size,lambdas.shape))
            assert lambdas.shape == (total_input_size, tcard)

        # computer posterior probability pr(y|x'v,e)
        input_names = []
        for pid in pids:
            input_names.append('v'+str(pid))
        for eid in eids:
            input_names.append('v'+str(eid))
        output_name = 'v'+str(node_y)
        if node_y not in nodes_abs:
            # children of abstracted itself is not an abstracted node 
            super_cpt = VE.posteriors(bn,inputs=input_names,output=output_name,evidence=total_input_lambdas)
            #print("super cpt: %s" % (super_cpt.shape,))
            shape=tuple(super_pcards)+(card,)
            super_cpt = super_cpt.reshape(shape)
            abs_bnNode = abstract_bn.node(name)
            abs_bnNode._cpt = super_cpt
        else:
            # special case that children of abstracted node itself is also an abstracted node
            scard = states_map_dict[node_y]
            super_cpt = VE.posteriors(bn,inputs=input_names,output=output_name,evidence=total_input_lambdas)
            #print("super cpt: %s" % (super_cpt.shape,))
            shape=tuple(super_pcards)+(card,)
            super_cpt = super_cpt.reshape(shape)
            arrays = []
            for states in states_map_dict[node_y]:
                array = np.sum(super_cpt[...,np.array(states)],axis=-1,keepdims=True)
                arrays.append(array)
            super_cpt = np.concatenate(arrays,axis=-1)
            abs_bnNode = abstract_bn.node(name)
            abs_bnNode._cpt = super_cpt

        nodes_reparam.append(node_y)
        print("Finish reparam child %s" %str(node_y))

    # case 3: for the remaining nodes, copy cpt from bn
    for node_i in range(n_nodes):
        if node_i not in nodes_abs and node_i not in nodes_children_of_abs:
            name = 'v'+str(node_i)
            bnNode = bn.node(name)
            cpt = bnNode.cpt
            abs_bnNode = abstract_bn.node(name)
            abs_bnNode._cpt = np.copy(cpt)
            nodes_reparam.append(node_i)

    print("num nodes: %d nodes processed:%s" %(n_nodes, nodes_reparam))
    assert len(nodes_reparam) == n_nodes 
    print("Finish reparam bn.")


            







        

def dot2(dag,node_q,nodes_evid,nodes_abs,fname="bn.gv"):
    d = Digraph()
    d.attr(rankdir='TD')
    for node in range(len(dag)):
        if node in nodes_evid:
            d.node('v'+str(node),shape="circle", style="filled")
        elif node in nodes_abs:
            d.node('v'+str(node),shape="doublecircle")
        else:
            d.node('v'+str(node),shape="circle")
    for node in range(len(dag)):
        for child in dag[node]:
            d.edge('v'+str(node),'v'+str(child))
    try:
        d.render(fname, view=False)
    except:
        print("Need to download graphviz")



def do_poly_tree_experiment():
    dag = get_random_polytree(NUM_NODES,NUM_ITERS)
    #order = topo_sort(dag)
    # print("order: ", order)
    bn = sample_random_BN(dag)
    try:
        bn.dot(fname='polytree.gv',view=False)
    except:
        print("need to download graphviz")
    q,e,x = random_query(dag)
    print("query: %s evidence: %s abstracted: %s" %(q,e,x))
    #x0 = random.choice(x)
    #y0 = random.choice(dag[x0])
    #evidences_above_x0 = evidences_at_or_above(x0,e,dag)
    #print("evidence above %d: %s" %(x0,evidences_above_x0))
    #evidences_in_the_side_of_y0 = evidences_in_the_lower_side_of(x0,y0,e,dag)
    #print("evidence below %d: %s" %(y0,evidences_in_the_side_of_y0))
    alive_evidences = alloc_alive_evidences(q,e,x,dag)
    for node,evid in alive_evidences.items():
        print("node: %s evidences: %s" %(node,evid))
    states_map_dict = get_states_map_dict(bn,x)
    abstract_bn = make_abstract_BN(bn,x,states_map_dict)
    assert len(abstract_bn.nodes) == len(bn.nodes)
    x0 = random.choice(x)
    abs_node_x0 = abstract_bn.node('v'+str(x0))
    print("x0 values: %s" %(abs_node_x0.values))
    evidence_values = {}
    for evid in e:
        evidence_values[evid] = np.random.randint(2)
    dot2(dag,q,e,x)
    reparam(bn,abstract_bn,x,states_map_dict,alive_evidences,evidence_values)
    input_names = ['v'+str(evid) for evid in e]
    output_name = 'v'+str(q)
    ecards = {}
    for eid in e:
        ename = 'v'+str(eid)
        eNode = bn.node(ename)
        ecard = len(eNode.values)
        ecards[eid] = ecard
        # cardinality of evidences
    evidence_ac = []
    for eid in e:
        ecard = ecards[eid]
        evalue = evidence_values[eid]
        lambdas = np.zeros(ecard)
        lambdas[evalue] = 1.0
        evidence_ac.append(lambdas.reshape((1,ecard)))

    query = VE.posteriors(bn,inputs=input_names,output=output_name,evidence=evidence_ac)
    query2 = VE.posteriors(abstract_bn,inputs=input_names,output=output_name,evidence=evidence_ac)
    print("query 1 %s" % (query,))
    print("query 2 %s" % (query2,)) 
    #dot(dag)
    dot2(dag,q,e,x)

# clip small values in distribution
def clip(dist):
    EPSILON = np.finfo('float32').eps
    safe_dist = np.where(np.less(dist, EPSILON), EPSILON, dist)
    return safe_dist

# computes Kullback-Leibler divergence score between dist_true and dist_pred
def KL_divergence(dist_true, dist_pred):
    assert dist_true.shape == dist_pred.shape
    batch_size = dist_true.shape[0]
    dist_true = clip(dist_true).reshape((batch_size,-1)) # clip and flatten
    dist_pred = clip(dist_pred).reshape((batch_size,-1)) 
    kl_loss = dist_true * np.log(dist_true/dist_pred)
    return np.sum(kl_loss, axis=-1)


def do_multiple_connected_experiment():
    dag = get_random_DAG()
    #dot(dag)
    #order = topo_sort(dag)
    # print("order: ", order)
    bn = sample_random_BN(dag)
    try:
        bn.dot(fname='multiple_bn.gv',view=False)
    except:
        print("need to download graphviz")
    q,e,x = random_query_with_ancestral_evidences(dag)
    print("query: %s evidence: %s abstracted: %s" %(q,e,x))
    #x0 = random.choice(x)
    #y0 = random.choice(dag[x0])
    #evidences_above_x0 = evidences_at_or_above(x0,e,dag)
    #print("evidence above %d: %s" %(x0,evidences_above_x0))
    #evidences_in_the_side_of_y0 = evidences_in_the_lower_side_of(x0,y0,e,dag)
    #print("evidence below %d: %s" %(y0,evidences_in_the_side_of_y0))
    empty_evidences = {}
    for node_x in x:
        for node_y in dag[node_x]:
            empty_evidences[node_y] = []
    alive_evidences = alloc_ancestral_evidences(q,e,x,dag)
    for node,evid in alive_evidences.items():
        print("node: %s evidences: %s" %(node,evid))
    try:
        dot2(dag,q,e,x)
    except:
        print("Need to download graphviz")
    #exit(1)
    states_map_dict = get_states_map_dict(bn,x)
    abstract_bn = make_abstract_BN(bn,x,states_map_dict)
    abstract_bn_2 = make_abstract_BN(bn,x,states_map_dict)
    assert len(abstract_bn.nodes) == len(bn.nodes)
    x0 = random.choice(x)
    abs_node_x0 = abstract_bn.node('v'+str(x0))
    print("x0 values: %s" %(abs_node_x0.values))
    evidence_values = {}
    for evid in e:
        evidence_values[evid] = np.random.randint(2)
    reparam(bn,abstract_bn,x,states_map_dict,alive_evidences,evidence_values)
    reparam(bn,abstract_bn_2,x,states_map_dict,empty_evidences,evidence_values)
    input_names = ['v'+str(evid) for evid in e]
    output_name = 'v'+str(q)
    ecards = {}
    for eid in e:
        ename = 'v'+str(eid)
        eNode = bn.node(ename)
        ecard = len(eNode.values)
        ecards[eid] = ecard
        # cardinality of evidences
    evidence_ac = []
    for eid in e:
        ecard = ecards[eid]
        evalue = evidence_values[eid]
        lambdas = np.zeros(ecard)
        lambdas[evalue] = 1.0
        evidence_ac.append(lambdas.reshape((1,ecard)))

    query = VE.posteriors(bn,inputs=input_names,output=output_name,evidence=evidence_ac)
    query1 = VE.posteriors(abstract_bn,inputs=input_names,output=output_name,evidence=evidence_ac)
    query2 = VE.posteriors(abstract_bn_2,inputs=input_names,output=output_name,evidence=evidence_ac)
    KL_loss_1 = KL_divergence(query,query1)
    KL_loss_2 = KL_divergence(query,query2)
    gain = KL_loss_1 / KL_loss_2
    print("query 0 %s" % (query,))
    print("query 1 %s" % (query1,))
    print("query 2 %s" % (query2,))
    print("KL loss 1: %s KL loss 2: %s gain: %s" %(KL_loss_1,KL_loss_2,gain) )
    #dot(dag)
    dot2(dag,q,e,x)


if __name__ == '__main__':
    #do_poly_tree_experiment()
    do_multiple_connected_experiment()
    


    

    