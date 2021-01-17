from graphviz import Digraph
import numpy as np
import random
import re
from copy import copy,deepcopy
import itertools as iter

from pathlib import Path
import os,sys

if __name__ == '__main__':
    #basepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    basepath = Path(__file__).resolve().parents[2]
    #print("basepath: %s" %basepath)
    sys.path.append(str(basepath))

from examples.polytreeTBN.config import *
import examples.polytreeBN.polytree as polytreeBN 

from tbn.tbn import TBN
from tbn.node import Node
from tbn.node2 import NodeV2
import tbn.cpt as CPT
import train.data as data
from tac import TAC, TACV2


# generate random cpt for node of cardinality card with parents of cardinality cards
def random_cpt(card,cards,deterministic=False):
    MIN_POS_VALUE = 0.8
    arrays = []
    # sample random cond dist
    def __random_dist(length):
        if deterministic:
            pos = random.uniform(MIN_POS_VALUE,1.0)
            neg = (1.0-pos)/(length-1)
            dist = np.ones(length) * neg
            index = random.randint(0,length-1)
            dist[index] = pos
            return dist
        else:
            dist = np.array([random.uniform(0.0,1.0) for _ in range(card)])
            dist = dist/np.sum(dist)
            return np.array(dist)

    for _ in iter.product(*list(map(range,cards))):
        array = __random_dist(card)
        arrays.append(array)

    shape = tuple(cards) + (card,)
    cpt = np.array(arrays).reshape(shape)
    return cpt



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

# return parents of every node in dag
def get_parent_list(dag):
    # get parents of each node
    n_nodes = len(dag)
    parent_list = [[] for _ in range(n_nodes)]
    for node in range(n_nodes):
        for child in dag[node]:
            parent_list[child].append(node)
    return parent_list

# return nodes connected to given nodes in dag
# dag - a dict of list representing the adjaceny list of graph
def connected_nodes(node_x,dag):
    adj_list = {}
    def __parents(node):
        parents = []
        for p in dag.keys():
            if node in dag[p]:
                parents.append(p)
        return parents

    for node in dag.keys():
        adj_list[node] = dag[node] + __parents(node)
        # get adjacneyc list from dag

    visited_nodes = []
    queue = [node_x]
    visited_nodes.append(node_x)
    while queue:
        curr = queue.pop(0)
        for neighbor in adj_list[curr]:
            if neighbor not in visited_nodes:
                visited_nodes.append(neighbor)
                queue.append(neighbor)
    return visited_nodes

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


# randomly generate a true BN of dag structure
# return a tbn object
def sample_random_BN(dag,node_q,nodes_evid,nodes_abs):
    # sample polytree DAG
    n_nodes = len(dag)
    parent_list = get_parent_list(dag)
    order = topo_sort(dag) 
    # return a topological ordering of nodes
    cards = []
    for node in range(n_nodes):
        if node in nodes_evid:
            cards.append(EVIDENCE_CARDINALITY)
        elif node in nodes_abs:
            cards.append(ABSTRACT_CARDINALITY)
        else:
            cards.append(np.random.randint(low=MIN_CARDINALITY,high=MAX_CARDINALITY+1))
            # random cardinality for each node 
    bnNode_cache = {}
    bn = TBN("polytree")
    # print("Start adding nodes...")
    for node in order:
        name = 'v'+str(node)
        card = cards[node]
        values = ['state_'+str(i) for i in range(card)]
        parentNodes = [bnNode_cache[p] for p in parent_list[node]]
        pcards = [cards[p] for p in parent_list[node]]
        cpt = random_cpt(card, pcards,deterministic=DETERMINISTIC_CPTS)
        bnNode = Node(name,values=values,parents=parentNodes,testing=False,cpt=cpt)
        bnNode_cache[node] = bnNode
        bn.add(bnNode)
        # add bn node to bn

    return bn,cards

# randomly generate a true TBN of dag structure
# return a tbn object
def sample_random_TBN(dag,sel_type='threshold'):
    p0 = 0.5
    # sample polytree DAG
    n_nodes = len(dag)
    parent_list = get_parent_list(dag)
    order = topo_sort(dag) 
    # return a topological ordering of nodes
    cards = [np.random.randint(low=MIN_CARDINALITY,high=MAX_CARDINALITY+1) for _ in range(n_nodes)]
    # random cardinality for each node 
    tbnNode_cache = {}
    tbn = TBN("polytree")
    # print("Start adding nodes...")
    for node in order:
        name = 'v'+str(node)
        card = cards[node]
        values = ['state'+str(i) for i in range(card)]
        parentNodes = [tbnNode_cache[p] for p in parent_list[node]]
        pcards = [cards[p] for p in parent_list[node]]
        p = np.random.uniform()
        if len(parentNodes) >= 1 and p >= p0:
            cpt1 = CPT.random(card,pcards)
            cpt2 = CPT.random(card,pcards)
            thres = CPT.random2(pcards)
            tbnNode = Node(name,values=values,parents=parentNodes,testing=True,
                cpt1=cpt1,cpt2=cpt2,threshold=thres)
            tbnNode_cache[node] = tbnNode
            tbn.add(tbnNode)
        else:
            cpt = CPT.random(card, pcards)
            tbnNode = Node(name,values=values,parents=parentNodes,testing=False,cpt=cpt)
            tbnNode_cache[node] = tbnNode
            tbn.add(tbnNode)
            # add bn node to bn
    return tbn,cards


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
    print("Search for a good query...")
    while True:
        n_nodes = len(dag)
        # choose a query node from leaf nodes
        leaves = [node for node in range(n_nodes) if len(dag[node]) == 0]
        node_q = np.random.choice(leaves)

        # ancestors_q of node q
        ancestors_q = ancestors(node_q,dag)
        # choose abstracted nodes from ancestors_q of node q
        ancestors_q = list(ancestors_q)
        num_abs_nodes = min(len(ancestors_q), MAX_NUM_ABSTRACTED_NODES)
        nodes_abs = list(np.random.choice(ancestors_q,size=num_abs_nodes,replace=False))

        # choose evidence nodes from the remaining nodes in dag
        nodes_remaining = [node for node in range(n_nodes) if node != node_q and 
            node not in nodes_abs]
        num_evid_nodes = min(len(nodes_remaining), MAX_NUM_EVIDENCE_NODES)
        #num_evid_nodes = min(len(nodes_remaining), MAX_NUM_EVIDENCE_NODES)
        nodes_evid = list(np.random.choice(nodes_remaining,size=num_evid_nodes,replace=False))
        # prune BN for this query
        dag_pruned,node_q,nodes_evid,nodes_abs = prune(dag,node_q,nodes_evid,nodes_abs)

        alive_evidences = polytreeBN.alloc_alive_evidences(node_q,nodes_evid,nodes_abs,dag_pruned)
        alive_count = sum(map(lambda evid: len(evid) > 0, alive_evidences.values()))
        # count node that has some alive evidence
        if len(nodes_evid) >= MIN_NUM_EVIDENCE_NODES and len(nodes_abs) >= MIN_NUM_ABSTRACTED_NODES:
            # find a good query
            if alive_count >= len(alive_evidences) // 2:
                break

    return dag_pruned,node_q, nodes_evid, nodes_abs

def prune_dag(dag,prunes):
    dag2 = {}
    for node in dag.keys():
        children = dag[node]
        if node in prunes:
            continue
        else:
            children = set(children) - set(prunes)
            dag2[node] = list(children)
    return dag2

# given dag and query, evidence,asbtracted nodes, prune dag with respect to query and evidence
def prune(dag,node_q,nodes_evid,nodes_abs):
    actives = set()
    for node in nodes_evid+[node_q]:
        #print("node: %d num nodes:%d" %(node, len(dag)))
        actives |= set(ancestors(node,dag))
    actives |= set(nodes_evid+[node_q])
    # keep pruning leaf nodes until q and e
    prunes = set(range(len(dag))) - actives
    dag = {i:children for i,children in enumerate(dag)}
    dag2 = prune_dag(dag,prunes) 
    connected = connected_nodes(node_q,dag2)
    prunes = dag2.keys() - set(connected)
    dag3 = prune_dag(dag2,prunes)
    # keep nodes connected to node_q
    nodes_evid = list(set(nodes_evid) & set(connected))
    nodes_abs = list(set(nodes_abs) & set(connected))

    connected.sort()
    ids = {node:i for i,node in enumerate(connected)}
    dag_pruned = [None]*len(connected)
    for node,children in dag3.items():
        node = ids[node]
        children = list(map(lambda x: ids[x], children))
        dag_pruned[node] = children

    node_q = ids[node_q]
    nodes_evid =  list(map(lambda x: ids[x], nodes_evid))
    nodes_abs = list(map(lambda x: ids[x], nodes_abs))
    return dag_pruned,node_q,nodes_evid,nodes_abs


