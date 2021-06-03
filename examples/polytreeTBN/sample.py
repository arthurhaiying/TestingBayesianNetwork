from graphviz import Digraph
import numpy as np
import itertools as iter
import random,csv
import re

from pathlib import Path
import os,sys
import warnings

from numpy.lib.polynomial import _polyfit_dispatcher

if __name__ == '__main__':
    #basepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    basepath = Path(__file__).resolve().parents[2]
    #print("basepath: %s" %basepath)
    sys.path.append(str(basepath))

from tbn.tbn import TBN
from tbn.node import Node
from examples.CaseStudy.NetParser import writeBN

output_dirpath = Path(__file__).resolve().parent / 'data'


# sample random cpt for node of card with parents of cards with uniform/deterministic cpt
# deterministic: a flag for deterministic or not
# min_pos_value: fraction of probability for the next state, set only when deterministic is set
def random_cpt(card,cards,deterministic=False,min_pos_rate=None):
    arrays = []
    # sample random cond dist
    def __old_random_dist(length):
        warnings.warn("this random dist is deprecated.", warnings.DeprecationWarning)
        if deterministic:
            # assign one state with probabiliy >= 0.8 and assign uniformly remaining value 
            pos = random.uniform(min_pos_rate,1.0)
            neg = (1.0-pos)/(length-1)
            dist = [neg]*length
            index = random.randint(0,length-1)
            dist[index] = pos
            return np.array(dist)
        else:
            dist = np.array([random.uniform(0.0,1.0) for _ in range(length)])
            dist = dist/sum(dist)
            return dist

    def __random_dist1(length):
        if deterministic:
            # keep assigning the next state with >=0.8 of remaining probability
            dist = np.zeros(length)
            prob = 1.0
            for i in range(length):
                if i < length-1:
                    pos = random.uniform(min_pos_rate,1.0)
                    pos *= prob
                    prob -= pos
                    dist[i] = pos
                else:
                    dist[i] = prob
            dist = dist/sum(dist) # for floating point error
            np.random.shuffle(dist)
            return dist
        else:
            dist = np.array([random.uniform(0.0,1.0) for _ in range(length)])
            dist = dist/sum(dist)
            return dist

    for _ in iter.product(*list(map(range,cards))):
        #array = __old_random_dist(card)
        array = __random_dist1(card)
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

# return parents of node x in dag
# dag: a list of list representing the adjacency list of a graph
def get_parents(dag,x):
    n_nodes = len(dag)
    parents = []
    for p in range(n_nodes):
        if x in dag[p]:
            parents.append(p)
    return parents

# return ancestors of node x in dag
# dag: a list of list representing the adjacency list of a graph
def get_ancestors(dag,x):
    ancestors = set()
    queue = [x]
    while queue:
        v = queue.pop(0)
        parents = get_parents(dag,v)
        for p in parents:
            if p not in ancestors:
                ancestors.add(p)
                queue.append(p)

    return list(ancestors)

# choose a random query over dag
# dag: a list of list representing the adjacency list of a graph
# return: a tuple (Q, E, A) where Q is the query node, E are evidence nodes, and A are abstracted nodes
def random_query(dag):
    n_nodes = len(dag)
    leaves = [x for x in range(n_nodes) if len(dag[x]) == 0]
    roots = [x for x in range(n_nodes) if len(get_parents(dag,x)) == 0]
    node_q = random.choice(leaves) # choose a random leaf as query
    nodes_a = get_ancestors(dag,node_q) # choose all ancestors of query as abstracted nodes
    #nodes_e = [x for x in leaves if x != node_q] # choose remaining leaves as evidence
    nodes_e =  [x for x in leaves+roots if x != node_q and x not in nodes_a]
    return node_q,nodes_e,nodes_a

# assign cardinality for nodes in dag
# card: cardinality of regular nodes
# card_a: original cardinality of abstracted nodes
# return: a list representing cardinality of each node id
def assign_cards(dag,nodes_a,card,card_a):
    n_nodes = len(dag)
    cards = [card]*n_nodes
    for x in nodes_a:
        cards[x] = card_a
    return cards

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
def make_bn(dag,cards,deterministic,min_pos_rate):
    bn = TBN('polytree')
    order = topo_sort(dag)
    for x in order:
        # add node in topological order
        name = 'v%d'%x
        card = cards[x]
        values = ['q%d'%i for i in range(card)]
        pids = get_parents(dag,x)
        pnodes = [bn.node('v%d'%pid) for pid in pids]
        pcards = [cards[pid] for pid in pids]
        cpt = random_cpt(card,pcards,deterministic,min_pos_rate)
        node_x = Node(name,values=values,parents=pnodes,testing=False,cpt=cpt)
        bn.add(node_x)

    return bn

# get node id from node name, assume that node name is v+str(id)
def node_id(node):
    return int(re.sub('\D','',node.name))

# direct sample instances (nodes_evid, node_q) from bn
# return: 
#   samples -- an np array representing sampled data (n_examples,n_vars)
def direct_sample(bn,vars,n_examples):
    print("Start sampling dataset of %d examples..." % n_examples)
    n_nodes = len(bn.nodes)
    samples = []

    for _ in range(n_examples):
        sample = [None]*n_nodes
        for node in bn.nodes:
            # for each node in topo order
            id = node_id(node) #id
            card = len(node.values)
            pnodes,cpt = node.parents,node.cpt
            pids = [node_id(p) for p in pnodes]
            pvalues = [sample[pid] for pid in pids] 
            # parents are already sampled
            try:
                cond = cpt[tuple(pvalues)]
                value = np.random.choice(np.arange(card),p=cond)
                sample[id] = value
            except ValueError:
                print("cpt shape: %s" %cpt.shape)
                print("node %d cpt is uncnormalized.")
                print("cond: %s" %cond)
                exit(1)
            # finish one sample
        samples.append(sample)

    # sampling done
    print("Finish sampling dataset")
    samples = np.array(samples)
    samples = samples[:,np.array(vars)]
    assert samples.shape == (n_examples,len(vars))
    return samples


def direct_sample_and_save_data(bn,vars,n_examples,dirpath):
    n_nodes = len(bn.nodes)
    samples = direct_sample(bn,vars,n_examples)
    # sample training data
    filepath = dirpath / 'train_examples.csv'
    with open(filepath,'w+') as file:
        writer = csv.writer(file)
        fields = ['v%d'%var for var in vars]
        writer.writerow(fields)
        writer.writerows(samples)
        file.flush()

    return samples

# visualize sampled dag and query
def dot(dag,nodes_e,nodes_a,fname):
    d = Digraph()
    d.attr(rankdir='TD')
    n_nodes = len(dag)
    # create nodes
    d.attr('node',width='0.5',fixedsize='true')
    for x in range(n_nodes):
        name = str(x)
        label = 'v'+name
        if x in nodes_a:
            d.node(name,label,shape='doublecircle',width='0.4')
        elif x in nodes_e:
            d.node(name,label,shape='circle',style='filled')
        else:
            name = str(x)
            label = 'v'+name
            d.node(name,label,shape='circle')
    # add edges
    for u in range(n_nodes):
        for v in dag[u]:
            d.edge(str(u),str(v))
    try:
        d.render(fname, view=False)
    except:
        print("Need to download graphviz")

def save_query(node_q,nodes_e,nodes_a,fname):
    with open(fname,'w+') as f:
        f.write(f"query: {node_q}")
        f.write('\n')
        evid = ['v%d'%e for e in nodes_e]
        abs = ['v%d'%a for a in nodes_a]
        f.write("evidence: {}".format(','.join(evid)))
        f.write('\n')
        f.write("abstract: {}".format(','.join(abs)))
        f.write('\n')
        f.flush()

def sample(n_nodes,card,card_a,n_iters,deterministic,min_pos_rate,n_examples,trial_id,output_dirpath):
    print("Start sampling trial %d..." %trial_id)
    # set random seeds
    random.seed(trial_id)
    np.random.seed(trial_id)
    data_dirpath = output_dirpath / f'BN{n_nodes}' / f'trial{trial_id}'
    if not data_dirpath.exists():
        Path.mkdir(data_dirpath,parents=True,exist_ok=True)
        # create missing directory

    dag = get_random_polytree(n_nodes,n_iters) # sample structure
    q,e,a = random_query(dag) # sample query
    cards = assign_cards(dag,a,card,card_a) # assign states
    bn = make_bn(dag,cards,deterministic,min_pos_rate) # create bn and sample cpt
    dot(dag,e,a,fname=data_dirpath/"polytree.gv") # save dag
    writeBN(bn,fname=data_dirpath/'polytree.net') # save bn
    save_query(q,e,a,fname=data_dirpath/"query.txt") # save query
    samples = direct_sample_and_save_data(bn,e+[q],n_examples,data_dirpath)
    print("Finish sampling trial %d." %trial_id)
    return bn,q,e,a,samples


if __name__ == '__main__':
    # set experimental parameters
    n_nodes,card,card_a = 20,2,5
    n_iters = 1000
    deterministic,min_pos_rate = True,0.8
    n_examples = 100
    sample(n_nodes,card,card_a,n_iters,deterministic,min_pos_rate,n_examples,
        trial_id=1,output_dirpath=output_dirpath)
    print("finish sampling")
    













