from graphviz import Digraph
import numpy as np
import random
import re
from copy import copy,deepcopy
import itertools as iter

from pathlib import Path
import os,sys
from multiprocessing import Pool
import tqdm
#from examples.polytreeTBN.Pool import MyPool

if __name__ == '__main__':
    #basepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    basepath = Path(__file__).resolve().parents[2]
    #print("basepath: %s" %basepath)
    sys.path.append(str(basepath))

from examples.polytreeTBN.config import *


# posteriors smaller than this value is ignored

from tbn.tbn import TBN
from tbn.node import Node
from tbn.node2 import NodeV2
import examples.polytreeTBN.model as model 
import examples.polytreeBN.polytree as polytreeBN 
import examples.polytreeTBN.polytree as polytreeTBN
from examples.polytreeTBN.LookUpTable import LookUpTable 
import tbn.cpt as CPT
import train.data as data
from tac import TAC, TACV2

from compile.opsgraph import OpsGraph as og
og.USE_FIXED_THRESHOLDS = USE_FIXED_THRESHOLDS
#og.USE_ASCEND_THRESHOLDS = USE_ASCEND_THRESHOLDS



# get node id from node name, assume that node name is v+str(id)
def node_id(node):
    return int(re.sub('\D','',node.name))

# direct sample instances (nodes_evid, node_q) from bn
# return: 
#   evidences -- a list of list representing hard evidences for nodes_evid (num_examples, evid_size)
#   marginals -- a list representing hard evidence for node_q (num_examples,)
def direct_sample(bn,node_q,nodes_evid,cards_dict,num_examples):
    print("Start sampling dataset...")
    n_nodes = len(bn.nodes)
    evid_size = len(nodes_evid)
    evidences, marginals = [],[]

    for _ in range(num_examples):
        sample = [None]*n_nodes
        for node in bn.nodes:
            # for each node in topo order
            id = node_id(node) #id
            card = cards_dict[id]
            parents,cpt = node.parents,node.cpt
            pids = [node_id(p) for p in parents]
            pvalues = [sample[pid] for pid in pids] 
            # parents are already sampled
            cond = cpt[tuple(pvalues)]
            value = np.random.choice(np.arange(card),p=cond)
            sample[id] = value

        evid = [sample[eid] for eid in nodes_evid] # get evidence value
        query = sample[node_q] # get query vale
        evidences.append(evid)
        marginals.append(query)

    # sampling done
    print("Finish sampling dataset")
    assert np.array(evidences).shape == (num_examples,evid_size)
    assert np.array(marginals).size == num_examples
    return evidences,marginals

# create incomplete bn/tbn over smaller cardinality space
def make_incomplete_tbn(bn,nodes_abs,cards_dict,alive_evidences,num_intervals=None):
    tbn = TBN("testing polytree")
    scards_dict = {node_x:(cards_dict[node_x]+1)//2 for node_x in nodes_abs} 
    # abstracted nodes lose half cardinalities
    for node in bn.nodes:
        # for each node parent before child
        id = node_id(node)
        name,values,parents = node.name,node.values,node.parents
        pids = list(map(node_id, parents))
        parents = list(map(lambda x: tbn.node(x.name),parents))
        if id in nodes_abs:
            scard = scards_dict[id]
            values = ['state_sup_'+str(i) for i in range(scard)]

        if id not in nodes_abs and id not in alive_evidences.keys():
            # base case: if not abstracted or child of abstracted nodes, copy node
            node2 = Node(name,values=values,parents=parents,testing=False)
            tbn.add(node2)

        elif id in alive_evidences.keys() and alive_evidences[id]:
            # for tbn, child of abstracted nodes become testing nodes
            assert num_intervals is not None
            card = len(values)
            pcards = tuple([len(p.values) for p in parents])
            if USE_FIXED_THRESHOLDS:
                step = 1.0/num_intervals
                thress = [step*(i+1) * np.ones(pcards) for i in range(num_intervals-1)]
                # fixed n-1 thresholds
                node2 = NodeV2(name,values=values,parents=parents,testing=True,thresholds=thress,
                    num_intervals=num_intervals)
            else:
                node2 = NodeV2(name,values=values,parents=parents,testing=True,num_intervals=num_intervals)
            tbn.add(node2)

        else:
            # if abstracted node or child of abstracted node or in-alive testing node
            node2 = Node(name,values=values,parents=parents,testing=False)
            tbn.add(node2)

    return tbn


def dot(dag,node_q,nodes_evid,nodes_abs,fname="bn.gv"):
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


# return mapping from original states to superstates
# states_map - a list of list representing the original states contained in each super-state
def get_cards_map(card, scard):
    cards_map = [[] for _ in range(scard)]
    for i in range(card):
        cards_map[i % scard].append(i)
    return cards_map

def clip(dist):
    epsilon = np.finfo('float32').eps
    dist_safe = np.where(dist<epsilon, epsilon, dist)
    return dist_safe

def KL_divergence(dist_p,dist_q):
    dist_p = clip(dist_p)
    dist_q = clip(dist_q)
    kl_loss = np.sum(dist_p * np.log(dist_p/dist_q),axis=-1)
    return np.mean(kl_loss)

# fit ac multiple times and return best result
def fit_and_find_best(tac,evidences,marginals):
    tac_weights_list = []
    loss_list = []

    train_data, val_data = data.random_split(evidences,marginals,percentage=0.2)
    train_evid, train_marg, _ = train_data
    val_evid, val_marg, _ = val_data
    # fit mulitple trials
    for i in range(NUM_FIT_TRIALS):
        print("train trial %d..." %i)
        tac.fit(train_evid,train_marg,loss_type='CE',metric_type='CE') # train AC
        tac_weights = tac.tac_graph.learned_weights # save weights for loading later
        loss = tac.metric(val_evid,val_marg,metric_type='CE') # validate tac
        tac_weights_list.append(tac_weights)
        loss_list.append(loss)

    INFINITY = 1000
    min_loss, min_index = INFINITY, None
    for i, loss in enumerate(loss_list):
        if loss < min_loss:
            min_loss = loss
            min_index = i

    best_tac_weights = tac_weights_list[min_index]
    tac.tac_graph.learned_weights = best_tac_weights
    tac.tac_graph.restore_saved_weights() # load best weight found
    assert np.isclose(min_loss, tac.metric(val_evid,val_marg,metric_type='CE')) # make sure best tac
    return tac

def reparam_tbn(args):
    tbn = polytreeTBN.reparam_tbn(*args)
    return tbn

def fit_tac(args):
    return fit_and_find_best(*args)

def do_learn_polytree_tbn_experiment():
    ''' step 1: sample true BN and query'''
    dag = model.get_random_polytree(NUM_NODES,NUM_ITERS)
    #dag,q,e,x = model.random_query(dag)
    dag,q,e,x = model.random_query(dag)
    dot(dag,q,e,x,fname="polytree.gv")
    print("query: %s evidence: %s abstracted: %s" %(q,e,x))

    inputs = ['v%d'%eid for eid in e]
    output = 'v%d' % q
    alive_evidences = polytreeBN.alloc_alive_evidences(q,e,x,dag) # all testing nodes are active
    print("testing nodes %s" %(alive_evidences,))
    bn,cards = model.sample_random_BN(dag,q,e,x) # sample true bn
    ac_true = TAC(bn,inputs,output,trainable=False)

    # direct sample training set
    ecards = list(map(lambda x: cards[x], e))
    qcard = cards[q]
    evidences,marginals = direct_sample(bn,q,e,cards,num_examples=NUM_EXAMPLES)
    evidences,marginals = data.evd_hard2lambdas(evidences,ecards), data.marg_hard2lambdas(marginals,qcard)

    ''' step 2: make incomplete tbn for learning'''
    tbn_learn_list = [make_incomplete_tbn(bn,x,cards,alive_evidences,num_intervals=interval) 
        for interval in intervals_list]
    tac_learn_list = [TAC(tbn_learn,inputs,output,trainable=True,sel_type=SELECT_CPT_TYPE)
        for tbn_learn in tbn_learn_list]

    ''' step 3: make incomplete tbn by handcraft'''
    print("Start reparam tbn...")
    tbn_hand_list = []
    scards = [(card+1)//2 for card in cards] # lose half states
    cards_map_dict = []
    for card,scard in zip(cards,scards):
        cards_map = get_cards_map(card,scard)
        cards_map_dict.append(cards_map)

    with Pool(processes=NUM_WORKERS) as pool:
        parameters = [(dag,bn,q,e,x,cards,scards,cards_map_dict,interval) for interval in intervals_list]
        for tbn_hand in tqdm(pool.imap_unordered(reparam_tbn,parameters),total=len(parameters)):
            tbn_hand_list.append(tbn_hand)

    tac_hand_list = [TAC(tbn_hand,inputs,output,trainable=False,sel_type='threshold')
        for tbn_hand in tbn_hand_list]
    print("Finish reparam tbn...")

    # training
    print("Start training...")
    #ac_incomplete.fit(evidences,marginals,loss_type='CE',metric_type='CE')

    with Pool(processes=NUM_WORKERS) as pool:
        parameters = [(tac_learn,evidences,marginals) for tac_learn in tac_learn_list]
        tac_learn_list = []
        for tac_learn in tqdm(pool.imap_unordered(fit_tac, parameters),total=len(parameters)):
            tac_learn_list.append(tac_learn)
    '''
    tac_learn_list  = [fit_and_find_best(tac_learn,evidences,marginals)
        for tac_learn in tac_learn_list]
        '''
    print("Finish training TAC...")

    # evaluation
    test_evidences = list(iter.product(*list(map(range,ecards)))) # enumerate all possible evidences
    test_evidences = data.evd_hard2lambdas(test_evidences,ecards)
    test_marginals = ac_true.evaluate(test_evidences)

    marginals_hand_list = [tac_hand.evaluate(test_evidences) for tac_hand in tac_hand_list]
    marginals_learn_list = [tac_learn.evaluate(test_evidences) for tac_learn in tac_learn_list]

    kl_hand_list = [KL_divergence(test_marginals,marginals_hand) for marginals_hand in marginals_hand_list]
    kl_learn_list = [KL_divergence(test_marginals,marginals_learn) for marginals_learn in marginals_learn_list]
    
    print("kl hand: %s kl learn: %s" %(kl_hand_list,kl_learn_list))
    print("Finish recover experiment")
    return kl_hand_list,kl_learn_list

def do_avg_learn_polytree_tbn_experiment():
    kl_hand_lists = [[] for _ in range(len(intervals_list))]
    kl_learn_lists = [[] for _ in range(len(intervals_list))]
    f = open("output_%d_for_learn_final.txt"%NUM_NODES, mode='w')

    for i in range(NUM_TRIALS):
        print("Start trial %d" %i)
        kl_hand, kl_learn = do_learn_polytree_tbn_experiment()
        for list, kl in zip(kl_hand_lists,kl_hand):
            list.append(kl)
        for list, kl in zip(kl_learn_lists,kl_learn):
            list.append(kl)

        means_hand = [np.array(kl_loss).mean() for kl_loss in kl_hand_lists]
        means_learn =  [np.array(kl_loss).mean() for kl_loss in kl_learn_lists]
        print("Trial %d kl hand: %s kl learn: %s" %(i,means_hand,means_learn))
        f.write("Trial %d kl hand: %s kl learn: %s" %(i,means_hand,means_learn))
        f.write('\n')
        f.flush()

    print("Finish experiment.")
    f.close()

def reparam(pid,args,result):
    print("Reparam p%d" % pid)
    tbn = polytreeTBN.reparam_tbn(*args)
    result.append((pid,tbn))
    return


if __name__ == '__main__':
    do_avg_learn_polytree_tbn_experiment()
    






