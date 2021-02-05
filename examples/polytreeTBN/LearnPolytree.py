from graphviz import Digraph
import numpy as np
import random
import re
from copy import copy,deepcopy
import itertools as iter

from pathlib import Path
import os,sys
import multiprocessing
from multiprocessing import Pool,Process
from tqdm import tqdm
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
    print("Start sampling dataset of %d examples..." % num_examples)
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
            if np.isclose(np.sum(cond),1.0):
                cond = cond / np.sum(cond)

            try:
                value = np.random.choice(np.arange(card),p=cond)
                sample[id] = value
            except ValueError:
                print("node %s has cpt do not sum to 1" %id)
                print("parents: %s" %pvalues)
                print("probability: %s" %cond)
                exit(1)


        evid = [sample[eid] for eid in nodes_evid] # get evidence value
        query = sample[node_q] # get query vale
        evidences.append(evid)
        marginals.append(query)

    # sampling done
    print("Finish sampling dataset")
    assert np.array(evidences).shape == (num_examples,evid_size)
    assert np.array(marginals).size == num_examples
    return evidences,marginals
    
class direct_sample_fun_wrapper:
    def __init__(self,bn,node_q,nodes_evid,cards_dict,num_examples):
        self.args = [bn,node_q,nodes_evid,cards_dict,num_examples]
    def __call__(self,num):
        np.random.seed(SEED+num) # make sure different examples 
        return direct_sample(*self.args)


def direct_sample_mp(bn,node_q,nodes_evid,cards_dict,num_examples):
    print("Start sampling dataset of %d examples..." % num_examples)
    NUM_SAMPLE_WORKERS = 15
    evidences,marginals = [],[]
    direct_sample_fun = direct_sample_fun_wrapper(bn,node_q,nodes_evid,cards_dict,num_examples//NUM_SAMPLE_WORKERS)
    with Pool(NUM_SAMPLE_WORKERS) as p:
        for evidence,marginal in tqdm(p.imap(direct_sample_fun, list(range(NUM_SAMPLE_WORKERS))),
            total=NUM_SAMPLE_WORKERS,desc="Sample data..."):
            evidences.extend(evidence)
            marginals.extend(marginal)
    assert len(evidences) == len(marginals)
    print("Finish sampling dataset of %d examples." %len(evidences))
    return evidences,marginals


# make incomplete bn over smaller cardinality space
def make_incomplete_bn(bn,nodes_abs,cards_dict,scards_dict):
    bn2 = TBN("%s_1" %bn.name)
    # abstracted nodes lose half cardinalities
    for node in bn.nodes:
        # for each node parent before child
        id = node_id(node)
        name,values,parents = node.name,node.values,node.parents
        #pids = list(map(node_id, parents))
        parents = list(map(lambda x: bn2.node(x.name),parents))
        if id in nodes_abs:
            scard = scards_dict[id]
            values = ['state_sup_'+str(i) for i in range(scard)]

        node2 = Node(name,values=values,parents=parents,testing=False)
        bn2.add(node2)

    return bn2

def make_incomplete_tbn2(bn,nodes_abs,cards_dict,scards_dict,num_intervals):
    tbn = TBN("test_%s_%d " % (bn.name,num_intervals))
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

        if len(parents) == 0:
            # for roots
            node2 = Node(name,values=values,parents=parents,testing=False)
            tbn.add(node2)

        else:
            # for other nodes
            card = len(values)
            pcards = tuple([len(p.values) for p in parents])
            num_cpts = num_intervals
            num_thress = num_intervals-1
            cpts = [CPT.random(card,pcards) for i in range(num_cpts)]
            if USE_FIXED_THRESHOLDS:
                step = 1.0/num_intervals
                thress = [step *(i+1)* np.ones(pcards) for i in range(num_thress)]
                # fixed n-1 thresholds
            else:
                thress = [np.random.rand(*pcards) for i in range(num_thress)]
            node2 = NodeV2(name,values=values,parents=parents,testing=True,cpts=cpts,thresholds=thress,
                    num_intervals=num_intervals)
            tbn.add(node2)

    return tbn



# create incomplete bn/tbn over smaller cardinality space
def make_incomplete_tbn(bn,nodes_abs,cards_dict,scards_dict,alive_evidences,num_intervals):
    tbn = TBN("test_%s_%d " % (bn.name,num_intervals))
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
            #assert num_intervals is not None
            # for other nodes
            card = len(values)
            pcards = tuple([len(p.values) for p in parents])
            num_cpts = num_intervals
            num_thress = num_intervals-1
            cpts = [CPT.random(card,pcards) for i in range(num_cpts)]
            if USE_FIXED_THRESHOLDS:
                step = 1.0/num_intervals
                thress = [step *(i+1)* np.ones(pcards) for i in range(num_thress)]
                # fixed n-1 thresholds
            else:
                thress = [np.random.rand(*pcards) for i in range(num_thress)]
            node2 = NodeV2(name,values=values,parents=parents,testing=True,cpts=cpts,thresholds=thress,
                    num_intervals=num_intervals)
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

def cond_KL_divergence(weight,dist_p,dist_q):
    batch_size = weight.size
    assert dist_p.shape[0] == batch_size and dist_q.shape[0] == batch_size
    dist_p = clip(dist_p)
    dist_q = clip(dist_q)
    kl_loss = np.sum(dist_p * np.log(dist_p/dist_q),axis=-1)
    kl_loss = weight * kl_loss
    return np.sum(kl_loss)


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

class learn_tbn_fun_wrapper:
    def __init__(self,inputs,output,sel_type):
        self.inputs = inputs
        self.output = output
        self.sel_type = sel_type
    def add_dataset(self,train_evid,train_marg,test_evid):
        self.train_evid = train_evid
        self.train_marg = train_marg
        self.test_evid = test_evid
    def __call__(self,tbn_learn):
        fnull = open(os.devnull,'w')
        #sys.stdout,sys.stderr = fnull, fnull
        num_intervals = int(tbn_learn.name.split('_')[-1])
        tac_learn = TAC(tbn_learn,self.inputs,self.output,sel_type=self.sel_type,trainable=True)
        #tac_learn = fit_and_find_best(tac_learn,self.train_evid,self.train_marg)
        tac_learn.fit(self.train_evid,self.train_marg,loss_type='CE',metric_type='CE',fname="logs/cpts/%s.txt" % tbn_learn.name)
        # train tac
        test_marg = tac_learn.evaluate(self.test_evid) # evaluate on testing set
        #sys.stdout,sys.stderr = sys.__stdout__, sys.__stderr__
        return test_marg


def do_learn_polytree_tbn_experiment(num_trial):
    print("Start trial %d..." %num_trial)
    ''' step 1: sample true BN and query'''
    random.seed(SEED + num_trial)
    np.random.seed(SEED + num_trial)
    ok = False
    dag,q,e,x = None,None,None,None
    while not ok:
        # search a good query
        dag = model.get_random_polytree(NUM_NODES,NUM_ITERS)
        #dag,q,e,x = model.random_query(dag)
        ok,dag,q,e,x = model.random_query(dag)

    dot(dag,q,e,x,fname="polytree.gv")
    print("query: %s evidence: %s abstracted: %s" %(q,e,x))

    inputs = ['v%d'%eid for eid in e]
    output = 'v%d' % q
    alive_evidences = polytreeBN.alloc_alive_evidences(q,e,x,dag) # all testing nodes are active
    print("testing nodes %s" %(alive_evidences,))
    bn,cards = model.sample_random_BN(dag,x) # sample true bn
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

    for interval in intervals_list:
        tbn_hand = polytreeTBN.reparam_tbn(dag,bn,q,e,x,cards,scards,cards_map_dict,num_intervals=interval)
        tbn_hand_list.append(tbn_hand)

    tac_hand_list = [TAC(tbn_hand,inputs,output,trainable=False,sel_type='threshold')
        for tbn_hand in tbn_hand_list]
    print("Finish reparam tbn...")

    # training
    print("Start training...")
    #ac_incomplete.fit(evidences,marginals,loss_type='CE',metric_type='CE')
    tac_learn_list  = [fit_and_find_best(tac_learn,evidences,marginals)
        for tac_learn in tac_learn_list]

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
    print("Finish trial %d..." %num_trial)
    return kl_hand_list,kl_learn_list


# Given true BN and query (q,e,x), do reparam tbn and learning tbn
def do_one_experiment(dag,bn,cards,scards,q,e,x):
    dot(dag,q,e,x,fname="polytree.gv")
    print("query: %s evidence: %s abstracted: %s" %(q,e,x))

    inputs = ['v%d'%eid for eid in e]
    output = 'v%d' % q
    alive_evidences = polytreeBN.alloc_alive_evidences(q,e,x,dag) # all testing nodes are active
    print("testing nodes %s" %(alive_evidences,))
    ac_true = TAC(bn,inputs,output,trainable=False)

    # direct sample training set
    ecards = list(map(lambda x: cards[x], e))
    qcard = cards[q]
    num_examples = 1
    for ecard in ecards:
        num_examples *= ecard
    num_examples *= EVIDENCE_FACTOR

    #evidences,marginals = direct_sample(bn,q,e,cards,num_examples=num_examples)
    evidences,marginals = direct_sample_mp(bn,q,e,cards,num_examples=num_examples)
    evidences,marginals = data.evd_hard2lambdas(evidences,ecards), data.marg_hard2lambdas(marginals,qcard)
    prob_evid = polytreeTBN.compute_prob_of_evidence(bn,e,cards)
    prob_evid = prob_evid.flatten()

    # make testing set
    test_evidences = list(iter.product(*list(map(range,ecards)))) # enumerate all possible evidences
    test_evidences = data.evd_hard2lambdas(test_evidences,ecards)
    test_marginals = ac_true.evaluate(test_evidences)

    ''' step 2: make incomplete bn/tbn for learning'''
    bn_learn = make_incomplete_bn(bn,x,cards,scards)
    assert not bn_learn.testing 
    tbn_learn_list = [make_incomplete_tbn2(bn,x,cards,scards,num_intervals=interval) 
        for interval in intervals_learn_list]
    tbn_learn_list = [bn_learn] + tbn_learn_list
    #tac_learn_list = [TAC(tbn_learn,inputs,output,trainable=True,sel_type=SELECT_CPT_TYPE)
        #for tbn_learn in tbn_learn_list]


    ''' step 3: make incomplete tbn by handcraft'''
    print("Start reparam tbn...")
    tbn_hand_list = []
    scards = [(card+1)//2 for card in cards] # lose half states
    cards_map_dict = []
    for card,scard in zip(cards,scards):
        cards_map = get_cards_map(card,scard)
        cards_map_dict.append(cards_map)

    bn_baseline = polytreeTBN.reparam_bn_avg_baseline(dag,bn,q,e,x,cards,scards,cards_map_dict)
    bn_baseline2 = polytreeTBN.reparam_bn_soft_baseline(dag,bn,q,e,x,cards,scards,cards_map_dict)
    reparam_tbn_fun = polytreeTBN.reparam_tbn_fun_wrapper(dag,bn,q,e,x,cards,scards,cards_map_dict)
    with Pool(NUM_WORKERS) as p:
        for tbn_hand in tqdm(p.imap(reparam_tbn_fun, intervals_hand_list),total=len(intervals_hand_list),desc="Reparam TBNs..."):
            tbn_hand_list.append(tbn_hand)
    print("Finish reparam TBNs")

    tbn_hand_list = [bn_baseline,bn_baseline2] + tbn_hand_list
    tac_hand_list = [TAC(tbn_hand,inputs,output,trainable=False,sel_type='threshold')
        for tbn_hand in tbn_hand_list]

    # training
    print("Start training...")
    #ac_incomplete.fit(evidences,marginals,loss_type='CE',metric_type='CE')

    marginals_learn_list = []
    learn_tbn_fun = learn_tbn_fun_wrapper(inputs,output,sel_type=SELECT_CPT_TYPE)
    learn_tbn_fun.add_dataset(evidences,marginals,test_evidences)
    with Pool(NUM_WORKERS) as p:
        for marginals in tqdm(p.imap(learn_tbn_fun, tbn_learn_list),total=len(tbn_learn_list),desc="Learning TBNs..."):
            marginals_learn_list.append(marginals)
            # train tac on dataset and evaluate

    print("Finish learning TBNs")

    # evaluation
    marginals_hand_list = [tac_hand.evaluate(test_evidences) for tac_hand in tac_hand_list]
    #marginals_learn_list = [tac_learn.evaluate(test_evidences) for tac_learn in tac_learn_list]

    kl_hand_list = [KL_divergence(test_marginals,marginals_hand) for marginals_hand in marginals_hand_list]
    kl_learn_list = [cond_KL_divergence(prob_evid,test_marginals,marginals_learn) for marginals_learn in marginals_learn_list]
    
    print("kl hand: %s kl learn: %s" %(kl_hand_list,kl_learn_list))
    return kl_hand_list,kl_learn_list



def do_multiprocess_learn_polytree_tbn_experiment(num_trial):
    print("Start trial %d..." %num_trial)
    ''' step 1: sample true BN and query'''
    random.seed(SEED + num_trial)
    np.random.seed(SEED + num_trial)
    ok = False
    dag,q,e,x = None,None,None,None
    while not ok:
        # search a good query
        dag = model.get_random_polytree(NUM_NODES,NUM_ITERS)
        #dag,q,e,x = model.random_query(dag)
        ok,q,e,x = model.random_query(dag)

    bn,cards = model.sample_random_BN(dag,x) 
    scards = {node_x: cards[node_x]//2 for node_x in x}
    kl_hand_list,kl_learn_list = do_one_experiment(dag,bn,cards,scards,q,e,x)
    return len(x),kl_hand_list,kl_learn_list

'''
def do_learn_polytree_tbn_experiment_for_evidences(num_trial):
    print("Start trial %d..." %num_trial)
    step 1: sample true BN and query
    random.seed(SEED + num_trial)
    np.random.seed(SEED + num_trial)
    success = False
    dag,node_q,nodes_abs = None,None,None
    nodes_evid_list_for_each_size = []
    while not success:
        dag = model.get_random_polytree(NUM_NODES,NUM_ITERS)
        n_nodes = len(dag)
        # choose a query node from leaf nodes
        leaves = [node for node in range(n_nodes) if len(dag[node]) == 0]
        node_q = np.random.choice(leaves)
        ancestors_q = model.ancestors(node_q,dag)
        # choose abstracted nodes from ancestors_q of node q
        ancestors_q = list(ancestors_q)
        if len(ancestors_q) < MIN_NUM_ABSTRACTED_NODES:
            continue

        num_abs_nodes = min(len(ancestors_q), MAX_NUM_ABSTRACTED_NODES)
        # all ancestors or no more than MAX ancestors as abstracted nodes
        nodes_abs = ancestors_q[:num_abs_nodes]
        for evid_size in evidence_size_list:
            ok,nodes_evid_list = model.random_query4(dag,node_q,nodes_abs,evid_size)
            if not ok:
                break
            else:
                nodes_evid_list_for_each_size.append(nodes_evid_list)
        if len(nodes_evid_list_for_each_size) == len(evidence_size_list):
            success = True

    # sample true BN
    bn,cards = model.sample_random_BN(dag,x) 
    scards = {node_x: cards[node_x]//2 for node_x in nodes_abs}
    kl_learn_result = []
    for nodes_evid_list in nodes_evid_list_for_each_size:
        kl_learn_mean = []
        for nodes_evid in nodes_evid_list:
            # for each evidence size, do num_evid_trials
            kl_learn = do_one_experiment(dag,bn,cards,scards,node_q,nodes_evid,nodes_abs)
            kl_learn_mean.append(kl_learn)
        kl_learn_mean = np.mean(np.array(kl_learn_mean),axis=0)
        kl_learn_result.append(kl_learn_mean)

    for evid_size,kl_learn in zip(evidence_size_list,kl_learn_result):
        print("Trial %d  evid size: %d kl learn: %s" %(num_trial, evid_size, kl_learn))

    return kl_learn_result

def do_avg_learn_polytree_tbn_experiment_for_evidences():
    kl_learn_result_list = [[] for _ in range(len(evidence_size_list))]
    f = open("output_%d_for_learn_evidence.txt"%NUM_NODES, mode='w')

    for num_trial in range(NUM_TRIALS):
        kl_learn_result = do_learn_polytree_tbn_experiment_for_evidences(num_trial)
        for list,kl_learn in zip(kl_learn_result_list, kl_learn_result):
            list.append(kl_learn)

        kl_learn_means_list = [np.mean(np.array(kl_learn_result),axis=0) for kl_learn_result in kl_learn_result_list]
        for i, (evid_size,kl_learn_means) in enumerate(zip(evidence_size_list,kl_learn_means_list)):
            print("Trial %d  evid size: %d kl learn: %s" %(num_trial, evid_size, kl_learn_means))
            f.write("Trial %d  evid size: %d kl learn: %s" %(num_trial, evid_size, kl_learn_means))
            f.write('\n')
            f.flush()

    print("Finish experiments for evidences")
    f.close()
    


def do_avg_learn_polytree_tbn_experiment():
    kl_hand_lists = [[] for _ in range(len(intervals_list))]
    kl_learn_lists = [[] for _ in range(len(intervals_list))]
    f = open("output_%d_for_learn_final.txt"%NUM_NODES, mode='w')

    for i in range(NUM_TRIALS):
        kl_hand, kl_learn = do_learn_polytree_tbn_experiment(i)
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
'''

'''
def do_avg_multiprocess_learn_polytree_tbn_experiment():
    kl_hand_lists = [[] for _ in range(len(intervals_list))]
    kl_learn_lists = [[] for _ in range(len(intervals_list))]
    f = open("output_%d_for_learn_final.txt"%NUM_NODES, mode='w')

    with Pool(NUM_WORKERS) as p:
        for i,(kl_hand,kl_learn) in enumerate(tqdm(p.imap(do_learn_polytree_tbn_experiment,range(NUM_TRIALS)),
            total=NUM_TRIALS,desc="Do polytree learning...")):
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
'''

def do_avg_multiprocess_learn_polytree_tbn_experiment2():
    kl_hand_lists = [[] for _ in range(len(intervals_hand_list)+1)]
    kl_learn_lists = [[] for _ in range(len(intervals_learn_list)+1)]
    f = open("output_%d_for_learn_icml.txt"%NUM_NODES, mode='w')

    for i in range(NUM_TRIALS):
        num_abs,kl_hand,kl_learn = do_multiprocess_learn_polytree_tbn_experiment(i)
        for list, kl in zip(kl_hand_lists,kl_hand):
            list.append(kl)
        for list, kl in zip(kl_learn_lists,kl_learn):
            list.append(kl)

        means_hand = [np.array(kl_loss).mean() for kl_loss in kl_hand_lists]
        means_learn =  [np.array(kl_loss).mean() for kl_loss in kl_learn_lists]
        print("Trial %d abstract: %d kl hand: %s kl learn: %s" %(i,num_abs,means_hand,means_learn))
        f.write("Trial %d abstract: %d kl hand: %s kl learn: %s" %(i,num_abs,means_hand,means_learn))
        f.write('\n')
        f.flush()

    print("Finish experiment.")
    f.close()



if __name__ == '__main__':
    #do_learn_polytree_tbn_experiment()
    multiprocessing.set_start_method('spawn')
    #do_avg_learn_polytree_tbn_experiment()
    #do_avg_multiprocess_learn_polytree_tbn_experiment()
    do_avg_multiprocess_learn_polytree_tbn_experiment2()
    #do_learn_polytree_tbn_experiment_for_evidences(0)
    #do_avg_learn_polytree_tbn_experiment_for_evidences()
    






