from graphviz import Digraph
import numpy as np
import random
import re
from copy import copy,deepcopy
import itertools as iter
from functools import reduce

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
from examples.CaseStudy.deleteEdgeOp import *
from examples.CaseStudy.RecoverCaseStudy import copy_for_recovery


# posteriors smaller than this value is ignored

from tbn.tbn import TBN
from tbn.node import Node
from tbn.node2 import NodeV2
import examples.polytreeTBN.model as model 
import examples.polytreeTBN.polytree as polytreeTBN
import examples.polytreeTBN.LearnPolytree as learnPolytreeTBN
import tbn.cpt as CPT
import train.data as data
from tac import TAC, TACV2

from compile.opsgraph import OpsGraph as og
og.USE_FIXED_THRESHOLDS = USE_FIXED_THRESHOLDS
print("Using fixed threshold: %s" % USE_FIXED_THRESHOLDS)


def sample_transition_cpt(order,card):
    arrays = []
    cards = (card,)*order
    # sample random cond dist
    def __random_dist(length,value):
        if length == 1:
            return [value]
        else:
            MIN_POS_RATE = 0.8
            pos = np.random.uniform(low=MIN_POS_RATE,high=1.0)
            pos = pos*value
            return [pos]+ __random_dist(length-1,value-pos)
    for _ in iter.product(*list(map(range,cards))):
        array = __random_dist(card,1.0)
        random.shuffle(array)
        arrays.append(array)
    shape = tuple(cards) + (card,)
    cpt = np.array(arrays).reshape(shape)
    return cpt

def sample_emission_cpt(card):
    MIN_POS_VALUE = 0.99
    pos = random.uniform(MIN_POS_VALUE,1.0)
    neg = (1.0-pos)/(card-1)
    cpt = np.ones((card,card)) * neg
    for i in range(card):
        cpt[i][i] = pos
        # set diagonal of cpt
    return cpt

def get_HMM_net(size,order,card):
    bn = TBN(f'HMM_1')
    values = ['v'+str(i) for i in range(card)]
    transition = sample_transition_cpt(order,card)
    emission = sample_emission_cpt(card)
    for i in range(order):
        # create intial hidden nodes
        name = 'h_'+str(i)
        parents = ['h_'+str(j) for j in range(i)]
        pnodes = [bn.node(p) for p in parents]
        cpt = np.ones((card,)*(i+1)) * (1.0/card)
         # use uniform initial CPT
        h_i = Node(name, values=values, parents=pnodes, testing=False, cpt=cpt)
        bn.add(h_i)
    for i in range(order, size):
        # create subsequent hidden nodes
        name = 'h_'+str(i)
        parents = ['h_'+str(j) for j in range(i-order, i)]
        pnodes = [bn.node(p) for p in parents]
        h_i = Node(name, values=values, parents=pnodes, testing=False, cpt=transition, cpt_tie="transition")
        bn.add(h_i)
    # create evidence nodes
    for i in range(size):
        name = 'e_'+str(i)
        parents = ['h_'+str(i)]
        pnodes = [bn.node(p) for p in parents]
        e_i = Node(name, values=values, parents=pnodes, testing=False, cpt=emission, cpt_tie="emission")
        bn.add(e_i)
    # finish creating hmm
    #tbn.dot(view=True)
    return bn


def make_fixed_thresholds(pcards,num_intervals):
    step = 1.0/num_intervals
    thress = [step *(i+1)* np.ones(pcards) for i in range(num_intervals-1)]
    return thress

def get_test_HMM_net(size,order,card,num_intervals):
    tbn = TBN(f'test_HMM_{num_intervals}')
    values = ['v'+str(i) for i in range(card)]
    for i in range(order):
        # create intial hidden nodes
        name = 'h_'+str(i)
        parents = ['h_'+str(j) for j in range(i)]
        pnodes = [tbn.node(p) for p in parents]
        cpt = np.ones((card,)*(i+1)) * (1.0/card)
        # use uniform initial CPT
        h_i = Node(name, values=values, parents=pnodes, testing=False, cpt=cpt)
        tbn.add(h_i)
    for i in range(order, size):
        # create subsequent hidden nodes
        name = 'h_'+str(i)
        parents = ['h_'+str(j) for j in range(i-order, i)]
        pnodes = [tbn.node(p) for p in parents]
        pcards = (card,)*order
        num_cpts = num_intervals
        num_thress = num_intervals-1
        cpts = [CPT.random(card,pcards) for i in range(num_cpts)]
        if not USE_FIXED_THRESHOLDS:
            thress = [np.random.rand(*pcards) for i in range(num_thress)]
        else:
            print("Using fixed thresholds!")
            thress = make_fixed_thresholds(pcards,num_intervals=num_intervals)
        h_i = NodeV2(name,values=values,parents=pnodes,testing=True,cpts=cpts,thresholds=thress,cpt_tie="transition",
            num_intervals=num_intervals)
        tbn.add(h_i)
    # create evidence nodes
    for i in range(size):
        name = 'e_'+str(i)
        parents = ['h_'+str(i)]
        pnodes = [tbn.node(p) for p in parents]
        pcards = (card,)
        num_cpts = num_intervals
        num_thress = num_intervals-1
        cpts = [CPT.random(card,pcards) for i in range(num_cpts)]
        thress = [np.random.rand(*pcards) for i in range(num_thress)]
        e_i = NodeV2(name, values=values, parents=pnodes, testing=True, cpts=cpts, cpt_tie="emission",
            num_intervals=num_intervals)
        tbn.add(e_i)
    # finish creating hmm
    #tbn.dot(view=True)
    return tbn

# direct sample instances (nodes_evid, node_q) from bn
# return: 
#   evidences -- a list of list representing hard evidences for nodes_evid (num_examples, evid_size)
#   marginals -- a list representing hard evidence for node_q (num_examples,)
def direct_sample(bn,node_q,nodes_evid,num_examples):
    print("Start sampling dataset of %d examples..." % num_examples)
    evid_size = len(nodes_evid)
    evidences, marginals = [],[]

    for _ in range(num_examples):
        sample = {}
        for node in bn.nodes:
            # for each node in topo order
            name = node.name
            card = len(node.values)
            parents,cpt = node.parents,node.cpt
            parents = [p.name for p in parents]
            pvalues = [sample[p] for p in parents] 
            # parents are already sampled
            cond = cpt[tuple(pvalues)]
            if np.isclose(np.sum(cond),1.0):
                cond = cond / np.sum(cond)

            try:
                value = np.random.choice(np.arange(card),p=cond)
                sample[name] = value
            except ValueError:
                print("node %s has cpt do not sum to 1" %id)
                print("parents: %s" %pvalues)
                print("probability: %s" %cond)
                exit(1)


        evid = [sample[e] for e in nodes_evid] # get evidence value
        query = sample[node_q] # get query vale
        evidences.append(evid)
        marginals.append(query)

    # sampling done
    print("Finish sampling dataset")
    assert np.array(evidences).shape == (num_examples,evid_size)
    assert np.array(marginals).size == num_examples
    return evidences,marginals

class direct_sample_fun_wrapper:
    def __init__(self,bn,node_q,nodes_evid,num_examples):
        self.args = [bn,node_q,nodes_evid,num_examples]
    def __call__(self,num):
        np.random.seed(SEED+num) # make sure different examples 
        return direct_sample(*self.args)


def direct_sample_mp(bn,node_q,nodes_evid,num_examples):
    print("Start sampling dataset of %d examples..." % num_examples)
    NUM_SAMPLE_WORKERS = 10
    evidences,marginals = [],[]
    direct_sample_fun = direct_sample_fun_wrapper(bn,node_q,nodes_evid,num_examples//NUM_SAMPLE_WORKERS)
    with Pool(NUM_SAMPLE_WORKERS) as p:
        for evidence,marginal in tqdm(p.imap(direct_sample_fun, list(range(NUM_SAMPLE_WORKERS))),
            total=NUM_SAMPLE_WORKERS,desc="Sample data..."):
            evidences.extend(evidence)
            marginals.extend(marginal)
    assert len(evidences) == len(marginals)
    print("Finish sampling dataset of %d examples." %len(evidences))
    return evidences,marginals


def prob_of_evidence(bn,nodes_evid,ecards):
    bn1 = deepcopy(bn)
    single = Node(name='single',parents=[],testing=False) # disconnected evidence
    bn1.add(single)

    inputs = ['single']
    outputs = nodes_evid
    ac = TACV2(bn1,inputs,outputs,trainable=False)
    num_examples = 2
    evidences = data.evd_random(size=num_examples,cards=[2],hard_evidence=True)
    marginals = ac.evaluate(evidences)
    assert marginals.shape == (2,)+tuple(ecards)
    assert np.allclose(marginals[0],marginals[1])
    return marginals[0]

def do_learn_hmm_experiment(hmm,size,card):

    assert not hmm.testing # true bn is not testing
    assert size == len(hmm.nodes) // 2
    assert card == len(hmm.nodes[0].values)

    nodes_evid = ['e_%d'%i for i in range(size-1)]
    node_q = 'h_%d' % (size-1)
    inputs = nodes_evid
    output = node_q
    ac_true = TAC(hmm,inputs,output,trainable=False)

    # step 1: sample dataset
    num_examples = NUM_EXAMPLES
    ecards = [card]*len(nodes_evid)
    qcard = card
    #evidences,marginals = direct_sample(hmm,node_q,nodes_evid,num_examples=num_examples)
    evidences,marginals = direct_sample_mp(hmm,node_q,nodes_evid,num_examples=num_examples)
    evidences,marginals = data.evd_hard2lambdas(evidences,ecards), data.marg_hard2lambdas(marginals,card)
    prob_evid = prob_of_evidence(hmm,nodes_evid,ecards)
    prob_evid = prob_evid.flatten()

    # make testing set
    test_evidences = list(iter.product(*list(map(range,ecards)))) # enumerate all possible evidences
    test_evidences = data.evd_hard2lambdas(test_evidences,ecards)
    test_marginals = ac_true.evaluate(test_evidences)

    ''' step 2: make incomplete bn/tbn for learning'''
    hmm_learn = get_HMM_net(size,order=1,card=card)
    assert not hmm_learn.testing 
    test_hmm_learn_list = [get_test_HMM_net(size,order=1,card=card,num_intervals=interval)
        for interval in intervals_learn_list]
    test_hmm_learn_list = [hmm_learn] + test_hmm_learn_list
    
     # training
    print("Start training...")
    #ac_incomplete.fit(evidences,marginals,loss_type='CE',metric_type='CE')

    marginals_learn_list = []
    learn_tbn_fun = learnPolytreeTBN.learn_tbn_fun_wrapper(inputs,output,sel_type=SELECT_CPT_TYPE)
    learn_tbn_fun.add_dataset(evidences,marginals,test_evidences)
    with Pool(NUM_WORKERS) as p:
        for marginals in tqdm(p.imap(learn_tbn_fun, test_hmm_learn_list),total=len(test_hmm_learn_list),desc="Learning hmms..."):
            marginals_learn_list.append(marginals)
            # train tac on dataset and evaluate

    print("Finish learning TBNs")

    # evaluation
    kl_learn_list = [learnPolytreeTBN.cond_KL_divergence(prob_evid,test_marginals,marginals_learn) for marginals_learn in marginals_learn_list]
    print("kl learn: %s" %(kl_learn_list,))
    return kl_learn_list


def do_HMM_experiment(size,order,card,num_trial):
    random.seed(SEED+num_trial)
    np.random.seed(SEED+num_trial)
    print("Do hmm of size {} order {} and card {}".format(size,order,card))
    hmm = get_HMM_net(size,order,card)
    edges = []
    for dst in range(order):
        for src in range(0,dst-1):
            edges.append(('h_'+str(src),'h_'+str(dst)))
    for dst in range(order,size):
        for src in range(dst-order,dst-1):
            edges.append(('h_'+str(src),'h_'+str(dst)))
    print("high order edges: %s" %edges)
    edges = edges[::-1]
    # step 1: do recovery experiment
    #kl_hand_list = do_recover_HMM_experiment(hmm,size,order,card,edges,node_q,nodes_evid)

    # step 2: do learn experiment
    kl_learn_list = do_learn_hmm_experiment(hmm,size,card)
    return kl_learn_list

def do_avg_recover_HMM_experiment(size,order,card):
    fixed = "fixed" if USE_FIXED_THRESHOLDS else "free"
    #kl_hand_lists = [[] for _ in range(len(intervals_hand_list)+2)]
    kl_learn_lists = [[] for _ in range(len(intervals_learn_list)+1)]
    f = open("output_size_%d_order_%d_card_%d_for_hmm_%s.txt"%(size,order,card,fixed), mode='w')

    for i in range(NUM_TRIALS):
        kl_learn = do_HMM_experiment(size,order,card,i)
        #for list, kl in zip(kl_hand_lists,kl_hand):
            #list.append(kl)
        for list, kl in zip(kl_learn_lists,kl_learn):
            list.append(kl)

        #means_hand = [np.array(kl_loss).mean() for kl_loss in kl_hand_lists]
        means_learn =  [np.array(kl_loss).mean() for kl_loss in kl_learn_lists]
        print("Trial %d size: %d  kl learn: %s" %(i,size,means_learn))
        f.write("Trial %d size: %d  kl learn: %s" %(i,size,means_learn))
        f.write('\n')
        f.flush()

    print("Finish experiment.")
    f.close()


if __name__ == '__main__':
    print("Start hmm experiment...")
    multiprocessing.set_start_method('spawn')
    #do_HMM_experiment(size=SIZE,order=ORDER,card=CARD,num_trial=0)
    do_avg_recover_HMM_experiment(size=SIZE,order=ORDER,card=CARD)
    print("finish hmm experiment.")







