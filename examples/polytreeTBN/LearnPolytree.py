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


NUM_INTERVALS = 2
# posteriors smaller than this value is ignored

from tbn.tbn import TBN
from tbn.node import Node
from tbn.node2 import NodeV2
import examples.polytreeTBN.model as model 
import examples.polytreeBN.polytree as polytreeBN 
from examples.polytreeTBN.LookUpTable import LookUpTable 
import tbn.cpt as CPT
import train.data as data
from tac import TAC, TACV2



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
def make_incomplete_tbn(bn,nodes_abs,cards_dict,alive_evidences,testing=False,fixed_cpt=False,num_intervals=None):
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
            cpt = node.cpt
            node2 = Node(name,values=values,parents=parents,testing=False,cpt=cpt,fixed_cpt=fixed_cpt)
            tbn.add(node2)

        elif id in alive_evidences.keys() and alive_evidences[id] and testing:
            # for tbn, child of abstracted nodes become testing nodes
            assert num_intervals is not None
            #node2 = NodeV2(name,values=values,parents=parents,testing=True,num_intervals=num_intervals)
            node2 = Node(name,values=values,parents=parents,testing=True)
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

def clip(dist):
    epsilon = np.finfo('float32').eps
    dist_safe = np.where(dist<epsilon, epsilon, dist)
    return dist_safe

def KL_divergence(dist_p,dist_q):
    dist_p = clip(dist_p)
    dist_q = clip(dist_q)
    kl_loss = np.sum(dist_p * np.log(dist_p/dist_q),axis=-1)
    return np.mean(kl_loss)

def do_learn_polytree_tbn_experiment():
    dag = model.get_random_polytree(NUM_NODES,NUM_ITERS)
    dag,q,e,x = model.random_query(dag)
    dot(dag,q,e,x,fname="polytree.gv")
    print("query: %s evidence: %s abstracted: %s" %(q,e,x))

    alive_evidences = polytreeBN.alloc_alive_evidences(q,e,x,dag)
    print("testing nodes %s" %(alive_evidences,))
    print("Start recover experiment...")
    bn,cards = model.sample_random_BN(dag,q,e,x) # sample true bn
    bn_incomplete = make_incomplete_tbn(bn,x,cards,alive_evidences,testing=False,fixed_cpt=FIXED_REGULAR_CPTS) # make incomplete bn
    tbn_incomplete = make_incomplete_tbn(bn,x,cards,alive_evidences,testing=True,fixed_cpt=FIXED_REGULAR_CPTS,
        num_intervals=NUM_INTERVALS) # make incomplete tbn

    # direct sample training set
    ecards = list(map(lambda x: cards[x], e))
    qcard = cards[q]
    evidences,marginals = direct_sample(bn,q,e,cards,num_examples=NUM_EXAMPLES)
    evidences,marginals = data.evd_hard2lambdas(evidences,ecards), data.marg_hard2lambdas(marginals,qcard)

    # compile AC/TAC
    inputs = ['v%d'%eid for eid in e]
    output = 'v%d' % q
    ac_true = TAC(bn,inputs,output,trainable=False)
    ac_incomplete = TAC(bn_incomplete,inputs,output,trainable=True)
    tac_incomplete = TAC(tbn_incomplete,inputs,output,trainable=True,sel_type=SELECT_CPT_TYPE)

    # training
    print("Start training AC...")
    #ac_incomplete.fit(evidences,marginals,loss_type='CE',metric_type='CE')
    ac_incomplete = fit_and_find_best(ac_incomplete,evidences,marginals)
    print("Finish training AC.")
    print("Start training TAC...")
    #tac_incomplete.fit(evidences,marginals,loss_type='CE',metric_type='CE')
    tac_incomplete = fit_and_find_best(tac_incomplete,evidences,marginals)
    print("Finish training TAC...")

    # evaluation
    test_evidences = list(iter.product(*list(map(range,ecards)))) # enumerate all possible evidences
    test_evidences = data.evd_hard2lambdas(test_evidences,ecards)
    test_marginals = ac_true.evaluate(test_evidences)
    marginals_ac = ac_incomplete.evaluate(test_evidences)
    marginals_tac = tac_incomplete.evaluate(test_evidences)
    kl_loss_ac = KL_divergence(test_marginals,marginals_ac)
    kl_loss_tac = KL_divergence(test_marginals,marginals_tac)
    gain = kl_loss_ac / kl_loss_tac
    print("kl loss AC: %.9f KL loss TAC: %.9f gain: %.3f " %(kl_loss_ac,kl_loss_tac,gain))
    print("Finish recover experiment")
    return kl_loss_ac, kl_loss_tac

'''
def fit_and_metric(tbn,inputs,outputs,sel_type,evidences,marginals):
    train_data, val_data = data.random_split(evidences,marginals,percentage=0.2)
    train_evid, train_marg = train_data
    val_evid, val_data = test_data
    tac = TAC(tbn,inputs,output,sel_type=sel_type,trainable=True)
    tac.fit(train_evid,train_marg,loss_type='CE',metric_type='CE') # train AC
    loss = tac.metric(test_evid,test_marg,metric_type='CE')
    return tac, loss
    '''

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




def do_avg_learn_polytree_tbn_experiment():
    bn_loss, tbn_loss = [], []
    f = open("output_learn_%d.txt"%NUM_NODES,'w')
    for i in range(NUM_TRIALS):
        loss_ac, loss_tac = do_learn_polytree_tbn_experiment()
        bn_loss.append(loss_ac)
        tbn_loss.append(loss_tac)
        mean_bn = np.array(bn_loss).mean()
        mean_tbn = np.array(tbn_loss).mean()
        print("Trial %d kl loss bn: %.9f kl loss tbn: %.9f" %(i,mean_bn,mean_tbn))
        f.write("Trial %d kl loss bn: %.9f kl loss tbn: %.9f\n" %(i,mean_bn,mean_tbn))
        f.flush()
    f.close()
    print("Finish experiment.")




if __name__ == '__main__':
    #do_learn_polytree_tbn_experiment()
    do_avg_learn_polytree_tbn_experiment()
    






