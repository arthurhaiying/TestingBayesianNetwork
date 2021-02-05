from graphviz import Digraph
import numpy as np

import itertools as iter
from functools import reduce
from pathlib import Path
import os,sys

if __name__ == '__main__':
    #basepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    basepath = Path(__file__).resolve().parents[2]
    #print("basepath: %s" %basepath)
    sys.path.append(str(basepath))

from examples.CaseStudy.NetParser import parseBN
from examples.CaseStudy.deleteEdgeOp import *
from examples.polytreeTBN.config import *

import examples.polytreeTBN.polytree as polytreeTBN
import examples.polytreeTBN.LearnPolytree as lpolytreeTBN
from tbn.tbn import TBN
import tbn.cpt as CPT
from tbn.node import Node
import train.data as data
from tac import TAC
import utils.utils as u


# experiment settings
filename = "examples/CaseStudy/sachs.net"
nodes_evid = ['PKC','Jnk','P38']
node_q = 'Akt'
nodes_abs = []


# missing edges
edges1 = [
        ('PKA', 'Mek'),
        ('PKA', 'Erk'),
        ('PKA', 'Akt'),
        ('PKC', 'Mek'),
        ('PKC', 'Raf'),
        ('PKC', 'Jnk'),
        ('PKC', 'P38'),
        
    ] 

edges2 = []


''''
filename = "examples/CaseStudy/asia.net"
nodes_evid = ['bronc','xray']
node_q = 'dysp'
nodes_abs = []

# missing edges
edges = [
        ('bronc','dysp'),
    ]
'''

'''
filename = 'examples/CaseStudy/child.net'

edges1 = [
    ('Disease','Age'),
] 

nodes_prune = ['Age']

edges2 = [
    ('LungParench','Grunting'),
    ('HypoxiaInO2','LowerBodyO2'),
    ('CardiacMixing', 'HypDistrib'),
    ('LungParench','HypoxiaInO2'),
    ('LungParench','ChestXray'),
]
# higher order edges
nodes_evid = ['BirthAsphyxia','GruntingReport','RUQO2','CO2Report','XrayReport','LVH']
node_q = 'LowerBodyO2'
nodes_abs = []
nodes_prune 
'''


def sanity_check(bn,bn2):
    ecards = [len(bn.node(e).values) for e in nodes_evid]
    ac = TAC(bn,inputs=nodes_evid,output=node_q,trainable=False)
    ac2 = TAC(bn2,inputs=nodes_evid,output=node_q,trainable=False)
    #evidences = [[0,2,1]]
    evidences = list(iter.product(*list(map(range,ecards))))
    evidences = data.evd_hard2lambdas(evidences,ecards)
    marginals = ac.evaluate(evidences)
    marginals2 = ac2.evaluate(evidences)
    print("marginals1: %s" %(marginals,))
    print("marginals2: %s" %(marginals2,))
    assert np.allclose(marginals,marginals2)

# make state mapping for joint nodes
# assume that the original state will be the last
def make_states_map(node,cards,jcards):
    card = cards[node]
    jcard = jcards[node]
    states_map = [[] for _ in range(card)]
    for jstate in range(len(jcard)):
        states_map[jstate % card].append(jstate)
        # assume that the original state is the last in joint state
    return states_map


# create map from node name to node id
def node_ids(bn):
    dict = {}
    for i,node in enumerate(bn.nodes):
        # for each node in topo order
        dict[node.name] = i
    return dict

# copy bn for recovery inputs
def copy_for_recovery(bn2):
    ids = node_ids(bn2) # assign ids
    bn3 = TBN(bn2.name+"_copy")
    for node in bn2.nodes:
        id = ids[node.name] # id
        name,values,parents,cpt = node.name,node.values,node.parents,np.copy(node.cpt)
        name = 'v%d'%id
        pids = [ids[p.name] for p in parents]
        parents = [bn3.node('v'+str(pid)) for pid in pids]
        node2 = Node(name,values=values,parents=parents,testing=False,cpt=cpt)
        # copy node
        bn3.add(node2)
    
    return bn3, ids


def do_recover_bn_experiment():
    bn = parseBN(filename)
    bn._name = "CHILD"
    print("query: %s evidence: %s abstracted: %s" %(node_q,nodes_evid,nodes_abs))
    n_nodes = len(bn.nodes)
    dag = get_dag(bn)
    dot(dag,node_q,nodes_evid,nodes_abs,fname="child1.gv")
    card_dict = {node.name:len(node.values) for node in bn.nodes}
    ''' step 1: construct bn2 with miss edges over joint state'''
    #bn2,joint_state_dict = delete_edges_and_reparam_bn(bn,edges)

    bn2,joint_state_dict = delete_edges_and_reparam_bn(bn,edges1)
    # remove higher order edges1

    bn2,joint_state_dict = delete_edges_and_reparam_bn2(bn2,edges2,joint_state_dict)
    # remove higher order edges2

    dag2 = get_dag(bn2)
    dot(dag2,node_q,nodes_evid,nodes_abs,fname="child2.gv")
    joint_card_dict = {}
    for node,joint_state in joint_state_dict.items():
        jcards = [card_dict[n] for n in joint_state]
        jcard = reduce(lambda x,y: x*y, jcards)
        joint_card_dict[node] = jcard


    
    nodes_joint = [k for k,v in joint_state_dict.items() if len(v) >= 2]# joint nodes
    joint_state_dict2 = {k:v for k,v in joint_state_dict.items() if len(v) >= 2}
    print("joint nodes: %s" %(nodes_joint))
    print("joint states dict: %s" %(joint_state_dict2))
    #sanity_check(bn,bn2) 
    print("pass sanity checks!")
    # check bn2 recovers bn

    ''' step 2: convert bn2 to bn3 for recovery inputs'''
    bn3, ids = copy_for_recovery(bn2)
    #test_ac3 = TAC(bn3,inputs=['either'],output='xray',trainable=False)
    #print("bn3 compiles!")
    #exit(0)
    bn3._name = 'child_%s' %SELECT_CPT_TYPE
    dag3 = [None] * n_nodes # create list version dag
    for node,children in dag2.items():
        id = ids[node]
        children = [ids[c] for c in children]
        dag3[id] = children

    q = ids[node_q]
    e = [ids[e] for e in nodes_evid]
    x = [ids[x] for x in nodes_joint] # joint nodes states are actually merged
    print("query: %s evidence: %s abstracted: %s" %(q,e,x))

    cards = [None] * n_nodes
    for node,jcard in joint_card_dict.items():
        id = ids[node]
        cards[id] = jcard

    print("nodes map: %s" %ids)

    scards = [None] * n_nodes
    for node,card in card_dict.items():
        id = ids[node]
        scards[id] = card

    cards_map_dict = []
    for card,scard in zip(cards,scards):
        cards_map = polytreeTBN.get_cards_map(card,scard)
        cards_map_dict.append(cards_map)
    # assume the orignal state for node joint is always the last 


    # do recovery experiment for tbn hand
    print("Start recover tbn...")
    #tbn = polytreeTBN.reparam_tbn(dag3,bn3,q,e,x,cards,scards,cards_map_dict,num_intervals=2)
    kl1, kl2, kl_list = polytreeTBN.do_one_experiment(dag3,bn3,q,e,x,cards,scards,cards_map_dict)
    print("finish recover tbn...")
    exit(0)

    # do learn experiment for tbn learn
    print("Start learn tbn...")
    kl_learn_list = []
    for i in range(NUM_EVID_TRIALS):
        kl_learn = lpolytreeTBN.do_one_experiment(dag3,bn3,cards,scards,q,e,x)
        kl_learn_list.append(kl_learn)
        kl_learn_means = np.mean(np.array(kl_learn_list),axis=0)
        print("trial %d kl learn: %s" %(i,kl_learn_means))
        
    print("final kl_learn: %s" % kl_learn_means)
    print("finish learn tbn.")
    #return kl1, kl2, kl_list, kl_list2


if __name__ == '__main__':
    do_recover_bn_experiment()
    #test_sachs()