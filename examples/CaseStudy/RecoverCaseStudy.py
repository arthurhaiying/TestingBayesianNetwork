from graphviz import Digraph
import numpy as np

import itertools as iter
from functools import reduce

if __name__ == '__main__':
    #basepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    basepath = Path(__file__).resolve().parents[2]
    #print("basepath: %s" %basepath)
    sys.path.append(str(basepath))

from examples.CaseStudy.NetParser import parseBN
from examples.CaseStudy.deleteEdgeOp import *
from tbn.tbn import TBN
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
edges = [
        ('PKA', 'Mek'),
        ('PKA', 'Erk'),
        ('PKA', 'Akt'),
        ('PKC', 'Mek'),
        ('PKC', 'Raf'),
        ('PKC', 'Jnk'),
        ('PKC', 'P38'),
        
    ] 

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
def make_states_map_for_joint_node(node,cards,jcards):
    card = cards[node]
    jcard = jcards[node]
    states_map = [[] for _ in range(card)]
    for jstate in range(len(jcard)):
        states_map[jstate % card].append(jstate)
        # assume that the original state is the last in joint state
    return states_map


def do_recover_bn_experiment():
    bn = parseBN(filename)
    card_dict = {node.name:len(node.values) for node in bn.nodes}
    bn2,joint_state_dict = delete_edge_and_reparam_bn(bn.edges)
    joint_card_dict = {}
    for node,joint_state in joint_state_dict.items():
        jcards = [card_dict[n] for n in joint_state]
        jcard = reduce(lambda x,y: x*y, jcards)
        joint_card_dict[node] = jcard
    
    nodes_joint = {k:v for k,v in joint_state_dict.items() if len(v) >= 2}# joint nodes
    joint_state_dict = {k:v for k,v in joint_state_dict if len(v) >= 2}
    print("joint nodes: %s" %(nodes_joint))
    print("joint states dict: %s" %(joint_state_dict))

    sanity_check(bn,bn2) 
    # check bn2 recovers bn

    states_map_dict = {}
    for jnode in joint_state_dict.items():
        # for each joint node
        states_map = make_states_map_for_joint_node(jnode,card_dict,joint_card_dict)
        states_map_dict[jnode] = states_map

    # reparam bn baseline
