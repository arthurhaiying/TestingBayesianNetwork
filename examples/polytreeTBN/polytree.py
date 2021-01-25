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
NUM_INTERVALS = 50
ZERO_POSTERIOR_VALUE = 0.0
# posteriors smaller than this value is ignored

from tbn.tbn import TBN
from tbn.node import Node
from tbn.node2 import NodeV2
from examples.polytreeTBN.model import *
import examples.polytreeBN.polytree as polytreeBN 
from examples.polytreeTBN.LookUpTable import LookUpTable, LookUpTableV2 
import tbn.cpt as CPT
import train.data as data
from tac import TAC, TACV2

# mark children of an abstracted node as testing nodes
# assume that each testing node Y is ordered with (id, child_id) where id is the index of its abstracted parent X
# and child id is the index of Y within the children of X, both wrt to parent X with largest id 
# return a list of testing nodes in this order
def testing_order(dag,node_q,nodes_abs):
    topo = topo_sort(dag)
    nodes_abs = [node for node in topo if node in nodes_abs] 
    # sort abstracted nodes
    ancestors_q = ancestors(node_q,dag)
    ancestors_q.append(node_q)
    nodes_testing = []
    for i,node_x in enumerate(nodes_abs):
        rightmost_y = None
        for node_y in dag[node_x]:
            # for each child Y of X
            if node_y not in ancestors_q:
                nodes_testing.append(node_y)
            else:
                rightmost_y = node_y

        # for rightmost child Y
        abs_parents = get_parent_list(dag)[rightmost_y]
        abs_parents = [p for p in abs_parents if p in nodes_abs]
        if all(p in nodes_abs[:i+1] for p in abs_parents):
            # children of all abstracted parents of Y have been added
            nodes_testing.append(rightmost_y)

    return nodes_testing
                
# allocate the set of alive evidences for each testing node with reference to Q in dag
# dag: a list of list representing directed adjacency in polytree dag
# (q,e,x): query node/evidence nodes/abstracted nodes
# return: active_evid_dict -- a dict of list representing the accumulative alive evidence for each testing node
def alloc_active_evidences(dag,nodes_evid,nodes_testing):
    add_order = add_order_for_tbn(dag,nodes_testing)
    parent_list = get_parent_list(dag)
    active_evid_dict = {}
    dag_acc = {}
    evid_acc = set()

    for node in add_order:
        # add one node to dag
        if node in nodes_testing:
            conn_nodes = set()
            for p in parent_list[node]:
                conn_nodes |= set(connected_nodes(p,dag_acc))
            #print("nodes connected to {}: {}".format(node,conn_nodes))
            active_evid = conn_nodes & evid_acc
            #alive_evid = evid_acc
            active_evid_dict[node] = list(active_evid)
        if node in nodes_evid:
            evid_acc.add(node)

        dag_acc[node] = []
        for p in parent_list[node]:
            dag_acc[p].append(node)

    return active_evid_dict


    


# return a topological ordering of nodes in dag such that 1) place non-testing nodes before testing nodes
# and 2) place testing nodes according to given order
# dag: a list of list representing directed adjacencies of DAG
from queue import PriorityQueue
def add_order_for_tbn(dag,nodes_testing):
    n_nodes = len(dag)
    in_degrees = [0]*n_nodes
    # assign priorities for each node
    priority = {node:-1 for node in range(n_nodes) if node not in nodes_testing}
    # non-testing nodes have higher priority 
    for i,node_t in enumerate(nodes_testing):
        priority[node_t] = i

    # compute in-degrees of each node in dag
    for node in range(n_nodes):
        for child in dag[node]:
            in_degrees[child]+=1
    
    pq = PriorityQueue()
    order = []
    roots = [node for node in range(n_nodes) if in_degrees[node] == 0] # roots
    for root in roots:
        pq.put((priority[root], root))

    while not pq.empty():
        _, curr = pq.get()
        order.append(curr)
        for child in dag[curr]:
            in_degrees[child]-=1
            # decrement in-degrees for each child
            if in_degrees[child] == 0:
                pq.put((priority[child], child))

    return order
            
            


# return mapping from original states to superstates
# states_map - a list of list representing the original states contained in each super-state
def get_cards_map(card, scard):
    cards_map = [[] for _ in range(scard)]
    for i in range(card):
        cards_map[i % scard].append(i)
    return cards_map

# assume that node name is v+str(id)
def node_id(node):
    return int(re.sub('\D','',node.name))


# compile tac for evaluating parent posterior on pids pr(x'v ||e) in tbn given evidence on eids
# be careful that some parents v can be evidence in eids; 
# in this case pr(x'v) is assigned zero if v is not consistent with given evidence
# return: 
#  tac - TAC for computing the parent posterior
#  ___evaluate_fn - a function that takes hard evidence as input, evaluates on tac, 
#                   and output np array representing parent posterior in shape (x',v)
def prepare_tac_for_parent_posterior(tbn,pids,eids,nodes_abs,cards_dict,scards_dict):
    abs_pids = [pid for pid in pids if pid in nodes_abs] # abstracted parents
    evd_pids = [] # some parents may be evidence
    for pid in pids:
        if pid in eids:
            assert pid not in nodes_abs
            evd_pids.append(pid)
    ecards = [cards_dict[eid] for eid in eids]
    pcards = []
    for pid in pids:
        if pid not in abs_pids:
            pcards.append(cards_dict[pid])
        else:
            pcards.append(scards_dict[pid])

    if not evd_pids:
        # base case: no evidence on parents
        inputs = ['v%d'%eid for eid in eids]
        outputs = ['v%d'%pid for pid in pids]
        tac = TACV2(tbn,inputs,outputs,sel_type='threshold',trainable=False)
        def __evaluate_fn(evidence):
            # evaluate tac on hard evidence
            evidence = np.array(evidence)
            print("evidence: %s" %evidence)
            lambdas = data.evd_hard2lambdas([evidence],ecards) # convert to tac input
            print("tac input: %s" %lambdas)
            marginals = tac.evaluate(lambdas)[0]
            assert marginals.shape == tuple(pcards)
            return marginals

        return __evaluate_fn

    else:
        # some parents v are also evidence
        inputs = ['v%d'%eid for eid in eids]
        outputs,output_cards = [],[]
        for i,pid in enumerate(pids):
            if pid not in evd_pids:
                outputs.append('v%d' % pid)
                output_cards.append(pcards[i])

        # exclude parents that are also evidence from outputs
        tac = TACV2(tbn,inputs,outputs,sel_type='threshold',trainable=False)
        def __evaluate_fn(evidence):
            # evaluate tac on evidence and expand to original posterior shape
            res = np.zeros(tuple(pcards))
            print("evidence: %s" % evidence)
            lambdas = data.evd_hard2lambdas([evidence],ecards) # convert to tac input
            print("tac input: %s" % lambdas)
            marginals = tac.evaluate(lambdas)[0]
            assert marginals.shape == tuple(output_cards)
            # assign to posterior pr(x',v) where v is consistent with evidence
            p_axes,p_values = [],[]
            for pid in evd_pids:
                # for parents v that are also evidence
                p_axes.append(pids.index(pid))
                p_values.append(evidence[eids.index(pid)])

            my_slc = [slice(None)]*len(pids)
            for axis,value in zip(p_axes,p_values):
                my_slc[axis] = value

            # index into result marginals
            res[tuple(my_slc)] = marginals
            return res

        return __evaluate_fn
            
# compile tac for evaluating the cond cpt for node pr(y|x've) in true bn given evidence on eids 
# becareful that some parents v might be evidence in eids
# and this node may be abstracted
# return:
#   tac: tac for computing the cond cpt pr(y|x'v,e) given evidence on e
#   __evaluate_fn: a function that takes hard evidence on eids as input and 
#                  output np array representing cond cpt pr(y|x've) in shape (x',v, y)
def prepare_tac_for_cond_cpts(bn,node,pids,eids,nodes_abs,cards_dict,scards_dict,cards_map_dict):
    card = cards_dict[node_id(node)]
    abs_pids = [pid for pid in pids if pid in nodes_abs] # abstracted parents
    evd_pids = [] # some parents may be evidence
    for pid in pids:
        if pid in eids:
            assert pid not in nodes_abs
            evd_pids.append(pid)

    ecards = [cards_dict[eid] for eid in eids]
    pcards = []
    for pid in pids:
        if pid not in abs_pids:
            pcards.append(cards_dict[pid])
        else:
            pcards.append(scards_dict[pid])

    if not evd_pids:
        # base case: no evidence on parents
        input_ids = pids + eids
        inputs = ['v%d'%id for id in input_ids]
        output = 'v%d'% node_id(node)
        ac = TAC(bn,inputs,output,trainable=False)
        # compile AC for evaluating posterior on y pr(y|xv'e)


    else:
        # some parents v are also evidence
        input_ids = [pid for pid in pids if pid not in evd_pids]
        input_ids += eids
        inputs = ['v%s'%id for id in input_ids]
        # exclude parents that are also evidence from inputs
        output = 'v%s' % node_id(node)
        ac = TAC(bn,inputs,output,trainable=False)
        # compile ac for evaluating the conditional cpt pr(y|x'v,e) given evidence on e

    def __evaluate_fn(evidence):
        # takes hard evidence and evaluate tac for cond cpt pr(y|x'v, e)
        # if v is not consistent with give evidence e, then use the value of v in e to override
        num_parent_states = 1
        for pcard in pcards:
            num_parent_states *= pcard
        if len(evidence)==0:
            evid_lambdas = []
        else:
            evidence = np.array(evidence)
            evidence_list = np.broadcast_to(evidence,shape=(num_parent_states,len(eids)))
            evid_lambdas = data.evd_hard2lambdas(evidence_list,ecards)
            # convert evidence on eids to tac inputs
        rows = [] 
        # a list of rows: each correspond to one super parents instantiation
        for pvalues in iter.product(*list(map(range,pcards))):
            # for each super parents instantiation (x',v)
            row = [] 
            for pid,pvalue in zip(pids,pvalues):
                if pid not in nodes_abs:
                    # if non abstracted parents
                    if pid in evd_pids:
                        continue
                        # state of this pid is fixed
                    pcard = cards_dict[pid]
                    lambda_ = np.zeros(pcard)
                    lambda_[pvalue] = 1.0
                    row.append(lambda_)
                else:
                    # if abstracted parents, use soft evidence 
                    pcard = cards_dict[pid]
                    values = cards_map_dict[pid][pvalue] 
                    # original values contained in superstate pvalue
                    lambda_ = np.zeros(pcard)
                    lambda_[np.array(values)] = 1.0/len(values)
                    row.append(lambda_)
            # row for one super parents instantiation     
            rows.append(row)

        parent_lambdas = data.evd_row2col(rows)
        lambdas = parent_lambdas+evid_lambdas
        marginals = ac.evaluate(lambdas)
        assert marginals.shape == (num_parent_states,card)
        shape = tuple(pcards) + (card,)
        marginals = marginals.reshape(shape)
        # return cond cpt pr(y|xv'e) for given evidence e
        if node_id(node) in nodes_abs:
            # if y is abstracted
            scard = scards_dict[node_id(node)]
            cards_map = cards_map_dict[node_id(node)]
            arrays = [np.sum(marginals[...,np.array(cards)], axis=-1, keepdims=True) for cards in cards_map]
            marginals = np.concatenate(arrays,axis=-1)
            assert marginals.shape == tuple(pcards)+(scard,)

        return marginals
        
    return __evaluate_fn

def compute_prob_of_evidence(bn,eids,cards_dict):
    bn1 = deepcopy(bn)
    single = Node(name='single',parents=[],testing=False) # disconnected evidence
    bn1.add(single)

    inputs = ['single']
    outputs = ['v%d'%eid for eid in eids]
    ecards = [cards_dict[eid] for eid in eids]
    ac = TACV2(bn1,inputs,outputs,trainable=False)
    num_examples = 2
    evidences = data.evd_random(size=num_examples,cards=[2],hard_evidence=True)
    marginals = ac.evaluate(evidences)
    assert marginals.shape == (2,)+tuple(ecards)
    assert np.allclose(marginals[0],marginals[1])
    return marginals[0]
    


# reparam testing node according to polytree policy 
# parameters:
# node -- the current node (in true bn)
# bn -- true bn
# tbn -- incomplete tbn that is being built
# num_intervals -- number of intervals for testing cpt of this node
# return: a testing node over incomplete cards is added into tbn
def reparam_testing_node(node,bn,tbn,nodes_abs,nodes_t,cards_dict,scards_dict,cards_map_dict,active_evidences,num_intervals):
    id = node_id(node) # id
    assert id in nodes_t
    print("Adding testing node %s..." %(id,))
    name,values,parents = node.name,node.values,node.parents
    parents = list(map(lambda x: tbn.node(x.name), parents))
    card = cards_dict[id] # original states
    scard, svalues = card, values
    if id in nodes_abs:
        # if this node is also abstracted
        print("Testing node %s is also abtracted!" %(id,))
        scard = scards_dict[id] # super states
        cards_map = cards_map_dict[id]
        svalues = ['state_' + ','.join(map(str,states)) for states in cards_map]

    #node_t = NodeV2(name,values=values,parents=parents,testing=True,num_intervals=num_intervals)

    pids = list(map(node_id, parents)) # id of parents
    eids = active_evidences[id] # active evidence for cpt selecting of this testing node
    if not eids:
        # if no active evidence for testing node
        # reparam cond cpt as pr(y|x'v)
        print("Testing node %d is not active: add as non testing node." %(id,))
        ecards = []
        pcards = [] # super parent states
        for pid in pids:
            if pid not in nodes_abs:
                pcards.append(cards_dict[pid])
            else:
                pcards.append(scards_dict[pid])
        cond_cpt_tac = prepare_tac_for_cond_cpts(bn,node,pids,eids,nodes_abs,cards_dict,scards_dict,cards_map_dict)
        fnull = open(os.devnull,'w')
        sys.stdout, fnull = fnull, sys.stdout
        cond_cpts = cond_cpt_tac([]) # get cpt pr(y|x'v)
        sys.stdout, fnull = fnull, sys.stdout
        assert cond_cpts.shape == tuple(pcards)+(scard,)
        # becareful that current node might be abstracted
        node2 = Node(name,values=svalues,parents=parents,testing=False,cpt=cond_cpts)
        tbn.add(node2)
        print("Finish adding testing node %d." %(id,))
        return

    pcards = [] # super parent states
    for pid in pids:
        if pid not in nodes_abs:
            pcards.append(cards_dict[pid])
        else:
            pcards.append(scards_dict[pid])

    # initialize an array of (x',v) lookup tables, each corresponding to one super parents instantiation 
    testing_cpts = np.empty(tuple(pcards), dtype=object)
    for index in np.ndindex(*pcards):
        testing_cpts[index] = LookUpTableV2(size=scard,num_intervals=num_intervals)
        #testing_cpts[index] = LookUpTable(size=scard,num_intervals=num_intervals)
        # initialize LookUpTable (thres -> cond cpt) for each super parent instantiation

    fnull = open(os.devnull,'w')
    sys.stdout,fnull = fnull, sys.stdout
    #prob_e = compute_prob_of_evidence(tbn,eids,cards_dict)
    prob_e = compute_prob_of_evidence(bn,eids,cards_dict)
    ppost_tac = prepare_tac_for_parent_posterior(tbn,pids,eids,nodes_abs,cards_dict,scards_dict)
    # compile tac for computing parent posterior pr(x'v||e) on current tbn
    cond_cpt_tac = prepare_tac_for_cond_cpts(bn,node,pids,eids,nodes_abs,cards_dict,scards_dict,cards_map_dict)
    # compile tac for computing the evidence-cond cpt pr(y|x'v, e) on true bn
    sys.stdout,fnull = fnull, sys.stdout

    # enumerate evidence instantiation
    ecards = [cards_dict[eid] for eid in eids]
    assert prob_e.shape == tuple(ecards)
    for evidence in iter.product(*list(map(range,ecards))):
        # for each possible evidence instantiation
        evidence = np.array(evidence)
        likelihood = prob_e[tuple(evidence)]
        #print("likelihood shape: %s" %(likelihood.shape,))
        fnull = open(os.devnull,'w')
        sys.stdout,fnull = fnull, sys.stdout
        posteriors = ppost_tac(evidence) # pr(x'v|e)
        cond_cpts = cond_cpt_tac(evidence) # pr(y|x'v,e)
        sys.stdout,fnull = fnull,sys.stdout
        assert posteriors.shape == tuple(pcards)
        try:
            assert cond_cpts.shape == tuple(pcards) + (scard,)
        except AssertionError:
            print("cond cpt shape: %s should be %s" %(cond_cpts.shape,tuple(pcards) + (scard,)))
            exit(1)
        assert np.allclose(np.sum(cond_cpts,axis=-1), 1.0) # check that cond cpt normalized
        for pvalues in iter.product(*list(map(range,pcards))):
            # for each super parent state 
            #  evidence -> posterior, evidence -> cond_cpt, map posterior to this cond_cpt
            posterior = posteriors[pvalues]
            cond_cpt = cond_cpts[pvalues]
            weight = likelihood*posterior
            #print("weight shape: %s" %(weight.shape,))
            if np.isclose(posterior, ZERO_POSTERIOR_VALUE):
                continue
            lut = testing_cpts[pvalues]
            #lut[posterior] = cond_cpt # for look up table v1
            lut[posterior] = (weight,cond_cpt) # for look up table v2
            # use this cond_cpt of posteior fall into this interval

    # export thresholds and cpts from lookup tables
    thresholds = [np.zeros(tuple(pcards)) for i in range(num_intervals-1)]
    cpts = [np.zeros(tuple(pcards)+(scard,)) for i in range(num_intervals)]
    num_conflicts = 0
    for index,lut in np.ndenumerate(testing_cpts):
        # for each (super parent value, look up table)
        thres = lut.thresholds()
        conds = lut.cond_cpts()
        for i in range(num_intervals-1):
            thresholds[i][index] = thres[i]
        for i in range(num_intervals):
            cpts[i][index] = conds[i]
        num_conflicts += lut.getNumConflicts()

    print("Find %d cpt conflicts for node %d." %(num_conflicts,id))

    # create a testing node with testing cpts 
    node_t = NodeV2(name,values=svalues,parents=parents,testing=True,
        cpts=cpts,thresholds=thresholds,num_intervals=num_intervals)
    tbn.add(node_t)
    print("Finish adding testing node %s." %(id,))

# reparam a non-testing node (abstracted or not abstracted) and add to tbn
# parameters:
#   node -- current node (in true bn) which is not testing in tbn
#   bn -- true bn
#   tbn -- incomplete tbn
def reparam_non_testing_node(node,bn,tbn,nodes_abs,nodes_t,cards_dict,scards_dict,cards_map_dict):
    id = node_id(node)
    assert id not in nodes_t
    # non testing node
    print("Adding node %s..." % (id,))
    name,values,parents,cpt = node.name,node.values,node.parents,node.cpt
    parents = list(map(lambda x: tbn.node(x.name), parents))
    card = cards_dict[id]
    if id in nodes_abs:
        # if abstracted nodes
        scard = scards_dict[id]
        cards_map = cards_map_dict[id]
        svalues = ['state_' + ",".join(map(str,cards)) for cards in cards_map]
        
    pids = [node_id(p) for p in parents]
    assert all(pid not in nodes_abs for pid in pids)
    # non testing node cannot have abstracted parents

    if id not in nodes_abs:
        cpt = np.copy(cpt)
        node2 = Node(name,values=values,parents=parents,testing=False,cpt=cpt)
        tbn.add(node2)
        print("Finish adding node %s." %(id,))

    else:
        # if abstracted node, sum cpt on last axis
        arrays = [np.sum(cpt[...,tuple(cards)],axis=-1,keepdims=True) for cards in cards_map]
        cpt = np.concatenate(arrays,axis=-1)
        node2 = Node(name,values=svalues,parents=parents,testing=False,cpt=cpt)
        tbn.add(node2)
        print("Finish adding node %s" %(id,))

def reparam_node_avg_baseline(node,bn,bn2,nodes_abs,nodes_t,cards_dict,scards_dict,cards_map_dict):
    id = node_id(node)
    # non testing node
    print("Adding node %s..." % (id,))
    name,values,parents,cpt = node.name,node.values,node.parents,node.cpt
    parents = list(map(lambda x: bn2.node(x.name), parents))
    card = cards_dict[id]
    if id in nodes_abs:
        # if not abstracted node
        scard = scards_dict[id]
        cards_map = cards_map_dict[id]
        svalues = ['state_' + ",".join(map(str,cards)) for cards in cards_map]
        #if len(svalues) == 1:
            #print("super cards is one for node %d." %(id,))
            #exit(1)
        
    if id not in nodes_abs and id not in nodes_t:
        # base case: not abstracted node or children of abstracted node
        # just copy node
        cpt = np.copy(cpt)
        node2 = Node(name,values=values,parents=parents,testing=False,cpt=cpt)
        bn2.add(node2)
        print("Finish adding node %s." %(id,))
    
    elif id not in nodes_t:
        # base case: abstracted node. Need to sum cpt
        arrays = [np.sum(cpt[...,tuple(cards)],axis=-1,keepdims=True) for cards in cards_map]
        cpt = np.concatenate(arrays,axis=-1)
        node2 = Node(name,values=svalues,parents=parents,testing=False,cpt=cpt)
        bn2.add(node2)
        print("Finish adding node %s" %(id,))
    
    else:
        # child of abstracted node. Can be abstracted node itself
        pids = [node_id(p) for p in parents]
        abs_pids = [pid for pid in pids if pid in nodes_abs]
        assert abs_pids # must have one abstracted parent
        pcards = []
        for pid in pids:
            if pid not in abs_pids:
                pcards.append(cards_dict[pid])
            else:
                pcards.append(scards_dict[pid])

        shape = tuple(pcards) + (card,)
        avg_cpt = np.zeros(shape=shape)
        for pvalues in np.ndindex(*pcards):
            # for each super parent instantiation
            my_slc = [] # index into original cpt
            for pid,pvalue in zip(pids,pvalues):
                if pid not in abs_pids:
                    my_slc.append(pvalue)
                else:
                    original_values = cards_map_dict[pid][pvalue] # original states within this superstate
                    my_slc.append(np.array(original_values))

            cond_cpt = cpt[tuple(my_slc)] # pr(y|xv) for all x in x'
            cond_cpt = np.mean(cond_cpt,axis=tuple(np.arange(cond_cpt.ndim-1))) # compute mean
            assert cond_cpt.shape == (card,)
            avg_cpt[tuple(pvalues)] = cond_cpt
            # average cpt for this super parents instantiation

        if id in nodes_abs:
            # if this node is also abstracted node 
            cards_map = cards_map_dict[id]
            arrays = [np.sum(avg_cpt[...,tuple(cards)],axis=-1,keepdims=True) for cards in cards_map]
            avg_cpt = np.concatenate(arrays,axis=-1)
            assert avg_cpt.shape == tuple(pcards) + (scard,)
            node2 = Node(name,values=svalues,parents=parents,testing=False,cpt=avg_cpt)
            bn2.add(node2)
            print("Finish adding node %s" %(id,))

        else:
            node2 = Node(name,values=values,parents=parents,testing=False,cpt=avg_cpt)
            assert avg_cpt.shape == tuple(pcards) + (card,)
            bn2.add(node2)
            print("Finish adding node %s" %(id,))

        

# reparam incomplete tbn where nodes_abs have smaller cardinality with respect to true bn for query pr(q|e)
# parameter:
#   bn -- true bn
#   dag -- a list of list representing adjacency list of true bn
#    node_q, nodes_evid,nodes_abs -- query node, evidence nodes, abstracted nodes
#   cards_dict, scards_dict,cards_map_dict -- for each node, original card, super card, card mapping dictionaries
#   num_intervals -- num of intervals for testing cpts 
def reparam_tbn(dag,bn,node_q,nodes_evid,nodes_abs,cards_dict,scards_dict,cards_map_dict,num_intervals,flag='testing'):
    tbn = TBN("testing polytree")
    nodes_testing = testing_order(dag,node_q,nodes_abs) # make children of abstracted nodes as testing nodes
    add_order = add_order_for_tbn(dag,nodes_testing) # topo order for adding nodes to tbn
    assert flag in ('testing','baseline')
    if flag == 'testing':
        active_evidences = alloc_active_evidences(dag,nodes_evid,nodes_testing) # active evidences for each testing node
    else:
        active_evidences = {node_t:[] for node_t in nodes_testing} # baseline: do not use evidence conditioning cpts
    # remove testing node with no active evidence
    for id in add_order:
        # add node in order
        name = 'v%d'%id
        node = bn.node(name)
        if id not in nodes_testing:
            # if node is not testing, add to tbn 
            reparam_non_testing_node(node,bn,tbn,nodes_abs,nodes_testing,
                cards_dict,scards_dict,cards_map_dict)
        else:
            # if testing node, reparam testing cpts and add to 
            reparam_testing_node(node,bn,tbn,nodes_abs,nodes_testing,
                cards_dict,scards_dict,cards_map_dict,active_evidences,num_intervals)

    return tbn

# reparam BN according to average policy
# for each children of abstracted node, use average cpt contained in superstates
def reparam_bn_avg_baseline(dag,bn,node_q,nodes_evid,nodes_abs,cards_dict,scards_dict,cards_map_dict):
    bn2 = TBN("incomplete polytree")
    nodes_testing = testing_order(dag,node_q,nodes_abs) # make children of abstracted nodes as testing nodes
    add_order = add_order_for_tbn(dag,nodes_testing) # topo order for adding nodes to tbn

    for id in add_order:
        # add node in order
        name = 'v%d'%id
        node = bn.node(name)
        reparam_node_avg_baseline(node,bn,bn2,nodes_abs,nodes_testing,cards_dict,scards_dict,cards_map_dict)
    return bn2

def clip(dist):
    epsilon = np.finfo('float32').eps
    dist_safe = np.where(dist<epsilon, epsilon, dist)
    return dist_safe

def KL_divergence(dist_p,dist_q):
    dist_p = clip(dist_p)
    dist_q = clip(dist_q)
    kl_loss = np.sum(dist_p * np.log(dist_p/dist_q),axis=-1)
    return np.mean(kl_loss)

# reparam BN according to soft policy
# for each child Y of abstracted node, reparam Pr(Y|X'V) as Pr(Y|X \in X'V)
def reparam_bn_soft_baseline(dag,bn,node_q,nodes_evid,nodes_abs,cards_dict,scards_dict,cards_map_dict):
    bn2 = TBN("incomplete polytree")
    nodes_testing = testing_order(dag,node_q,nodes_abs) # make children of abstracted nodes as testing nodes
    add_order = add_order_for_tbn(dag,nodes_testing) # topo order for adding nodes to bn
    active_evidences = {node_t:[] for node_t in nodes_testing} # baseline: do not use evidence conditioning cpts
    # remove testing node with no active evidence
    for id in add_order:
        # add node in order
        name = 'v%d'%id
        node = bn.node(name)
        if id not in nodes_testing:
            # if node is not testing, add to bn2 
            reparam_non_testing_node(node,bn,bn2,nodes_abs,nodes_testing,
                cards_dict,scards_dict,cards_map_dict)
        else:
            # if testing node, reparam testing cpts and add to 
            reparam_testing_node(node,bn,bn2,nodes_abs,nodes_testing,
                cards_dict,scards_dict,cards_map_dict,active_evidences,num_intervals=None)

    return bn2

    
            



    



    







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


# do polytree experiment
def main1():
    dag = get_random_polytree(NUM_NODES,NUM_ITERS)
    bn,cards = sample_random_BN(dag)
    q,e,x = random_query(dag)
    dot(dag,q,e,x,fname="bn.gv")
    print("query: %s evidence: %s abstracted: %s" %(q,e,x))
    t = testing_order(dag,q,x)
    print("testing: %s" %(t,))
    add_order = add_order_for_tbn(dag,t)
    print("tbn order: %s" %(add_order,))
    alive_evid = alloc_active_evidences(dag,e,t)
    print("alive evidences: %s"% (alive_evid,))
    alive_evid2 = polytreeBN.alloc_alive_evidences(q,e,x,dag)
    print("alive evidences2:", alive_evid2)
    scards = [(card+1)//2 for card in cards] # lose half states
    cards_map_dict = []
    for card,scard in zip(cards,scards):
        cards_map = get_cards_map(card,scard)
        cards_map_dict.append(cards_map)

    tbn = reparam_tbn(dag,bn,q,e,x,cards,scards,cards_map_dict,num_intervals=NUM_INTERVALS)
    inputs = ['v%d'%eid for eid in e]
    output = 'v%d' % q
    true_ac = TAC(bn,inputs,output,trainable=False)
    incomplete_tac = TAC(tbn,inputs,output,trainable=False,sel_type='threshold') 

    ecards = [cards[eid] for eid in e]
    evidences = list(iter.product(*list(map(range,ecards)))) # enumerate all possible evidences
    evidences = data.evd_hard2lambdas(evidences,ecards)
    marginals1 = true_ac.evaluate(evidences)
    marginals2 = incomplete_tac.evaluate(evidences)
    print("marginals for true bn: %s" %(marginals1))
    print("marginals for incomplete tbn: %s" %(marginals2))
    kl_loss = KL_divergence(marginals1,marginals2)
    print("loss: %.6f" %(kl_loss,))

def do_polytree_tbn_experiment():
    dag = get_random_polytree(NUM_NODES,NUM_ITERS)
    dag,q,e,x = random_query(dag)
    bn,cards = sample_random_BN(dag,q,e,x)
    dot(dag,q,e,x,fname="polytree.gv")
    print("query: %s evidence: %s abstracted: %s" %(q,e,x))
    t = testing_order(dag,q,x)
    alive_evid = alloc_active_evidences(dag,e,t)
    print("alive evidences: %s"% (alive_evid,))
    scards = [(card+1)//2 for card in cards] # lose half states
    cards_map_dict = []
    for card,scard in zip(cards,scards):
        cards_map = get_cards_map(card,scard)
        cards_map_dict.append(cards_map)

    bn_baseline = reparam_bn_avg_baseline(dag,bn,q,e,x,cards,scards,cards_map_dict)
    bn_baseline2 = reparam_bn_soft_baseline(dag,bn,q,e,x,cards,scards,cards_map_dict)
    tbn_incomplete = reparam_tbn(dag,bn,q,e,x,cards,scards,cards_map_dict,num_intervals=NUM_INTERVALS)
    inputs = ['v%d'%eid for eid in e]
    output = 'v%d' % q
    ac_true = TAC(bn,inputs,output,trainable=False)
    ac_baseline = TAC(bn_baseline,inputs,output,trainable=False)
    ac_baseline2 = TAC(bn_baseline2,inputs,output,trainable=False)
    tac_incomplete = TAC(tbn_incomplete,inputs,output,trainable=False,sel_type='threshold')

    ecards = [cards[eid] for eid in e]
    evidences = list(iter.product(*list(map(range,ecards)))) # enumerate all possible evidences
    evidences = data.evd_hard2lambdas(evidences,ecards)
    marginals = ac_true.evaluate(evidences)
    marginals_baseline = ac_baseline.evaluate(evidences)
    marginals_incomplete = tac_incomplete.evaluate(evidences)
    marginals_baseline2 = ac_baseline2.evaluate(evidences)
    print("marginals for true bn: %s" %(marginals))
    print("marginals for tbn baseline: %s" %(marginals_baseline))
    print("maringals for tbn baseline2: %s" %(marginals_baseline2))
    print("marginals for tbn incomplete: %s" %(marginals_incomplete))
    kl_loss_baseline = KL_divergence(marginals,marginals_baseline)
    kl_loss_baseline2 = KL_divergence(marginals,marginals_baseline2)
    kl_loss = KL_divergence(marginals,marginals_incomplete)
    print("kl loss: %.9f" %kl_loss)
    print("Kl loss baseline: %.9f kl loss baseline2: %.9f kl loss: %.9f gain2: %.3f" %(kl_loss_baseline,kl_loss_baseline2,
        kl_loss,kl_loss_baseline2/kl_loss))

    return kl_loss,kl_loss_baseline,kl_loss_baseline2
    
def do_avg_polytree_tbn_experiment():
    kl_losses, kl_losses_baseline, kl_losses_baseline2 = [],[],[]
    f = open('output%d.txt'%NUM_NODES, 'w')
    for i in range(NUM_TRIALS):
        kl_loss,kl_loss_baseline,kl_loss_baseline2 = do_polytree_tbn_experiment()
        kl_losses.append(kl_loss)
        kl_losses_baseline.append(kl_loss_baseline)
        kl_losses_baseline2.append(kl_loss_baseline2)
        kl_loss_mean = np.array(kl_losses).mean()
        kl_loss_baseline_mean = np.array(kl_losses_baseline).mean()
        kl_loss_baseline2_mean = np.array(kl_losses_baseline2).mean()

        print("Trial: %d kl: %.9f kl baseline1: %.9f kl baseline2: %.9f" %(i,kl_loss_mean,kl_loss_baseline_mean,kl_loss_baseline2_mean))
        f.write("Trial: %d kl: %.9f kl baseline1: %.9f kl baseline2: %.9f" %(i,kl_loss_mean,kl_loss_baseline_mean,kl_loss_baseline2_mean))
        f.write('\n')
        f.flush()

    print("Finish experiments.")
    f.close()

def test_chain():
    dag = [[1],[2],[]] # v0 -> v1 -> v2
    q,e,x = 2,[0],[1]
    bn,cards = sample_random_BN(dag,q,e,x)
    dot(dag,q,e,x,fname="bn.gv")
    print("query: %s evidence: %s abstracted: %s" %(q,e,x))
    t = testing_order(dag,q,x)
    print("testing: %s" %(t,))
    add_order = add_order_for_tbn(dag,t)
    print("tbn order: %s" %(add_order,))
    alive_evid = alloc_active_evidences(dag,e,t)
    print("alive evidences: %s"% (alive_evid,))
    alive_evid2 = polytreeBN.alloc_alive_evidences(q,e,x,dag)
    print("alive evidences2:", alive_evid2)
    scards = [(card+1)//2 for card in cards] # lose half states
    cards_map_dict = []
    for card,scard in zip(cards,scards):
        cards_map = get_cards_map(card,scard)
        cards_map_dict.append(cards_map)

    tbn = reparam_tbn(dag,bn,q,e,x,cards,scards,cards_map_dict,num_intervals=NUM_INTERVALS)
    inputs = ['v%d'%eid for eid in e]
    output = 'v%d' % q
    true_ac = TAC(bn,inputs,output,trainable=False)
    incomplete_tac = TAC(tbn,inputs,output,trainable=False,sel_type='threshold') 

    ecards = [cards[eid] for eid in e]
    evidences = list(iter.product(*list(map(range,ecards)))) # enumerate all possible evidences
    evidences = data.evd_hard2lambdas(evidences,ecards)
    marginals1 = true_ac.evaluate(evidences)
    marginals2 = incomplete_tac.evaluate(evidences)
    print("marginals for true bn: %s" %(marginals1))
    print("marginals for incomplete tbn: %s" %(marginals2))

    


def test_cpt_selection_v2():
    dag = get_random_polytree(NUM_NODES,NUM_ITERS)
    tbn,cards = sample_random_TBN(dag)
    tbn2 = TBN("polytree2")
    for node in tbn._add_order:
        if not node.testing:
            # copy regular nodes
            name,values,parents,cpt = node.name,node.values,node.parents,node.cpt
            parents = [tbn2.node(p.name) for p in parents]
            node2 = Node(name,values=values,parents=parents,testing=False,cpt=cpt)
            tbn2.add(node2)
        else:
            #if testing node, convert to nodeV2
            name,values,parents = node.name,node.values,node.parents
            parents = [tbn2.node(p.name) for p in parents]
            node2 = NodeV2(name,values=values,parents=parents,testing=True,
                cpts=[node.cpt1,node.cpt2],thresholds=[node.threshold],num_intervals=2)
            tbn2.add(node2)

    query,evidences,_ = random_query(dag)
    num_examples = 1
    inputs = ['v'+str(evid) for evid in evidences]
    output = 'v'+str(query)
    ecards = [cards[evid] for evid in evidences]
    evidences = data.evd_random(size=num_examples,cards=ecards,hard_evidence=True)
    tac = TAC(tbn,inputs=inputs,output=output,sel_type='sigmoid',trainable=False)
    tac2 = TAC(tbn2,inputs=inputs,output=output,sel_type='sigmoid',trainable=False)
    marginals = tac.evaluate(evidences)
    print("marginals1: %s" %marginals)
    marginals2 = tac2.evaluate(evidences)
    print("marginals2: %s" %marginals2)

def test_TAC_v2():
    dag = get_random_polytree(NUM_NODES,NUM_ITERS)
    dag,query,evidences,_ = random_query(dag)
    tbn,cards = sample_random_TBN(dag)
    num_examples = 5
    dummy = Node(name='evid',values=['v0','v1'],parents=[]) # dummy evidence
    tbn.add(dummy)
    #inputs = ['v'+str(evid) for evid in evidences]
    inputs = ['evid']
    output = 'v'+str(query)
    outputs = [output]
    ecards = [cards[evid] for evid in evidences]
    #dot(dag,query,evidences,[],fname="testtbn.gv")
    #print("query: %s evidences: %s" %(query,evidences))
    evidences = data.evd_random(size=num_examples,cards=[2],hard_evidence=True)
    tac = TAC(tbn,inputs=inputs,output=output,sel_type='sigmoid',trainable=False)
    tac2 = TACV2(tbn,inputs=inputs,outputs=outputs,sel_type='sigmoid',trainable=False)
    marginals = tac.evaluate(evidences)
    print("marginals1: %s" %marginals)
    marginals2 = tac2.evaluate(evidences)
    #print("query cards: %s" % (qcards,))
    #print("marginals2 shape: %s" %(marginals2.shape,))
    print("marginals2: %s" %marginals2)
    assert np.allclose(marginals,marginals2)

def do_polytree_tbn_experiment_for_intervals():
    #intervals_list = [2] + list(10*i for i in range(1,6))
    dag = get_random_polytree(NUM_NODES,NUM_ITERS)
    dag,q,e,x = random_query(dag)
    bn,cards = sample_random_BN(dag,q,e,x)
    dot(dag,q,e,x,fname="polytree.gv")
    print("query: %s evidence: %s abstracted: %s" %(q,e,x))
    t = testing_order(dag,q,x)
    alive_evid = alloc_active_evidences(dag,e,t)
    print("alive evidences: %s"% (alive_evid,))
    scards = [(card+1)//2 for card in cards] # lose half states
    cards_map_dict = []
    for card,scard in zip(cards,scards):
        cards_map = get_cards_map(card,scard)
        cards_map_dict.append(cards_map)

    bn_baseline = reparam_bn_avg_baseline(dag,bn,q,e,x,cards,scards,cards_map_dict)
    bn_baseline2 = reparam_bn_soft_baseline(dag,bn,q,e,x,cards,scards,cards_map_dict)

    tbn_list = []
    for intervals in intervals_list:
        tbn = reparam_tbn(dag,bn,q,e,x,cards,scards,cards_map_dict,num_intervals=intervals)
        tbn_list.append(tbn)

    inputs = ['v%d'%eid for eid in e]
    output = 'v%d' % q
    ac_true = TAC(bn,inputs,output,trainable=False)
    ac_baseline = TAC(bn_baseline,inputs,output,trainable=False)
    ac_baseline2 = TAC(bn_baseline2,inputs,output,trainable=False)

    tac_list = []
    for tbn in tbn_list:
        tac = TAC(tbn,inputs,output,trainable=False,sel_type='threshold') 
        tac_list.append(tac)

    ecards = [cards[eid] for eid in e]
    evidences = list(iter.product(*list(map(range,ecards)))) # enumerate all possible evidences
    evidences = data.evd_hard2lambdas(evidences,ecards)
    marginals = ac_true.evaluate(evidences)
    marginals_baseline = ac_baseline.evaluate(evidences)
    marginals_baseline2 = ac_baseline2.evaluate(evidences)
    marginals_list = [tac.evaluate(evidences) for tac in tac_list]

    kl_loss_baseline1 = KL_divergence(marginals,marginals_baseline)
    kl_loss_baseline2 = KL_divergence(marginals,marginals_baseline2)
    kl_loss_list = [KL_divergence(marginals,marginals_tbn) for marginals_tbn in marginals_list]
    #print("kl loss: %.9f" %kl_loss)
    print("kl loss 1: %.9f kl loss 2: %.9f kl loss list: %s " %(kl_loss_baseline1, kl_loss_baseline2,
        kl_loss_list,))
    return kl_loss_baseline1, kl_loss_baseline2, kl_loss_list

def do_avg_polytree_tbn_experiment_for_intervals():
    kl_loss_1, kl_loss_2 = [], []
    kl_loss_list = [[] for _ in range(len(intervals_list))]
    f = open("output_%d_for_interval.txt"%NUM_NODES, mode='w')

    for i in range(NUM_TRIALS):
        kl_1, kl_2, kl_list = do_polytree_tbn_experiment_for_intervals()
        kl_loss_1.append(kl_1)
        kl_loss_2.append(kl_2)
        for list, kl in zip(kl_loss_list,kl_list):
            list.append(kl)

        mean_1, mean_2 = np.array(kl_loss_1).mean(), np.array(kl_loss_2).mean()
        means_list = [np.array(kl_loss).mean() for kl_loss in kl_loss_list]
        print("Trial %d kl loss 1: %.9f kl loss 2: %.9f kl loss list: %s " %(i,mean_1,mean_2,means_list))
        f.write("Trial %d kl loss 1: %.9f kl loss 2: %.9f kl loss list: %s " %(i,mean_1,mean_2,means_list))
        f.write('\n')
        f.flush()

    print("Finish experiment.")
    f.close()


def test_dot():
    dag = get_random_polytree(NUM_NODES,NUM_ITERS)
    dag,q,e,x = random_query(dag)
    bn,cards = sample_random_BN(dag,q,e,x)
    print("query: %d evidences: %s abstracted: %s" %(q,e,x))
    dot(dag,q,e,x,fname="polytree.gv")
    evidences = polytreeBN.alloc_alive_evidences(q,e,x,dag)
    print("alive evidences: %s" %(evidences,))



if __name__ == '__main__':
    #main1()
    #test_dot()
    #test_cpt_selection_v2()
    #test_TAC_v2()
    #test_chain()
    #main1()
    #do_polytree_tbn_experiment()
    #do_avg_polytree_tbn_experiment()
    do_polytree_tbn_experiment_for_intervals()

    
    
