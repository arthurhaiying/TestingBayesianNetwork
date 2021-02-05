from graphviz import Digraph
import numpy as np

from copy import copy,deepcopy
from pathlib import Path
import sys,os

from functools import reduce

import itertools as iter

if __name__ == '__main__':
    #basepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    basepath = Path(__file__).resolve().parents[2]
    #print("basepath: %s" %basepath)
    sys.path.append(str(basepath))

from examples.CaseStudy.NetParser import parseBN
from tbn.tbn import TBN
import tbn.cpt as CPT
from tbn.node import Node
import train.data as data
from tac import TAC,TACV2
import utils.utils as u


# return a dict of list representing underlying dag of BN
def get_dag(bn):
    dag = {node.name:[] for node in bn.nodes}
    for node in bn.nodes:
        name = node.name
        parents = node.parents
        parents = list(map(lambda x:x.name, parents))
        for p in parents:
            dag[p].append(name)

    return dag

# return a directed path from src node to dst node if exists
def directed_path(dag,src,dst):
    visited = set()
    previous = {} # keep track of last hop to this node
    ok = False
    queue = [src]
    visited.add(src)
    while queue:
        curr = queue.pop(0)
        if curr == dst:
            # if reached dst
            ok = True
            break
        for child in dag[curr]:
            if child not in visited:
                queue.append(child)
                visited.add(child)
                previous[child] = curr
    
    if not ok:
        # path do not exists
        return None
    path = [dst]
    curr = dst
    # trace path from dst to src
    while curr != src:
        curr = previous[curr]
        path.append(curr)
    path = path[::-1]
    return path

# delete a higher order edge from U to X
# choose alternative directed path and add states of U to intermediate nodes
# dag: the input dag
# return: 
#   dag2 - the resulted dag after this edge is deleted
#   intermediate - intermediate nodes whose states need to be appended
def delete_higher_order_edge(dag,node_u,node_x):
    assert node_x in dag[node_u]
    dag2 = {node:copy(children) for node,children in dag.items()}
    dag2[node_u].remove(node_x) # remove edge
    path = directed_path(dag2,node_u,node_x)
    assert path is not None # a directed path must exist
    #print("u: %s x: %s alternativepath: %s" %(node_u,node_x,path))
    intermedate = path[1:-1]
    return dag2,intermedate

# plan for deleting some higher order edges from input dag
# for each higher order edge (U,X), add state of U to all intermediate nodes
# return:
#   dag2 - the resulted dag after all higher order edges have been pruned
#   joint_states - a dict of list representing the joint states of the bn for recovery
def make_plan_for_delete_higher_order_edges(dag,edges):
    joint_state_dict = {node:set([node]) for node in dag}
    dag2 = {node:copy(children) for node,children in dag.items()}
    for (node_u,node_x) in edges:
        # for each higher order edge
        dag2,intermediate = delete_higher_order_edge(dag2,node_u,node_x)
        #print("u: %s x: %s intermediate: %s" %(node_u,node_x,intermediate))
        for node_i in intermediate:
            states = joint_state_dict[node_u] # for accumulative delete edge 
            joint_state_dict[node_i] |= states

    order = topo_sort(dag2)
    for node,joint_state in joint_state_dict.items():
        joint_state = sorted(list(joint_state),key=lambda x: order.index(x))
        joint_state_dict[node] = joint_state
        # joint states are sorted according to dag2
    return dag2,joint_state_dict


# return parents of every node in dag
def get_parent_list(dag):
    # get parents of each node
    parent_list = {node:[] for node in dag.keys()}
    for node in dag.keys():
        for child in dag[node]:
            parent_list[child].append(node)
    return parent_list

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



# return a topological ordering of nodes in dag
# dag: a dict of list representing directed adjacencies of DAG
def topo_sort(dag):

    in_degrees = {node:0 for node in dag.keys()}

    # compute in-degrees of each node in dag
    for node in dag.keys():
        for child in dag[node]:
            in_degrees[child]+=1
    
    queue = [node for node in dag.keys() if in_degrees[node] == 0] # roots
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


def check_normal_cpt(cpt):
    cpt = np.sum(cpt,axis=-1)
    assert np.allclose(cpt,1.0)

# reparam bn over dag2 using joint state space for nodes
# bn: true bn with underlying dag
# dag2: dag2 with missing higher order edges
# joint_state_dict: plan for removing higher order edges
# return: bn2 - another BN over dag2 with nodes using joint state space
def reparam_bn_over_joint_state(bn,dag2,nodes_joint,joint_state_dict):
    card_dict = {node.name:len(node.values) for node in bn.nodes}
    joint_card_dict = {}
    for node,joint_state in joint_state_dict.items():
        jcards = [card_dict[n] for n in joint_state]
        jcard = reduce(lambda x,y: x*y, jcards)
        joint_card_dict[node] = jcard

    order = topo_sort(dag2) 
    parent_list = get_parent_list(dag2)
    bn2 = TBN(bn.name+'2')
    for name in order:
        # add node in topological order according to dag2
        #print("Reparam node: %s..." %name)
        node = bn.node(name)
        values,pnodes1,cpt1 = node.values,node.parents,deepcopy(node.cpt)
        parents1 = [p.name for p in pnodes1]
        parents2 = get_parent_list(dag2)[name] # parents of node in dag2
        #print("parents 2: %s" % parents2)
        pnodes2 = list(map(bn2.node, parents2))

        if name not in nodes_joint and not (set(parents2) & set(nodes_joint)):
            # base case: not joint node and parents are not joint node
            # just copy node
            node2 = Node(name=name,values=values,parents=pnodes2,testing=False,cpt=np.copy(cpt1))
            bn2.add(node2)
            
        elif name not in nodes_joint and set(parents2) & set(nodes_joint):
            #print("Reparam child %s..." %name)
            # case2: not joint node but some parents X are joint node (X,U)
            # becareful about the order of the parents
            parents1 = list(map(lambda x:x.name, pnodes1)) # original parents X
            joint_parents2 = []
            for p in parents2:
                joint_parents2.extend(joint_state_dict[p])# joint parents (X,U)
            #print("parents1: %s" %parents1)
            #print("joint parents2: %s" %joint_parents2)
            assert set(parents1).issubset(joint_parents2) # all original parents are included
            '''
            for i in range(len(joint_parents2)-1):
                u.input_check(order.index(joint_parents2[i]) <= order.index(joint_parents2[i+1]),
                    "joint parents of node %s are not sorted!" %name)
            '''

            cpt1_shape = tuple(card_dict[p] for p in parents1) + (card_dict[name],) # X,Y
            joint_cpt2_shape = tuple(card_dict[p] for p in joint_parents2) + (card_dict[name],) # (U,X,Y)
            final_cpt2_shape = tuple(joint_card_dict[p] for p in parents2)  + (card_dict[name],) # (U*X,Y)
            assert cpt1.shape == cpt1_shape

            # sort original cpt
            parents1_axis = [(p,i) for i,p in enumerate(parents1)]
            parents1_axis.sort(key=lambda x: joint_parents2.index(x[0]))
            sorted_parents1 = [p for p,i in parents1_axis]
            #print("sorted parents1: %s" %sorted_parents1)
            sorted_axis = tuple([i for p,i in parents1_axis])
            sorted_axis = sorted_axis + (-1,)
            cpt2 = np.transpose(cpt1,axes=sorted_axis) # cpt2 = (X,Y)
            # add extra parents
            my_slc = [slice(None)]*len(joint_cpt2_shape)
            j = 0
            for i,p2 in enumerate(joint_parents2):
                p1 = sorted_parents1[j]
                if p1 == p2:
                    j+=1
                else:
                    my_slc[i] = np.newaxis

            cpt2 = cpt2[tuple(my_slc)]
            cpt2 = np.broadcast_to(cpt2,joint_cpt2_shape) # cpt2 = (U,X,Y)
            cpt2 = np.reshape(cpt2,final_cpt2_shape) # cpt2 = (U*X, Y)
            check_normal_cpt(cpt2)
            #print("Finish reparam child %s" %name)

            node2 = Node(name,values=values,parents=pnodes2,testing=False,cpt=cpt2)
            bn2.add(node2)

        elif name in nodes_joint:
            # case 3: node are joint node and some parents may be joint nodes
            #print("Reparam joint node %s" %name)
            joint_nodes = joint_state_dict[name] 
            joint_nodes_values = [bn.node(n).values for n in joint_nodes]
            joint_values = []
            for jvalues in iter.product(*joint_nodes_values):
                l = ['%s=%s'%(jnode,jvalue)
                    for jnode, jvalue in zip(joint_nodes, jvalues)]
                state = ','.join(l)
                joint_values.append(state)
                # create joint state for this node

            parents1 = list(map(lambda x:x.name, pnodes1)) # original parents X
            #print("parents1: %s" %parents1)
            joint_parents2 = []
            for p in parents2:
                joint_parents2.extend(joint_state_dict[p]) # joint parents (X,U)
            '''
            for i in range(len(joint_parents2)-1):
                u.input_check(order.index(joint_parents2[i]) <= order.index(joint_parents2[i+1]),
                    "joint parents of node %s are not sorted!" %name)
            '''
            #print("joint parents2: %s" %joint_parents2)
            dummy_nodes = [n for n in joint_nodes if n != name]
            u.input_check(len(set(dummy_nodes)) == len(dummy_nodes), "no repeated dummy nodes")
            #print("dummy nodes: %s" %dummy_nodes)
            assert all(n in joint_parents2 for n in dummy_nodes)
            dummy_nodes_cards = [card_dict[n] for n in dummy_nodes]
            # nodes in both joint parents and joint states
            assert set(parents1).issubset(joint_parents2) # all original parents are included

            cpt1_shape = tuple(card_dict[p] for p in parents1) + (card_dict[name],) # X,Y
            joint_cpt2_shape = tuple(card_dict[p] for p in joint_parents2) \
                + (card_dict[name],) # pr(U,X ==> Y)
            dummy_joint_cpt2_shape = tuple(card_dict[p] for p in joint_parents2) \
                + tuple(card_dict[n] for n in joint_nodes) # pr(U,X ==> X,Y)
            final_cpt2_shape = tuple(joint_card_dict[p] for p in parents2)  \
                + (joint_card_dict[name],) # pr(U*X ==> X*Y)
            assert cpt1.shape == cpt1_shape

            # sort original cpt and  add extra parents
            parents1_axis = [(p,i) for i,p in enumerate(parents1)]
            parents1_axis.sort(key=lambda x: joint_parents2.index(x[0]))
            sorted_parents1 = [p for p,i in parents1_axis]
            #print("sorted parents1: %s" %sorted_parents1)
            sorted_axis = tuple([i for p,i in parents1_axis])
            sorted_axis = sorted_axis + (-1,)
            cpt2 = np.transpose(cpt1,axes=sorted_axis) # cpt2 = (X,Y)
            # add extra parents
            my_slc = [slice(None)]*len(joint_cpt2_shape)
            j = 0
            for i,p2 in enumerate(joint_parents2):
                p1 = sorted_parents1[j]
                if p1 == p2:
                    j+=1
                else:
                    my_slc[i] = np.newaxis

            joint_cpt2 = cpt2[tuple(my_slc)]
            joint_cpt2 = np.broadcast_to(cpt2,joint_cpt2_shape) # cpt2 = pr(U,X ==> Y)

            # add dummy states which must be consistent
            dummy_joint_cpt2 = np.zeros(dummy_joint_cpt2_shape) # pr(U,X ==> X, Y)
            for dvalues in iter.product(*list(map(range,dummy_nodes_cards))):
                # for each dummy parent state x
                slc_joint_cpt2 = [slice(None)] * len(joint_cpt2_shape)
                slc_dummy_joint_cpt2 = [slice(None)] * len(dummy_joint_cpt2_shape)
                for dnode,dvalue in zip(dummy_nodes,dvalues):
                    index = joint_parents2.index(dnode) # index of x in U,X
                    dummy_index = joint_nodes.index(dnode) # index of x in U,X ==> X, Y
                    slc_joint_cpt2[index] = dvalue
                    slc_dummy_joint_cpt2[index] = dvalue
                    slc_dummy_joint_cpt2[len(joint_parents2) + dummy_index] = dvalue
                    # index into joint cpt and dummy joint cpt

                cond = joint_cpt2[tuple(slc_joint_cpt2)]
                dummy_joint_cpt2[tuple(slc_dummy_joint_cpt2)] = cond 
                # set pr(X,Y|U,X) = pr(X|U,X)

            final_cpt2 = np.reshape(dummy_joint_cpt2,final_cpt2_shape) # convert to U*X ==> X*Y
            check_normal_cpt(final_cpt2)
            #print("Finish reparam joint node %s." %name)

            node2 = Node(name,values=joint_values,parents=pnodes2,testing=False,cpt=final_cpt2)
            bn2.add(node2)

        #print("Finish reparam node %s." %name)

    return bn2

# compute Pr(Q|E)
def posterior_marginal(bn,nodes_q,nodes_evid,card_dict):
    ecards = [card_dict[e] for e in nodes_evid]
    qcards = [card_dict[q] for q in nodes_q]
    if not nodes_evid:
        # no inputs
        bn1 = deepcopy(bn)
        single = Node(name='single',parents=[],testing=False) # disconnected evidence
        bn1.add(single)
        inputs = ['single']
        outputs = nodes_q
        ac = TACV2(bn1,inputs,outputs,trainable=False)
        num_examples = 2
        evidences = data.evd_random(size=num_examples,cards=[2],hard_evidence=True)
        marginals = ac.evaluate(evidences)
        assert marginals.shape == (num_examples,)+tuple(qcards)
        assert np.allclose(marginals[0],marginals[1])
        return marginals[0]

    else:
        inputs = nodes_evid
        outputs = nodes_q
        ac = TACV2(bn,inputs,outputs,trainable=False)
        evidences = list(iter.product(*list(map(range,ecards))))
        num_examples = len(evidences)
        evidences = data.evd_hard2lambdas(evidences,ecards)
        marginals = ac.evaluate(evidences)
        assert marginals.shape == (num_examples,) + tuple(qcards)
        shape = tuple(ecards) + tuple(qcards)
        marginals = marginals.reshape(shape)
        return marginals



# remove higher order edges and return bn over joint states               
def delete_edges_and_reparam_bn(bn,edges):
    dag = get_dag(bn)
    dag2, joint_state_dict = make_plan_for_delete_higher_order_edges(dag,edges)
    # make plan for deleting higher order edges 
    # dag2: dag after removed edges
    # joint state dict: states of (joint) nodes after deleting edges
    nodes_joint = [n for n in dag.keys() if len(joint_state_dict[n]) >= 2] # joint nodes
    bn2 = reparam_bn_over_joint_state(bn,dag2,nodes_joint,joint_state_dict)
    return bn2,joint_state_dict

# delete a higher order edge 2 of from U to X
# choose alternative path from U to X that contains no v-structure and append states of U
# dag: the input dag
# return: 
#   dag2 - the resulted dag after this edge is deleted
#   node_block - the blocking node along the alternative path
#   left_path - directed path from blocking node to X
#   right_path - directed path from blocking node to U
def delete_higher_order_edge2(dag,node_u,node_x):
    dag2 = {k:copy(v) for k,v in dag.items()}
    dag2[node_u].remove(node_x) # delete edge from U to X
    ancestors_u = ancestors(node_u,dag2)
    node_block = None
    left_path,right_path = None,None
    # assume that all nodes in the right path have single parent
    for ancestor in ancestors_u:
        path = directed_path(dag2,ancestor,node_x)
        if path is not None:
            # find alternative path
            node_block = ancestor
            left_path = path[1:] # path from B to X
            right_path = directed_path(dag2,ancestor,node_u)[1:] # path from B to U
            break

    if node_block is None:
        raise RuntimeError("No alternative path found!")
    return dag2,node_block,left_path,right_path


# delete one higher order edge 2 and reparam BN over joint states
def delete_one_edge_and_reparam_bn2(bn,edge):
    node_u,node_x = edge
    dag = get_dag(bn)
    dag2,node_block,left_path,right_path = delete_higher_order_edge2(dag,node_u,node_x)
    nodes_joint = [node_block] + left_path[:-1] + right_path[:-1]
    joint_state_dict = {}
    for node in dag.keys():
        if node not in nodes_joint:
            joint_state_dict[node] = [node]
        else:
            joint_state_dict[node] = [node_u,node]
            # assume that original state is always the last
    card_dict = {node.name:len(node.values) for node in bn.nodes}
    joint_card_dict = {}
    for node,joint_state in joint_state_dict.items():
        jcards = [card_dict[n] for n in joint_state]
        jcard = reduce(lambda x,y: x*y, jcards)
        joint_card_dict[node] = jcard

    # make plan for deleteing this higher order edge 2
    bn2 = TBN(bn.name+'2')
    order = topo_sort(dag2)
    parents2_list = get_parent_list(dag2)

    def __reparam_regular_node(node):
        # for nodes not joint and not children of joint nodes
        bnNode = bn.node(node)
        name,values,pnodes1,cpt = bnNode.name,bnNode.values,bnNode.parents,bnNode.cpt
        parents = [p.name for p in pnodes1]
        pnodes2 = [bn2.node(p) for p in parents]
        node2 = Node(name,values=values,parents=pnodes2,testing=False,cpt=cpt)
        bn2.add(node2)
        return

    def __reparam_child_of_joint_node(node):
        # for nodes not joint but children of joint nodes
        # node can be children of multiple joint nodes
        bnNode = bn.node(node)
        name,values,pnodes1,cpt1 = bnNode.name,bnNode.values,bnNode.parents,bnNode.cpt
        parents = [p.name for p in pnodes1]
        pnodes2 = [bn2.node(p) for p in parents]
        joint_parents = []
        for p in parents:
            joint_parents.extend(joint_state_dict[p])# joint parents (U,parents)

        joint_cpt2_shape = tuple(card_dict[p] for p in joint_parents) + (card_dict[name],) # (U,parents ==> X)
        final_cpt2_shape = tuple(joint_card_dict[p] for p in parents)  + (card_dict[name],) # (U*parents ==> X)
        # add extra parents to cpt
        shape = []
        i = 0
        for p2 in joint_parents:
            p1 = parents[i]
            if p1 == p2:
                shape.append(card_dict[p2])
                i+=1
            else:
                shape.append(1)

        #print("name: %s shape: %s" %(name,shape))
        shape.append(card_dict[name])
        joint_cpt2 = cpt1.reshape(tuple(shape))
        joint_cpt2 = np.broadcast_to(joint_cpt2,joint_cpt2_shape)
        final_cpt2 = joint_cpt2.reshape(final_cpt2_shape)
        node2 = Node(name,values=values,parents=pnodes2,testing=False,cpt=final_cpt2)
        bn2.add(node2)
        return

    def __reparam_blocking_node(node):
        # for blocking node S ==> U,S
        bnNode = bn.node(node)
        name,values,pnodes1,cpt1 = bnNode.name,bnNode.values,bnNode.parents,bnNode.cpt
        parents = [p.name for p in pnodes1]
        pnodes2 = [bn2.node(p) for p in parents]

        joint_nodes = joint_state_dict[node]
        assert joint_nodes == [node_u,node] # append state U to state S
        joint_values = []
        for value_u in bn.node(node_u).values:
            for value in values:
                joint_value = '%s=%s,%s=%s' %(node_u,value_u,node,value) 
                joint_values.append(joint_value)

        joint_cpt2_shape = tuple(card_dict[p] for p in parents) \
            + (card_dict[node_u],card_dict[node]) # (parent(S) ==> U,S)
        final_cpt2_shape = tuple(card_dict[p] for p in parents) \
            + (joint_card_dict[node],) # (parent(S) ==> U*S)
        joint_cpt2 = posterior_marginal(bn,joint_nodes,parents,card_dict)
        assert joint_cpt2.shape == joint_cpt2_shape
        # pr(U,S|parent(S))   
        final_cpt2 = joint_cpt2.reshape(final_cpt2_shape)
        node2 = Node(name,values=joint_values,parents=pnodes2,cpt=final_cpt2)
        bn2.add(node2)
        return

    def __reparam_intermediate_node(node):
        # for nodes along the left and right path except node x
        bnNode = bn.node(node)
        name,values,pnodes1,cpt1 = bnNode.name,bnNode.values,bnNode.parents,bnNode.cpt
        parents1 = [p.name for p in pnodes1]
        parents = copy(parents1)
        if node == node_x:
            parents.remove(node_u)
        # in bn2, node x no longer have parent u 
        pnodes2 = [bn2.node(p) for p in parents]
        joint_parents = []
        for p in parents:
            joint_parents.extend(joint_state_dict[p])# joint parents (U,parents)
        if node == node_x:
            assert len(joint_parents) == len(parents1)
        else:
            assert len(joint_parents) == len(parents1)+1 # one dummy parent U

        joint_nodes = joint_state_dict[node]
        if node == node_u or node == node_x:
            assert joint_nodes == [node]
        else:
            assert joint_nodes == [node_u,node] # append state U to state S
            joint_values = []
            for value_u in bn.node(node_u).values:
                for value in values:
                    joint_value = '%s=%s,%s=%s' %(node_u,value_u,node,value) 
                    joint_values.append(joint_value)

        if node == node_x:
            parents1 = [(p,i) for i,p in enumerate(parents1)]
            parents1.sort(key=lambda x: joint_parents.index(x[0]))
            sorted_axis = [i for p,i in parents1]
            sorted_axis.append(-1)
            joint_cpt2 = np.transpose(cpt1,axes=sorted_axis)
            final_cpt2_shape = tuple(joint_card_dict[p] for p in parents) + (card_dict[node],)
            final_cpt2 = joint_cpt2.reshape(final_cpt2_shape)
            node2 = Node(name,values=values,parents=pnodes2,testing=False,cpt=final_cpt2)
            bn2.add(node2)

        elif node == node_u:
            joint_cpt2_shape = tuple(card_dict[p] for p in joint_parents) \
                + (card_dict[name],) # (U,parents ==> U)
            joint_cpt2 = np.zeros(joint_cpt2_shape)
            # for each dummy u state
            for value_u in range(card_dict[node_u]):
                my_slc = [slice(None)]*len(joint_cpt2_shape)
                index = joint_parents.index(node_u)
                my_slc[index] = value_u
                my_slc[-1] = value_u
                joint_cpt2[tuple(my_slc)] = 1.0
                # assign one for consistent u
            final_cpt2_shape = tuple(joint_card_dict[p] for p in parents)  \
                + (joint_card_dict[name],) # (U*parents ==> U*Y)
            final_cpt2 = joint_cpt2.reshape(final_cpt2_shape)
            node2 = Node(name,values=values,parents=pnodes2,testing=False,cpt=final_cpt2)
            bn2.add(node2)

        else:
            joint_cpt2_shape = tuple(card_dict[p] for p in joint_parents) \
                + (card_dict[node_u],card_dict[name]) # (U,parents ==> U,Y)
            joint_cpt2 = np.zeros(joint_cpt2_shape)
            # for each dummy u state
            for value_u in range(card_dict[node_u]):
                my_slc = [slice(None)]*len(joint_cpt2_shape)
                index = joint_parents.index(node_u)
                my_slc[index] = value_u
                my_slc[-2] = value_u
                joint_cpt2[tuple(my_slc)] = cpt1

            final_cpt2_shape = tuple(joint_card_dict[p] for p in parents)  \
                + (joint_card_dict[name],) # (U*parents ==> U*Y)
            final_cpt2 = joint_cpt2.reshape(final_cpt2_shape)
            node2 = Node(name,values=joint_values,parents=pnodes2,testing=False,cpt=final_cpt2)
            bn2.add(node2)
        return

    # add node in topological order according to dag after missing edge
    for node in order:
        if node == node_block:
            # for blocking node
            __reparam_blocking_node(node)
        elif node in left_path or node in right_path:
            # for intermediate node and node_u and node_x
            __reparam_intermediate_node(node)
        elif set(parents2_list[node]) & set(nodes_joint):
            # for children of joint nodes
            __reparam_child_of_joint_node(node)
        else:
            __reparam_regular_node(node)

    return bn2,joint_state_dict

# state dict: current state dict after edges1 have been removed
def delete_edges_and_reparam_bn2(bn,edges2,state_dict):
    bn2 = bn
    for edge2 in edges2:
        # remove edge2 one by one
        node_u, node_x = edge2
        print("Removing edge %s -> %s..." %(node_u,node_x))
        bn2, dict = delete_one_edge_and_reparam_bn2(bn2,edge2)
        nodes_joint = {k for k,v in dict.items() if len(v) == 2}
        print("Nodes %s become joint!" %nodes_joint)
        state_dict_next = {}
        # compute current state dict
        for node,states in state_dict.items():
            if node in nodes_joint:
                state_next = state_dict[node_u] + states
                state_dict_next[node] = state_next
                print("Node %s become %s" %(node,state_dict_next[node]))
            else:
                state_dict_next[node] = states
        state_dict = state_dict_next
        print()

    return bn2,state_dict





        

        
      

def dot(dag,node_q,nodes_evid,nodes_abs,labels=None,fname="bn.gv"):
    d = Digraph()
    d.attr(rankdir='TD')
    for node in dag.keys():
        if node in nodes_evid:
            d.node(node,shape="circle", style="filled")

        elif node in nodes_abs:
            d.node(node,shape="doublecircle")
        else:
            d.node(node,shape="circle")
    for node in dag.keys():
        for child in dag[node]:
            d.edge(node,child)
    try:
        d.render(fname, view=False)
    except:
        print("Need to download graphviz")


def test_sachs():
    filename = 'examples/CaseStudy/sachs.net'
    print("start reading net...")
    bn = parseBN(filename)
    print("finish reading net.")

    edges = [
        ('PKA', 'Mek'),
        ('PKA', 'Erk'),
        ('PKA', 'Akt'),
        ('PKC', 'Mek'),
        ('PKC', 'Raf'),
        ('PKC', 'Jnk'),
        ('PKC', 'P38'),
        
    ] 
    # higher order edges

    nodes_evid = ['PKC','Jnk','P38']
    node_q = 'Akt'
    dag = get_dag(bn)
    dot(dag,node_q,nodes_evid,[],fname="alarm1.gv")
    #exit(1)
    # nodes with larger joint states
    print("Start reparam bn...")
    #bn2 = reparam_bn_over_joint_state(bn,dag2,nodes_joint,joint_states)
    bn2,joint_state_dict = delete_edges_and_reparam_bn(bn,edges)
    print("Finish reparam bn.")
    joint_state_dict = {key:value for key,value in joint_state_dict.items() if len(value) >= 2}
    print("joint states %s" %(joint_state_dict,))

    dag2 = get_dag(bn)
    dot(dag2,node_q,nodes_evid,[],fname="sachs2.gv")
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

'''
    bn0 = TBN("bn0")
    for node in bn.nodes:
        if node.name != "dysp":
            name,values,pnodes1,cpt = node.name,node.values,node.parents,node.cpt
            parents = [p.name for p in pnodes1]
            pnodes0 = [bn0.node(p) for p in parents]
            bn0.add(Node(name,values=values,parents=pnodes0,cpt=cpt))
        else:
            name,values = node.name,node.values
            pnodes0 = [bn0.node('either')]
            cpt = CPT.random(2,[2])
            bn0.add(Node(name,values=values,parents=pnodes0,cpt=cpt))

    print("this is bn0")
    test_ac = TAC(bn,inputs=['either'],output='xray',trainable=False)
    print("bn0 compiles!")
    exit(1)
'''

def test_asia():
    filename = 'examples/CaseStudy/asia.net'
    print("start reading net...")
    bn = parseBN(filename)
    print("finish reading net.")


    edges = [
        ('bronc', 'dysp'),
    ] 
    # higher order edges

    nodes_evid = ['bronc','xray']
    node_q = 'dysp'
    dag = get_dag(bn)
    dot(dag,node_q,nodes_evid,[],fname="asia.gv")

    dag2,node_block,left_path,right_path = delete_higher_order_edge2(dag,'bronc','dysp')
    print("node block: %s" %node_block)
    print("left path: %s" %left_path)
    print("right path: %s" %right_path)

    bn2,joint_state_dict = delete_edges_and_reparam_bn2(bn,edges[0])
    print("joint nodes: %s" %joint_state_dict)

    for bnNode in bn2.nodes:
        print("......................Node......................")
        name,values,pnodes,cpt = bnNode.name,bnNode.values,bnNode.parents,bnNode.cpt
        parents = [p.name for p in pnodes]
        for p in pnodes:
            assert p._tbn == bn2
            # same nodes in tbn
        print("name:", name)
        print("values:", values)
        print("parents: ", parents)
        print("origin cpt", bn.node(name).cpt)
        print("cpt: ", cpt)
        print()


    test_ac2 = TAC(bn2,inputs=['either'],output='xray',trainable=False)
    print("bn2 compiles!")
    exit(1)


    ecards = [len(bn.node(e).values) for e in nodes_evid]
    ac = TAC(bn,inputs=nodes_evid,output=node_q,trainable=False)
    ac2 = TAC(bn2,inputs=nodes_evid,output=node_q,trainable=False)
    #evidences = [[0,1]]
    evidences = list(iter.product(*list(map(range,ecards))))
    evidences = data.evd_hard2lambdas(evidences,ecards)
    marginals = ac.evaluate(evidences)
    marginals2 = ac2.evaluate(evidences)
    print("marginals1: %s" %(marginals,))
    print("marginals2: %s" %(marginals2,))
    assert np.allclose(marginals,marginals2)

if __name__ == '__main__':
    filename = 'examples/CaseStudy/child.net'
    print("start reading net...")
    bn = parseBN(filename)
    print("finish reading net.")

    edges1 = [
        ('Disease','Age'),
    ] 

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
    dag = get_dag(bn)
    dot(dag,node_q,nodes_evid,[],fname="child.gv")
    # nodes with larger joint states
    print("Start reparam bn...")
    ''' Step 1: remove all higher order edges 1'''
    nodes_joint = []
    bn2,joint_state_dict = delete_edges_and_reparam_bn(bn,edges1)
    bn2,joint_state_dict = delete_edges_and_reparam_bn2(bn,edges2,joint_state_dict)



    #bn2 = reparam_bn_over_joint_state(bn,dag2,nodes_joint,joint_states)
    #bn2,joint_state_dict = delete_edges_and_reparam_bn(bn,edges)
    print("Finish reparam bn.")
    joint_state_dict = {key:value for key,value in joint_state_dict.items() if len(value) >= 2}
    print("nodes joint are %s" %(joint_state_dict,))

    dag2 = get_dag(bn)
    dot(dag2,node_q,nodes_evid,[],fname="child2.gv")
    ecards = [len(bn.node(e).values) for e in nodes_evid]
    ac = TAC(bn,inputs=nodes_evid,output=node_q,trainable=False)
    ac2 = TAC(bn2,inputs=nodes_evid,output=node_q,trainable=False)
    #evidences = [[0,0,1,0,1]]
    evidences = list(iter.product(*list(map(range,ecards))))
    evidences = data.evd_hard2lambdas(evidences,ecards)
    marginals = ac.evaluate(evidences)
    marginals2 = ac2.evaluate(evidences)
    print("marginals1: %s" %(marginals,))
    print("marginals2: %s" %(marginals2,))
    assert np.allclose(marginals,marginals2)
    

    



    







    