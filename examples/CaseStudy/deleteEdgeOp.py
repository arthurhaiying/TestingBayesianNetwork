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
from tbn.node import Node
import train.data as data
from tac import TAC
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
        print("u: %s x: %s intermediate: %s" %(node_u,node_x,intermediate))
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
        parents2 = get_parent_list(dag2)[name] # parents of node in dag2
        #print("parents 2: %s" % parents2)
        pnodes2 = list(map(bn2.node, parents2))

        if name not in nodes_joint and not (set(parents2) & set(nodes_joint)):
            # base case: not joint node and parents are not joint node
            # just copy node
            node2 = Node(name=name,values=values,parents=pnodes2,testing=False,cpt=np.copy(cpt1))
            bn2.add(node2)
            
        elif name not in nodes_joint and set(parents2) & set(nodes_joint):
            print("Reparam child %s..." %name)
            # case2: not joint node but some parents X are joint node (X,U)
            # becareful about the order of the parents
            parents1 = list(map(lambda x:x.name, pnodes1)) # original parents X
            joint_parents2 = []
            for p in parents2:
                joint_parents2.extend(joint_state_dict[p])# joint parents (X,U)
            print("parents1: %s" %parents1)
            print("joint parents2: %s" %joint_parents2)
            assert set(parents1).issubset(joint_parents2) # all original parents are included
            for i in range(len(joint_parents2)-1):
                u.input_check(order.index(joint_parents2[i]) <= order.index(joint_parents2[i+1]),
                    "joint parents of node %s are not sorted!" %name)

            cpt1_shape = tuple(card_dict[p] for p in parents1) + (card_dict[name],) # X,Y
            joint_cpt2_shape = tuple(card_dict[p] for p in joint_parents2) + (card_dict[name],) # (U,X,Y)
            final_cpt2_shape = tuple(joint_card_dict[p] for p in parents2)  + (card_dict[name],) # (U*X,Y)
            assert cpt1.shape == cpt1_shape

            # sort original cpt
            parents1_axis = [(p,i) for i,p in enumerate(parents1)]
            parents1_axis.sort(key=lambda x: joint_parents2.index(x[0]))
            sorted_parents1 = [p for p,i in parents1_axis]
            print("sorted parents1: %s" %sorted_parents1)
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
            print("Finish reparam child %s" %name)

            node2 = Node(name,values=values,parents=pnodes2,testing=False,cpt=cpt2)
            bn2.add(node2)

        elif name in nodes_joint:
            # case 3: node are joint node and some parents may be joint nodes
            print("Reparam joint node %s" %name)
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
            print("parents1: %s" %parents1)
            joint_parents2 = []
            for p in parents2:
                joint_parents2.extend(joint_state_dict[p]) # joint parents (X,U)
            for i in range(len(joint_parents2)-1):
                u.input_check(order.index(joint_parents2[i]) <= order.index(joint_parents2[i+1]),
                    "joint parents of node %s are not sorted!" %name)
            print("joint parents2: %s" %joint_parents2)
            dummy_nodes = [n for n in joint_nodes if n != name]
            u.input_check(len(set(dummy_nodes)) == len(dummy_nodes), "no repeated dummy nodes")
            print("dummy nodes: %s" %dummy_nodes)
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
            print("sorted parents1: %s" %sorted_parents1)
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
            print("Finish reparam joint node %s." %name)

            node2 = Node(name,values=joint_values,parents=pnodes2,testing=False,cpt=final_cpt2)
            bn2.add(node2)

        #print("Finish reparam node %s." %name)

    return bn2

# remove higher order edges and return bn over joint states               
def delete_edge_and_reparam_bn(bn,edges):
    dag = get_dag(bn)
    dag2, joint_state_dict = make_plan_for_delete_higher_order_edges(dag,edges)
    # make plan for deleting higher order edges 
    # dag2: dag after removed edges
    # joint state dict: states of (joint) nodes after deleting edges
    nodes_joint = [n for n in dag.keys() if len(joint_state_dict[n]) >= 2] # joint nodes
    bn2 = reparam_bn_over_joint_state(bn,dag2,nodes_joint,joint_state_dict)
    return bn2,joint_state_dict

   





    


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

if __name__ == '__main__':
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
    dot(dag,node_q,nodes_evid,[],fname="sachs.gv")
    # nodes with larger joint states
    print("Start reparam bn...")
    #bn2 = reparam_bn_over_joint_state(bn,dag2,nodes_joint,joint_states)
    bn2,joint_state_dict = delete_edge_and_reparam_bn(bn,edges)
    print("Finish reparam bn.")
    joint_state_dict = {key:value for key,value in joint_state_dict.items() if len(value) >= 2}
    print("joint states %s" %(joint_state_dict,))

    dag2 = get_dag(bn)
    dot(dag2,node_q,nodes_evid,[],fname="sachs2.gv")
    

    '''
    name = "PKA"
    node1 = bn.node(name)
    node2 = bn2.node(name)
    parent1 = "PKC"
    parent2 = "PKC"
    joint_parent2 = ["PKC"]
    joint_state = ["PKC","PKA"]
    assert tuple(joint_state) == tuple(joint_states[name])
    print("joint values %s" % node2.values)
    cpt1 = node1.cpt
    cpt2 = node2.cpt
    assert cpt2.shape == (3,9)
    print("cpt1", cpt1)
    print("cpt2", cpt2)
    for c in range(3):
        for a in range(3):
            for c_prime in range(3):
                if c_prime != c:
                    assert np.isclose(cpt2[c][c_prime*3+a],0)
                else:
                    assert np.isclose(cpt2[c][c_prime*3+a],cpt1[c][a]) 

    cpt2 = np.reshape(cpt2,(3,3,3))
    print("cpt2", cpt2)


    name = "P38"
    node1 = bn.node(name)
    node2 = bn2.node(name)
    parent1 = "PKA"
    parent2 = "PKA"
    joint_parent2 = ["PKC","PKA"]
    print("values %s" % node2.values)
    cpt1 = node1.cpt
    cpt2 = node2.cpt
    assert cpt2.shape == (9,3)
    print("cpt1", cpt1)
    #print("cpt2", cpt2)
    for c in range(3):
        for a in range(3):
            for p in range(3):
                assert cpt2[c*3+a][p] == cpt1[a][c][p] 

    cpt2 = np.swapaxes(np.reshape(cpt2,(3,3,3)),0,1)
    print("cpt2", cpt2)
    '''
    
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
    







    