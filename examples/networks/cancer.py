from tbn.tbn import TBN
from tbn.node import Node




# get cancer true model
def cancer_bn():
    # cpts
    cpt_a = [0.6, 0.4]
    cpt_b = [[0.2, 0.8], [0.75, 0.25]]
    cpt_c = [[0.8, 0.2], [0.1, 0.9]]
    cpt_d = [[[0.95, 0.05], [0.9, 0.1]], [[0.8, 0.2], [0.0, 1.0]]]
    cpt_e = [[0.7, 0.3], [0., 1.0]]

    # nodes
    bn = TBN("cancer")
    A = Node("A", parents=[], cpt=cpt_a)
    B = Node("B", parents=[A], cpt=cpt_b)
    C = Node("C", parents=[A], cpt=cpt_c)
    D = Node("D", parents=[B, D], cpt=cpt_d)
    E = Node("E", parents=[E], cpt=cpt_e)
    nodes = [A, B, C, D, E]
    for n in nodes:
        bn.add(n)
    return bn

