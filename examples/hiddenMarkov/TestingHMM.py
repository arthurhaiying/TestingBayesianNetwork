import math
import random
import numpy as np

from tbn.tbn import TBN
from tbn.node import Node
from utils import utils as u

from itertools import product

def getTestingHMM(size, card, N, testing=False, param=False, transition1=None, transition2=None, emission=None):
    # define an N order testing HMM model of length size with cardinality hidden states
    assert size >= 2 and card >= 2 and N >= 1
    if param:
        u.input_check(np.array(transition1).shape == (card,) * (N + 1), "wrong size for transition matrix")
        u.input_check(np.array(emission).shape == (card, card), "wrong size for emission matrix")
        if testing:
            u.input_check(np.array(transition2).shape == (card,) * (N + 1), "wrong size for transition2 matrix")
        # check the size of transition and emission probabilities
    hmm = TBN(f'thmm_{N}_{size}')
    values = ['v' + str(i) for i in range(card)]
    hidden_nodes = []
    # store list of hidden nodes
    # add first N hidden nodes
    for i in range(N):
        name = 'h_' + str(i)
        parents = [hidden_nodes[j] for j in range(i)]
        cpt = (1./card) * np.ones(shape=(card,)*(i+1))
        # create a uniform conditional cpt
        hidden_i = Node(name, values=values, parents=parents, cpt=cpt)
        # the first N nodes cannot be testing
        hmm.add(hidden_i)
        hidden_nodes.append(hidden_i)
        # add hidden nodes
    # add the subsequent hidden nodes
    for i in range(N, size):
        name = 'h_' + str(i)
        parents = [hidden_nodes[j] for j in range(i-N, i)]
        if not testing:
            hidden_i = Node(name, values=values, parents=parents, cpt=transition1, cpt_tie="transition")
        else:
            hidden_i = Node(name, values=values, parents=parents, testing=True, cpt1=transition1, cpt2=transition2, cpt_tie="transition")
        hmm.add(hidden_i)
        hidden_nodes.append(hidden_i)
    # add evidence
    for i in range(size):
        name = 'e_' + str(i)
        parents = [hidden_nodes[i]]
        evidence_i = Node(name, values=values, parents=parents, cpt=emission, cpt_tie="emission")
        hmm.add(evidence_i)
    # finish defining the hmm
    # hmm.dot(view=True)
    print("Finish creating a {}-order testing hmm of length {} and cardinality {}".format(N, size, card))
    return hmm