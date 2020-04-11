import math
import random
import numpy as np

from tbn.tbn import TBN
from tbn.node import Node
from utils import utils as u

from itertools import product

def getHMM(size, card, param=False, transition=None, emission=None):
    # define an HMM model of size nodes with card hidden states
    assert size >= 2 and card >= 2
    if param:
        assert transition is not None and emission is not None
        u.input_check(np.array(transition).shape == (card, card), f'wrong size for transition matrix')
        u.input_check(np.array(emission).shape == (card, card), f'wrong size for matrix')
        # check size for transition and emission matrix
    hmm = TBN(f'hmm_{size}')
    values = ['v' + str(i) for i in range(card)]
    hidden_nodes = [] 
    # store list of created hidden nodes
    for i in range(size):
        # add hidden node
        if i == 0:
            uniform_cpt = [1./card] * card;
            hidden_i = Node('h_0', values=values, parents=[], cpt=uniform_cpt)
            hmm.add(hidden_i)
            # notice that H0 is uniform if parametrized
            hidden_nodes.append(hidden_i)
        else:
            hidden_i = Node('h_' + str(i), values=values, parents=[hidden_nodes[i - 1]], cpt_tie="transition", cpt=transition)
            hmm.add(hidden_i)
            hidden_nodes.append(hidden_i)
        # add evidence node
        evidence_i = Node('e_' + str(i), values=values, parents=[hidden_nodes[i]], cpt_tie="emission", cpt=emission)
        hmm.add(evidence_i)
        # finish creating the hmm
    #hmm.dot(view=True)
    print("Finish creating HMM_{} with cardinality {}".format(size, card))
    return hmm


def predict(size, evidence, transition, emission):
    # do query Pr(X_T+1 | Y[0:T]) using the forward algorithm
    card = np.array(transition).shape[0]
    prob_x_T = forward_dp(size-1, evidence, transition, emission)
    prob_x_next = np.zeros(card)
    for state_next in range(card):
        sum = 0
        for state in range(card):
            sum += transition[state][state_next] * prob_x_T[state]
            # summation pr(X_T+1 | X_T) * pr(X_T, Y[0:T])
        prob_x_next[state_next] = sum
    # normalize
    prob_x_next = prob_x_next / np.sum(prob_x_next)
    return prob_x_next

def forward_dp(size, evidence, transition, emission):
    # compute the joint probability Pr(X_T, Y[0:T]) using the forward algorithm
    card = np.array(transition).shape[0]
    # get the cardinality of the hmm
    marginal_table = np.zeros(shape=(size, card))
    # store the joint probabilities
    for state in range(card):
        state_y0 = evidence[0]
        prob_x0 = (1./card) * emission[state][state_y0]
        #print("prob_x0 is: {}".format(prob_x0))
        # pr(x0, y0) = pr(x0)pr(y0|x0)
        marginal_table[0][state] = prob_x0
        # compute pr(X0, Y0)
    for t in range(1, size):
        for state in range(card):
            sum = 0
            for state_prev in range(card):
                sum += marginal_table[t-1][state_prev] * transition[state_prev][state]
                # summation of pr(X_(t-1), Y[1:t-1])pr(X_t|X_(t-1))
            state_y = evidence[t]
            prob_x = emission[state][state_y] * sum
            marginal_table[t][state] = prob_x
            # pr(X_t, Y[0:t]) = pr(Y_t|X_t) * summation
    # return an array of pr(X_T, Y[0:T])
    # print("probabilities: %s" %(marginal_table))
    return marginal_table[size-1]


def getNthOrderHMM(size, card, N, param=False, transition=None, emission=None):
    # define an N order HMM model of length size with cardinality hidden states
    assert size >= 2 and card >= 2 and N >= 1
    if param:
        u.input_check(np.array(transition).shape == (card,) * (N + 1), "wrong size for transition matrix")
        u.input_check(np.array(emission).shape == (2, 2), "wrong size for emission matrix")
        # check the size of transition and emission probabilities
    hmm = TBN(f'hmm_{N}_{size}')
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
        hmm.add(hidden_i)
        hidden_nodes.append(hidden_i)
        # add hidden nodes
    # add the subsequent hidden nodes
    for i in range(N, size):
        name = 'h_' + str(i)
        parents = [hidden_nodes[j] for j in range(i-N, i)]
        hidden_i = Node(name, values=values, parents=parents, cpt=transition, cpt_tie="transition")
        hmm.add(hidden_i)
        hidden_nodes.append(hidden_i)
    # add evidence
    for i in range(size):
        name = 'e_' + str(i)
        parents = [hidden_nodes[i]]
        evidence_i = Node(name, values=values, parents=parents, cpt=emission, cpt_tie="emission")
        hmm.add(evidence_i)
    # finish defining the hmm
    #   hmm.dot(view=True)
    print("Finish creating a {}-order hmm of length {} and cardinality {}".format(N, size, card))
    return hmm


def predictThirdOrder(size, evidence, transition, emission):
    # do query pr(Xt+1 | y[0:t])
    card = np.array(transition).shape[0]
    marginal_t = forwardThirdOrder(size-1,evidence, transition, emission)
    # compute joint probability pr(X[t-2:t], Y[0:t])
    marginal_next = np.expand_dims(marginal_t, axis=-1)*transition
    marginal_next = np.sum(marginal_next, axis = tuple(range(marginal_next.ndim - 1)))
    # normalize the probabilities
    marginal_next = marginal_next / np.sum(marginal_next)
    return marginal_next


def forwardThirdOrder(size, evidence, transition, emission):
    # compute joint probability pr(X[t-2:t], Y[0:t]) in third order hmm
    card = np.array(transition).shape[0]
    marginal_table = np.zeros((size, card, card, card))
    # store probability pr(X[t-2:t], Y[0:t])
    # first three nodes
    for t in range(3):
        for states in product(range(card), repeat=t+1):
            # for each state of (X0, ..., Xt)
            marginal = (1./card)**(t+1)
            # uniform prob pr(X0,..., Xt)
            for j in range(t):
                x_j = states[j]
                y_j = evidence[j]
                marginal *= emission[x_j][y_j]
                # pr(X0-t, Y0-t) = pr(x0-t) prod(pr(y_i|x_i))
            states += (0,) * (3 - len(states))
            # pad 0 to the last index
            marginal_table[t][states] = marginal
    # the following nodes
    for end in range(3, size):
        start = end - 2
        # normal recursion
        if start >= 3:
            for states_curr in product(range(card), repeat=3):
                marginal = 1;
                for state_x, state_y in zip(states_curr, evidence[start:end+1]):
                    marginal *= emission[state_x][state_y]
                # marginal = pr(Yt|Xt)pr(Yt+1|Xt+1)pr(Yt+2|Xt+2)
                sum = 0
                for states_prev in product(range(card), repeat=3):
                    summend = marginal_table[start-1][states_prev]
                    # summend = pr(X[t-3:t-1], Y[0:t-1])pr(Xt|parents)pr(Xt+1|parent)pr(Xt+2|parents)
                    for i in range(3):
                        state_x = states_curr[i]
                        states_parents = states_prev[i:] + states_curr[:i]
                        summend *= transition[states_parents][state_x]
                    sum += summend
                marginal *= sum
                # marginal = pr(Yt|Xt)pr(Yt+1|Xt+1)pr(Yt+2|Xt+2) sum pr(X[t-3:t-1], Y[0:t-1])pr(Xt|parents)pr(Xt+1|parent)pr(Xt+2|parents)
                marginal_table[end][states_curr] = marginal
        else:
            for states_curr in product(range(card), repeat=3):
                marginal = 1;
                for state_x, state_y in zip(states_curr, evidence[start:end+1]):
                    marginal *= emission[state_x][state_y]
                    # marginal = pr(Yt|Xt)pr(Yt+1|Xt+1)pr(Yt+2|Xt+2)
                sum = 0
                for states_prev in product(range(card), repeat=start):
                    index = states_prev + (0,)*(3 - len(states_prev))
                    summend = marginal_table[start-1][index]
                    for i in range(3):
                        state_x = states_curr[i]
                        if start + i < 3:
                            summend *= 1./card
                            # x_i still has uniform cpt
                        else:
                            states_parents = states_prev[-(3-i):] + states_curr[:i]
                            summend *= transition[states_parents][state_x]
                    sum += summend
                marginal *= sum
                # marginal = pr(Yt|Xt)pr(Yt+1|Xt+1)pr(Yt+2|Xt+2) sum pr(X[0:t-1], Y[0:t-1])pr(Xt|parents)pr(Xt+1|parent)pr(Xt+2|parents)
                marginal_table[end][states_curr] = marginal
            # finish calculating the marginal probabilities
    #print("the probabilities : %s" % marginal_table[size-1])
    return marginal_table[size-1]






















if __name__ == '__main__':
    getNthOrderHMM(5, 2, 2)








