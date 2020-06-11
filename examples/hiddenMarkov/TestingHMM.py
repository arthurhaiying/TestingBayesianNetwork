import math
import random
import numpy as np

from tbn.tbn import TBN
from tbn.node import Node
from tac import TAC
import tbn.cpt as cpt
import utils.utils as u
import train.data as data

from itertools import product

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


class HiddenMarkovModel:

    def __init__(self, size, card, order=1, testing=False, evid_card=None, inputs=None, output=None, param=False, 
        transition1=None, transition2=None, emission=None, threshold=None,sel_type='linear',gamma_opt=None,default_gamma=1000):
        # create a testing HMM model
        assert order >= 1 and card >= 2
        assert size >= order
        self.size = size
        self.card = card
        if evid_card is None:
            evid_card = card
        self.evid_card = evid_card
        self.order = order
        self.testing = testing
        self.sel_type = sel_type
        self.gamma_opt = gamma_opt
        self.default_gamma = default_gamma
        # specify cpt selection method
        # initialize HMM size parameters
        if param:
            # if a parameterization provided for this HMM
            u.input_check(np.array(transition1).shape == (card,)*(order+1), "wrong size for transition matrix")
            u.input_check(np.array(emission).shape == (card, evid_card), "wrong size for emission matrix")
            if testing:
                # also check transition2 
                u.input_check(np.array(transition2).shape == (card,)*(order+1), "wrong size for transition2 matrix")
            if threshold is not None:
                # if threshold provided
                u.input_check(np.array(threshold).shape == (card,)*order, "wrong size for threshold matrix")
        # initialize HMM probabilities
        self.transition1 = transition1
        self.transition2 = transition2
        self.emission = emission
        self.threshold = threshold
        self._tbn = self.__getHMMNet(size, card, evid_card, order, testing, transition1, transition2, emission, threshold, default_gamma)
        self._tbn.set_select_type(sel_type)
        if gamma_opt is not None:
            self._tbn.set_gamma_option(gamma_opt)
        # compile an AC for query Pr(X_n|E[0:n-1])
        self.inputs = inputs if inputs is not None else ['e_' + str(j) for j in range(size-1)]
        self.output = output if output is not None else 'h_' + str(size-1)
        # ask query Pr(X_n|E[0:n-1]) by default
        self._tac_for_inference = TAC(self._tbn, inputs=self.inputs, output=self.output,trainable=False)
        #logging.info("Finish compileing tac for {}-order thmm of size {}".format(order, size))
        self._tac_for_learning = None

    @staticmethod
    def __getHMMNet(size, card, evid_card, order, testing, transition1, transition2, emission, threshold, gamma):
        # create a testing HMM net of order of size of cardinaility
        tbn = TBN(f'tHMM_{order}_{size}_{card}')
        hiddenNodeCache = []
        values = ['v'+str(i) for i in range(card)]
        evid_values = ['v'+str(i) for i in range(evid_card)]
        # create first N hidden nodes
        for i in range(order):
            name = 'h_'+str(i)
            parents = [hiddenNodeCache[j] for j in range(i)]
            cpt = np.ones((card,)*(i+1)) * (1./card)
            h_i = Node(name, values=values, parents=parents, testing=False, cpt=cpt)
            # first N nodes cannot be testing
            tbn.add(h_i)
            hiddenNodeCache.append(h_i)
        # create subsequent hidden nodes
        for i in range(order, size):
            name = 'h_'+str(i)
            parents = [hiddenNodeCache[j] for j in range(i-order, i)]
            if testing:
                h_i = Node(name, values=values, parents=parents, testing=True, cpt1=transition1, cpt2=transition2,threshold=threshold,gamma=gamma,cpt_tie="transition")
                # ToDo: add threshold
            else:
                h_i = Node(name, values=values, parents=parents, testing=False, cpt=transition1, cpt_tie="transition")
            tbn.add(h_i)
            hiddenNodeCache.append(h_i)
        # create evidence nodes
        for i in range(size):
            name = 'e_'+str(i)
            parents = [hiddenNodeCache[i]]
            e_i = Node(name, values=evid_values, parents=parents, testing=False, cpt=emission, cpt_tie="emission")
            tbn.add(e_i)
        # finish creating hmm
        #tbn.dot(view=True)
        #logging.info("Finish creating {}-order tHMM of size {}".format(order, size))
        return tbn

    def forward_tac(self, evidences):
        u.input_check(len(evidences) == len(self.inputs), "Number of TAC input does not match.")
        result = self._tac_for_inference.evaluate(evidences)
        return result

    def metric(self,evidences,marginals,metric_type):
        u.input_check(len(evidences) == len(self.inputs), "Number of TAC input does not match.")
        result = self._tac_for_inference.metric(evidences,marginals,metric_type=metric_type)
        print("The testing loss is %.5f" % result)
        return result
    
    def forward_dp(self, evidence):
        # use forward algorithm to compute pr(Xn|E1:n)
        assert self.order == 1
        size = self.size
        card = self.card
        marginals_table = np.zeros((size, card))
        # store marginal probability pr*(X_n, E[0:n])
        # base case: pr*(X0, E0)
        for state in range(card):
            state_e_0 = evidence[0]
            marginal_x_0 = (1./card) * self.emission[state][state_e_0]
            # pr*(X0, E0) = pr(X0)pr(E0|X0)
            marginals_table[0][state] = marginal_x_0
        # recursion
        for i in range(1, size):
            state_e_curr = evidence[i]
            marginal_x_prev = marginals_table[i-1]
            # precomputed marginals pr*(X_n-1, E[0:n-1])
            cond_x_prev = marginal_x_prev / np.sum(marginal_x_prev)
            # normalize to get precomputed conditionals pr*(X_n-1 | E[0:n-1])
            if self.testing:
                if self.sel_type == "linear":
                    selected_transition = self.transition2 + (self.transition1 - self.transition2) * (cond_x_prev.reshape((card, 1)))
                # use linear selection for cpts pr(X_n|X_n-1)
                elif self.sel_type == 'threshold':
                    indicator = cond_x_prev >= self.threshold
                    selected_transition = self.transition1*(indicator.reshape((card, 1))) + self.transition2*((1.0 - indicator).reshape((card, 1)))
                elif self.sel_type == 'sigmoid':
                    difference = cond_x_prev - self.threshold
                    gamma = 1000
                    def sigmoid(x):
                        y = 1.0 / (1.0 + np.exp(-x))
                        return y
                    indicator = sigmoid(gamma*difference)
                    selected_transition = self.transition1*(indicator.reshape((card, 1))) + self.transition2*((1.0 - indicator).reshape((card, 1)))
            else:
                selected_transition = self.transition1
            marginal_x_curr = marginal_x_prev.reshape((card, 1)) * selected_transition * self.emission[:,state_e_curr].reshape((1, card))
            # compute pr(X_n, X_n-1, E[0:n]) = pr(X_n-1, E[0:n-1]) * pr(X_n|X_n-1) * pr(E_n|X_n)
            marginal_x_curr = np.sum(marginal_x_curr, axis=0) # sum out X_n-1
            #print("the marginals: %s" %marginal_x_curr)
            marginals_table[i] = marginal_x_curr
        # now compute the query result
        marginal_x_N = marginals_table[size-1]
        cond_x_N = marginal_x_N / np.sum(marginal_x_N)
        return cond_x_N

    def sample_dp(self,num_examples):
        # sample labeled data corresponding to forward query
        evidences = data.evd_random(num_examples, [self.card]*self.size, hard_evidence=True)
        evidences_row = data.evd_col2row(evidences)
        marginals = []
        for row in evidences_row:
            l = [np.argmax(lambdas != 0) for lambdas in row]
            marginal = self.forward_dp(l)
            marginals.append(marginal)
            # compute query
        marginals = np.array(marginals)
        return evidences, marginals

    def learn(self,evidences,marginals,filename=None):
        u.input_check(len(evidences) == len(self.inputs), "Number of tac inputs does not match.")
        # remember to create a new tac for learning
        if self._tac_for_learning is None:
            self._tac_for_learning = TAC(self._tbn, inputs=self.inputs, output=self.output,trainable=True)
        #logging.info("Finish compiling tac for learning the forward query")
        #fit tac for learning the forward query
        self._tac_for_learning.fit(evidences, marginals,loss_type='CE',metric_type='CE',fname=filename)
        self._tac_for_inference = self._tac_for_learning # use the learned tac for inference
        #logging.info("Finish learning the forward query")

        
def play():
    transition1 = cpt.random(2, [2])
    transition2 = cpt.random(2, [2])
    emission = cpt.random(2, [2])
    threshold = np.random.rand(2)
    hmm = HiddenMarkovModel(8,2,order=1,testing=True,param=True, 
        transition1=transition1, transition2=transition2, emission=emission, threshold=threshold, sel_type='sigmoid')
    evidence = np.random.randint(2, size=8)
    evidence_ac = np.zeros((8, 2))
    evidence_ac[np.arange(8), evidence] = 1
    print("the tac evidence is: %s" % evidence_ac)
    evidence_ac = data.evd_row2col([evidence_ac]*1)
    query_tac = hmm.forward_tac(evidence_ac)
    print("The tac result is: %s" % query_tac)
    query_forward = hmm.forward_dp(evidence)
    print("The forward result is: %s" % query_forward)
    hmm.sel_type = "linear"
    query_forward2 = hmm.forward_dp(evidence)
    print("The forward result using linear selection is: %s" %query_forward2)
    if np.all(np.isclose(query_tac[0], query_forward, rtol=1e-3)):
        print("Successfully validate first order testing HMM")
    else:
        print("Inconsistent query result for testing HMM")

def validateTestingHMM(size, card, num_examples=10):
    transition1 = cpt.random(card, [card])
    transition2 = cpt.random(card, [card])
    emission = cpt.random(card, [card])
    logging.info("Staring testing for 1-hmm of size %s" %(size,))
    hmm = HiddenMarkovModel(size,card,order=1,testing=True,param=True, transition1=transition1, transition2=transition2, emission=emission)
    evidence_tac = data.evd_random(num_examples, [card]*size, hard_evidence=True)
    evidence_row = data.evd_col2row(evidence_tac)
    print("evidence row %s" % evidence_row)
    evidence_dense = []
    for row in evidence_row:
        l = [np.argmax(lambdas != 0) for lambdas in row]
        evidence_dense.append(l)
    print("hard evidence: %s" % evidence_dense)
    query_tac = hmm.forward_tac(evidence_tac)
    query_forward = []
    for evidence in evidence_dense:
        query_forward.append(hmm.forward_dp(evidence))
        # compute query using linear recursion
    query_forward = np.array(query_forward)
    logging.info("tac query: %s" % query_tac)
    logging.info("forward query: %s" % query_forward)
    if np.all(np.isclose(query_tac, query_forward, rtol=1e-3)):
        logging.info("Successfully validate 1-thmm of size %s" % (size,))
    else:
        logging.info("Inconsistent query result for 1-thmm of size %s" % (size,))

def testTBN():
    tbn = TBN("simple_net1")
    h_0 = Node("h_0", parents=[], testing=False)
    tbn.add(h_0)
    h_1 = Node("h_1", parents=[h_0],testing=False)
    tbn.add(h_1)
    h_2 = Node("h_2", values=["v0", "v1", "v2"], parents=[h_0, h_1], testing=False)
    tbn.add(h_2)
    h_3 = Node("h_3", parents=[h_2, h_1], testing=True)
    tbn.add(h_3)
    tac = TAC(tbn, inputs=["h_0"], output="h_3", trainable=False)

def test_training(size, card, num_examples):
    transition1 = cpt.random(card, [card])
    transition2 = cpt.random(card, [card])
    emission = cpt.random(card, [card])
    threshold = cpt.random2([card])
    hmm = HiddenMarkovModel(size,card,order=1,testing=True,param=True, 
        transition1=transition1, transition2=transition2, emission=emission, threshold=threshold, sel_type='sigmoid',gamma_opt='free')
    logging.info("Staring testing for 1-hmm of size %s" %(size,))
    evidences, marginals = hmm.sample_dp(num_examples)
    hmm.learn(evidences, marginals)
    print("Finish testing tac")


    






