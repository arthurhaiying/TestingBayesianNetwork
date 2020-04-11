import numpy as np
import tac
import examples.hiddenMarkov.hmm as hmm
import train.data as data
import utils.utils as u



#np.random.seed(2048)
f = open('logHMM.txt', 'w')
# open a log file


def logging(message):
	print(message)
	f = open('logHMM.txt', 'a')
	f.write(message + '\n')
	f.close()

def validateHMM(size, card, num_examples=10):
	# validate AC of HMM model by running query pr(X_t，Y[1:t]) compared to the forward algorithm
	transition = np.random.rand(card, card)
	transition_sum = np.sum(transition, axis=1, keepdims=True)
	transition = transition / transition_sum
	emission = np.random.rand(card, card)
	emission_sum = np.sum(emission, axis=1, keepdims=True)
	emission = emission / emission_sum
	# generate random transition and emission probabilities
	bn = hmm.getHMM(size, card, param=True, transition=transition, emission=emission)
	logging("Starting testing HMM of length {}".format(size))
	# define an hmm model with these parameters
	inputs = ['e_' + str(i) for i in range(size-1)]
	output = 'h_' + str(size-1)
	ac = tac.TAC(bn, inputs, output, trainable=False)
	# compile an ac that computes pr(X_T+1, Y[1:T])
	evidence_ac, evidence_dp = generate_hard_evidence(size-1, num_examples)
	labels_ac = ac.evaluate(evidence_ac)
	logging("ac labels: %s" %(labels_ac))
	labels_dp = []
	for evid in evidence_dp:
		label = hmm.predict(size, evid, transition, emission)
		labels_dp.append(label)
	labels_dp = np.stack(labels_dp)
	logging("forward labels: %s" %(labels_dp))
	if u.equal(labels_ac, labels_dp, tolerance=True):
		logging("Successfully validate first order HMM of length length {}\n".format(size))
	else:
		logging("Inconsistence queries for first order HMM of length {}\n".format(size))

def validateThirdOrderHMM(size, card, num_examples=10):
	# validate AC of HMM model by running query pr(X_t，Y[1:t]) compared to the forward algorithm
	transition = np.random.rand(card, card, card, card)
	transition_sum = np.sum(transition, axis=-1, keepdims=True)
	transition = transition / transition_sum
	emission = np.random.rand(card, card)
	emission_sum = np.sum(emission, axis=1, keepdims=True)
	emission = emission / emission_sum
	# generate random transition and emission probabilities
	bn = hmm.getNthOrderHMM(size, card, 3, param=True, transition=transition, emission=emission)
	# define an hmm model with these parameters
	logging("Start testing third order HMM of length {}".format(size))
	inputs = ['e_' + str(i) for i in range(size-1)]
	output = 'h_' + str(size-1)
	ac = tac.TAC(bn, inputs, output, trainable=False)
	# compile an ac that computes pr(X_T+1, Y[1:T])
	evidence_ac, evidence_dp = generate_hard_evidence(size-1, 1)
	labels_ac = ac.evaluate(evidence_ac)
	logging("ac labels: %s" %(labels_ac))
	labels_dp = []
	for evid in evidence_dp:
		label = hmm.predictThirdOrder(size, evid, transition, emission)
		labels_dp.append(label)
	labels_dp = np.stack(labels_dp)
	logging("forward labels: %s" %(labels_dp))
	if u.equal(labels_ac, labels_dp, tolerance=True):
		logging("Successfully validate third order HMM of length {}\n".format(size))
	else:
		logging("Inconsistence queries for third order HMM of length {}\n".format(size))


def generate_hard_evidence(size, num_examples):
	evidence_ac, evidence_dp = [], []
	for i in range(num_examples):
		arr = np.random.randint(2, size=size)
		evidence_dp.append(arr)
		evidence = np.zeros((size, 2))
		evidence[np.arange(size), arr] = 1
		evidence_ac.append(evidence)
	# remember to reshape evidence into ac inputs type
	evidence_ac = data.evd_row2col(evidence_ac)
	return evidence_ac, evidence_dp




