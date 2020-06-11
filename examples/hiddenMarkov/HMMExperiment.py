from examples.hiddenMarkov.TestingHMM import HiddenMarkovModel
import numpy as np
import utils.utils as u
from itertools import product
import os.path
import csv
import random
import train.data as data
import tbn.cpt as cpt

# training data settings
USE_DETERMINISTIC_CPT = 0
NUM_EXAMPLES = 16384
STEP_SIZE = 0.2
TRANSITION_STEP_SIZE = 0.02
EMISSION_STEP_SIZE = 0.02
# CPT selection settings
SELECT_TYPE = 'sigmoid'
GAMMA_OPTION = 'fixed'
DEFAULT_GAMMA_VALUE = 1000
# training settings
NUM_FOLDS = 5

def enumerate_cpts(card,cards,step_size=0.05):
    # enumerate all cpts for node card with parents cards
    if not cards:
        yield from enumerate_dist(card,step_size)
    else:
        card0 = cards[0]
        sub_cpts = [enumerate_cpts(card, cards[1:], step_size) for _ in range(card0)]
        for cpt in product(*sub_cpts):
            cpt = np.array(cpt)
            yield cpt

# generate all distribution of card states with step size
def enumerate_dist(card,step_size):
    if USE_DETERMINISTIC_CPT:
        # use almost deterministic cpt
        for i in range(card):
            cpt = [0.05]*card 
            cpt[i] = 1.0-0.05*(card-1)
            yield cpt
    else:
        def __enumerate_dist_rec(card,sum,step_size):
            if card == 1:
                dist = (sum,)
                yield dist
            else:
                for head in np.arange(0,sum+step_size,step_size):
                    if head > sum:
                        head = sum # small trick to include the end
                    for tail in __enumerate_dist_rec(card-1,sum-head,step_size):
                        dist = (head,) + tail
                        yield dist
        yield from __enumerate_dist_rec(card, 1.0, step_size)


def enumerate_transitions1(card,step_size):
    # enumerate first order transition cpts that simulates a loop
    # hidden node h_i = k implies h_i+1 = k + 1 with high prob
    min_pos_rate = 0.8
    for pos_rate in np.arange(1.0,min_pos_rate-step_size,-step_size):
        false_pos_rate = (1.0-pos_rate)/(card-1)
        cpt = np.ones((card,card))*false_pos_rate
        indices = tuple(range(1,card)) + (0,)
        cpt[np.arange(card),indices] = pos_rate
        yield cpt


def enumerate_sensors(card, step_size):
    # enumerate almost deterministic emission cpts
    min_pos_rate = 0.8
    for pos_rate in np.arange(1.0,min_pos_rate-step_size,-step_size):
        false_pos_rate = (1.0-pos_rate)/(card-1)
        cpt = [[pos_rate if i == j else false_pos_rate for j in range(card)] for i in range(card)]
        cpt = np.array(cpt)
        yield cpt

def enumerate_HMM(card,order,step_size):
    # enumerate possible model parameters for n order HMM
    # enumerate tranision prob by step size, use noisy sensor for emission prob
    cards = (card,)*order # parant states
    for transition_cpt in enumerate_transitions1(card,TRANSITION_STEP_SIZE):
        for emission_cpt in enumerate_sensors(card,EMISSION_STEP_SIZE):
        # enumerate transition cpt for node card with parents cards
            yield (transition_cpt, emission_cpt)


def direct_sample(size,card,order,transition,emission,missing_card):
    # if not missing card, sample Pr(H_t|E[0:t-1]) otherwise sample Pr(E_t|E[0:t-1])
    u.input_check(np.array(transition).shape == (card,)*(order+1), "wrong size for transition matrix")
    u.input_check(np.array(emission).shape == (card, card), "wrong size for emission matrix")
    # sample the first n nodes
    hiddens, evidences = [], []
    for i in range(order):
        h_i = np.random.randint(card)
        e_i = np.random.choice(card,p=emission[h_i]) # sample e_i from pr(e_i|h_i)
        hiddens.append(h_i)
        evidences.append(e_i)
    # sample the following nodes
    for i in range(order,size):
        parents = hiddens[-order:] # get parent states of h_i
        cond_prob = transition[tuple(parents)]
        h_i = np.random.choice(card,p=cond_prob)  # sample h_i from pr(h_i|h[i-n:i])
        e_i = np.random.choice(card,p=emission[h_i]) # sample e_i from h_i
        hiddens.append(h_i)
        evidences.append(e_i)
    # if missing card, predict the next hidden state, so return E[0:t-1] and h_t
    if not missing_card:
        return evidences[:-1],hiddens[-1]
    else:
        # otherwise return E[0:t-1] and E[t]
        return evidences[:-1],evidences[-1]

def direct_sample_training_set(size,card,order,transition,emission,missing_card,num_trial):
    # if not missing card, sample E[0:t-1] and h_t otherwise sample E[0:t-1] and E_t
    # sample num_examples data from HMM and save them to file
    fields = ['E_'+str(i) for i in range(size-1)]
    label = 'H_'+str(size-1) if not missing_card else 'E_'+str(size-1)
    # if not missing card predict the next hidden state otherwise the next evidence
    fields.append(label)
    evidences, marginals = [],[]
    # save the data in csv file
    dirname = os.path.abspath(os.getcwd())
    filename = 'training_data_{}.csv'.format(num_trial)
    filepath = os.path.join(dirname,'examples','hiddenMarkov','datasets',filename)
    with open(filepath,'w+') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for i in range(NUM_EXAMPLES):
            evidence,marginal = direct_sample(size,card,order,transition,emission,missing_card=missing_card)
            evidences.append(evidence)
            marginals.append(marginal)
            writer.writerow(evidence+[marginal])
            # sample data and save to file
    evidences = reshape_evidence(evidences,[card]*(size-1))
    marginals = reshape_marginal(marginals,card)
    # reshape evidence and label for tac inputs
    return evidences,marginals
    

# convert evidence into shape num_inputs * num_examples * num_states
def reshape_evidence(evidences,cards):
    assert len(evidences[0]) == len(cards)
    evidences_tac = []
    for evidence in evidences:
        vectors = []
        for i,state in enumerate(evidence):
            vector = np.zeros(cards[i])
            vector[state] = 1.0
            vectors.append(vector)
        # convert hard evidence into one hot vector
        evidences_tac.append(vectors)
    # convert evidence into col based
    evidences_tac = data.evd_row2col(evidences_tac)
    return evidences_tac

# concatenate list of evidences. Need to concatenate lambda array of each input
def concate_evidences(evidences):
    num_inputs = len(evidences[0])
    result = []
    for i in range(num_inputs):
        inputs = tuple([evidence[i] for evidence in evidences])
        inputs = np.concatenate(inputs)
        result.append(inputs)
    return result

def concate_marginals(marginals):
    return np.concatenate(tuple(marginals))

def reshape_marginal(marginal, card):
    l = []
    for i in marginal:
        vector = np.zeros(card)
        vector[i] = 1.0
        l.append(vector)
        # convert to one hot label
    return np.array(l)

def cross_validate(hmm,evidences,marginals,num_folds,num_trial):
    # assume that evidences and marginals have been reshaped for tac inputs
    assert evidences[0].shape[0] == marginals.shape[0]
    num_records = evidences[0].shape[0]
    num_inputs = len(evidences)
    assert num_inputs == hmm.size - 1
    evidences, marginals = data.__shuffle(evidences,marginals)
    # shuffle evidencs and marginals according to indices
    evid_partitions = []
    marg_partitions = []
    partition_size = int(num_records/num_folds)
    for start in range(0,num_records,partition_size):
        end = start + partition_size
        if end > num_records:
            end = num_records
        evid = data.evd_slice(evidences,start,end)
        marg = marginals[start:end]
        evid_partitions.append(evid)
        marg_partitions.append(marg)
        # split the dataset into num_fold partitions
    loss_avg = 0
    # path to save learned ACs and TACS
    workdir = os.getcwd()
    dirname = os.path.join(workdir,'logs','cpts','trial%d'%num_trial)
    ac_dirname = os.path.join(dirname,'AC')
    tac_dirname = os.path.join(dirname,'TAC')
    if not os.path.exists(dirname):
        try:
            os.makedirs(ac_dirname)
            os.makedirs(tac_dirname)
        except OSError as err:
            print("Failed to create directory for saving tacs: %s" %str(err))
            exit(1)
    for i in range(num_folds):
        # for each trial
        print("Start fold {}...".format(i))
        testing_evid = evid_partitions[i]
        testing_marg = marg_partitions[i]
        train_evid_list = evid_partitions[:i] + evid_partitions[i+1:]
        train_marg_list = marg_partitions[:i] + marg_partitions[i+1:]
        training_evid = concate_evidences(train_evid_list)
        training_marg = concate_marginals(train_marg_list)
        # prepare the training and testing dataset
        if hmm.testing:
            filename = os.path.join(tac_dirname,'TAC_%d.txt'%i)
        else:
            filename = os.path.join(ac_dirname,'AC_%d.txt'%i)
            # save learn cpts to file
        hmm.learn(training_evid,training_marg,filename=filename)
        loss = hmm.metric(testing_evid,testing_marg,metric_type='CE')
        loss_avg += loss
        print("Finish fold {}...".format(i))
    loss_avg /= num_folds
    print("The average cross validation loss is %0.5f" % loss_avg)
    return loss_avg


def test_cross_validate(size,card,order):
    transition = cpt.random(card, [card]*order)
    emission = cpt.random(card, [card])
    hmm = HiddenMarkovModel(size,card,order=1,testing=True,sel_type=SELECT_TYPE,gamma_opt=GAMMA_OPTION)
    evidences,marginals = direct_sample_training_set(size,card,order,transition,emission,missing_card=False,num_trial='test')
    loss = cross_validate(hmm,evidences,marginals,num_folds=NUM_FOLDS)
    



'''
def run_single(size,card,order,transition,emission,queue,num_trial):
    # given a parameterization of n order testing hmm, compare tac and ac performance
    hmm = HiddenMarkovModel(size,card,order=1,testing=False) # define first order hmm
    thmm = HiddenMarkovModel(size,card,order=1,testing=True,sel_type=SELECT_TYPE,gamma_opt=GAMMA_OPTION) # define first order testing hmm
    # direct sample the training set
    evidences,marginals = direct_sample_training_set(size,card,order,transition,emission,num_trial=1)
    loss_ac = cross_validate(hmm,evidences,marginals,num_folds=5) # train hmm 
    loss_tac = cross_validate(thmm,evidences,marginals,num_folds=5) # train testing hmm
    print("Trial {}: the AC loss is {.5f} and the TAC loss is {.5f}".format(num_trial,loss_ac,loss_tac))
    result = (num_trial,loss_ac,loss_tac)
    queue.put(result)
def run_master(size,card,order,num_workers):
    '''