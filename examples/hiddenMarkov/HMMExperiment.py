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
NUM_EXAMPLES = 16384
STEP_SIZE = 0.05
# CPT selection settings
SELECT_TYPE = 'sigmoid'
GAMMA_OPTION = 'tied'
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
    if card == 2:
        # use almost deterministic cpt
        yield (0.95,0.05)
        yield (0.05,0.95)
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
        return __enumerate_dist_rec(card, 1.0, step_size)

def enumerate_HMM(card,order,step_size):
    # enumerate possible model parameters for n order HMM
    # enumerate tranision prob by step size, use noisy sensor for emission prob
    non_diag_value = 0.01
    diag_value = 1-non_diag_value*(card-1)
    emission_cpt = [[diag_value if i == j else non_diag_value for j in range(card)] for i in range(card)]
    emission_cpt = np.array(emission_cpt)
    cards = (card,)*order # parant states
    for transition_cpt in enumerate_cpts(card,cards,step_size):
        # enumerate transition cpt for node card with parents cards
        yield (transition_cpt, emission_cpt)


def direct_sample(size,card,order,transition,emission):
    # sample data E[0:t-1] and H_t from n order HMM 
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
    for i in range(order,size-1):
        parents = hiddens[-order:] # get parent states of h_i
        cond_prob = transition[tuple(parents)]
        h_i = np.random.choice(card,p=cond_prob)  # sample h_i from pr(h_i|h[i-n:i])
        e_i = np.random.choice(card,p=emission[h_i]) # sample e_i from h_i
        hiddens.append(h_i)
        evidences.append(e_i)
    # now we sample h_t
    parents = hiddens[-order:]
    cond_prob = transition[tuple(parents)]
    h_t = np.random.choice(card,p=cond_prob)  # sample h_i from pr(h_i|h[i-n:i])
    return evidences, h_t

def direct_sample_training_set(size,card,order,transition,emission,num_trial):
    # sample num_examples data from HMM and save them to file
    fields = ['E_'+str(i) for i in range(size-1)]
    fields.append('H_'+str(size-1))
    evidences, marginals = [],[]
    # save the data in csv file
    dirname = os.path.abspath(os.getcwd())
    filename = 'training_data_{}.csv'.format(num_trial)
    filepath = os.path.join(dirname,'examples','hiddenMarkov','datasets',filename)
    with open(filepath,'w+') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for i in range(NUM_EXAMPLES):
            evidence,marginal = direct_sample(size,card,order,transition,emission)
            evidences.append(evidence)
            marginals.append(marginal)
            writer.writerow(evidence+[marginal])
            # sample data and save to file
    return evidences, marginals

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

def reshape_marginal(marginal, card):
    l = []
    for i in marginal:
        vector = np.zeros(card)
        vector[i] = 1.0
        l.append(vector)
        # convert to one hot label
    return np.array(l)

def cross_validate(hmm,evidences,marginals,num_folds):
    assert len(evidences) == len(marginals)
    num_records = len(evidences)
    num_inputs = len(evidences[0])
    assert num_inputs == hmm.size - 1
    indices = list(range(num_records))
    random.shuffle(indices)
    # shuffle the training data set
    partitions = []
    partition_size = int(num_records/num_folds)
    for start in range(0,num_records,partition_size):
        end = start + partition_size
        if end > num_records:
            end = num_records
        evid = [evidences[i] for i in indices[start:end]]
        marg = [marginals[i] for i in indices[start:end]]
        partitions.append((evid,marg))
        # split the dataset into num_fold partitions
    loss_avg = 0
    for i in range(num_folds):
        # for each trial
        print("Start fold {}...".format(i))
        testing_evid, testing_marg = partitions[i]
        training_evid, training_marg = [],[]
        for j, (evid,marg) in enumerate(partitions):
            if j == i:
                pass
            training_evid.extend(evid)
            training_marg.extend(marg)
            # prepare the training and testing data
        training_evid = reshape_evidence(training_evid,[hmm.card]*num_inputs)
        #print("num evidence is: %d" %len(training_evid))
        # reshape evidence to tac inputs
        training_marg = reshape_marginal(training_marg, hmm.card)
        testing_evid = reshape_evidence(testing_evid,[hmm.card]*num_inputs)
        testing_marg = reshape_marginal(testing_marg, hmm.card)
        # prepare the training and testing dataset
        hmm.learn(training_evid,training_marg)
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
    evidences,marginals = direct_sample_training_set(size,card,order,transition,emission,num_trial='test')
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
    













        
    
    
    
















    
    

        
        
    
    


