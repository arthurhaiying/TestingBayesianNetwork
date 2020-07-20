import numpy as np
import utils.utils as u
from itertools import product
import itertools as iter
import os.path
import csv
import random
import train.data as data
import tbn.cpt as cpt

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

EPSILON = np.finfo('float32').eps

def enumerate_cpts(card,cards,step_size,is_deterministic):
    # enumerate all cpts for node card with parents cards
    if not cards:
        yield from enumerate_dist(card,step_size,is_deterministic)
    else:
        card0 = cards[0]
        sub_cpts = [enumerate_cpts(card, cards[1:], step_size, is_deterministic) for _ in range(card0)]
        for cpt in product(*sub_cpts):
            cpt = np.array(cpt)
            yield cpt

# generate all distribution of card states with step size
def enumerate_dist(card,step_size,is_deterministic):
    #print("Start enumerating dist")
    if is_deterministic:
        # use almost deterministic cpt
        max_pos_rate = 1.0
        min_pos_rate = 0.8
        for pos_rate in np.arange(max_pos_rate,min_pos_rate-step_size,-step_size):
            if pos_rate < min_pos_rate:
                pos_rate = min_pos_rate
            false_pos_rate = (1.0-pos_rate)/(card-1)
            for i in range(card):
                cpt = [false_pos_rate]*card 
                cpt[i] = pos_rate
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
        #print("Finish enumerating dist")

def enumerate_dist_shuffled(card,step_size,is_deterministic):
    dists = list(enumerate_dist(card,step_size,is_deterministic))
    random.shuffle(dists)
    for dist in dists:
        yield dist

def enumerate_cpts_shuffled(card,cards,step_size,is_deterministic,dist_fun=None):
    # enumerate all cpts for node card with parents cards
    if dist_fun is None:
        dists = list(enumerate_dist(card,step_size,is_deterministic))
    else:
        # use fun to filter possible dists
        dists = list(filter(dist_fun,enumerate_dist(card,step_size,is_deterministic)))
        print("Length of dists %d"%len(dists))
    # list of possible dists of length card 
    def __enum_dist_shuffled():
        # enumerate dists in random order
        indices = list(range(len(dists)))
        random.shuffle(indices)
        for index in indices:
            yield dists[index]
            # shuffle index instead of the actual list
    num_dists = 1
    for card in cards:
        num_dists *= card
    dists_generator = [__enum_dist_shuffled() for _ in range(num_dists)]
    for dists in product(*dists_generator):
        cpt = np.array(dists)
        cpt = cpt.reshape(tuple(cards)+(card,))
        yield cpt

def enumerate_transitions1(card,order,step_size):
    # enumerate transition cpts where each cond dist is almost deterministic
    cards = [card]*order
    yield from enumerate_cpts(card,cards,step_size,is_deterministic=True)

def enumerate_transitions2(card,step_size):
    # enumerate first order transition cpts that simulates a loop
    # hidden node h_i = k implies h_i+1 = k + 1 with high prob
    max_pos_rate = 1.0
    min_pos_rate = 0.8
    for pos_rate in np.arange(max_pos_rate,min_pos_rate-step_size,-step_size):
        if pos_rate < min_pos_rate:
            pos_rate = min_pos_rate
        false_pos_rate = (1.0-pos_rate)/(card-1)
        cpt = np.ones((card,card))*false_pos_rate
        indices = tuple(range(1,card)) + (0,)
        cpt[np.arange(card),indices] = pos_rate
        yield cpt

def enumerate_sensors1(card):
    # use single almost deterministic sensor
    false_pos_rate = 0.01
    pos_rate = 1-false_pos_rate*(card-1)
    emission_cpt = [[pos_rate if i == j else false_pos_rate for j in range(card)] for i in range(card)]
    emission_cpt = np.array(emission_cpt)
    yield emission_cpt

def gen_mixed_transition_cpt(card,order,components,weights):
    # generate a multinomial mixture distribution for transition cpt
    u.input_check(len(weights) == order, "Length of weights does not match.")
    u.input_check(np.isclose(np.sum(weights),1.0), "Weights for multinomial components are not normalized.")
    u.input_check(len(components) == order, "Number of multinomial components does not match.")
    for component in components:
        u.input_check(component.shape == (card,card), "Invalid shape for multinomial components.")
        u.input_check(np.allclose(np.sum(component,axis=-1), 1.0), "Slice of multinomial components is not normalized.")
    # check input parameters
    cpt = np.zeros((card,)*(order+1))
    for indices in product(range(card),repeat=order):
        cond = np.zeros(card)
        for index, weight, component in zip(indices,weights,components):
            cond += weight * component[index]
            # sum up compoents from parents
        cpt[indices] = cond
    return cpt

def gen_loop_cpts(card,order,pos_rate):
    # generate cpts according to all possible combinations of unique loop paths
    all_loops = iter.permutations(range(1,card))
    for loops in iter.combinations(all_loops,r=order):
        # choose order unique loops
        false_pos_rate = (1.0-pos_rate)/(card-1)
        # compute the false pos rate
        loop_cpts = []
        for loop in loops: 
            def gen_cpt_from_loop(loop):
                assert len(loop) == card
                loop_cpt = np.ones((card,card))*false_pos_rate
                for i,start in enumerate(loop):
                    end = loop[i+1] if i != len(loop)-1 else 0
                    loop_cpt[start][end] = pos_rate
                    # set cpt according to loop transition
                return loop_cpt
            loop = (0,) + loop
            loop_cpt = gen_cpt_from_loop(loop)
            loop_cpts.append(loop_cpt)
            # make cpts according to loop paths
        yield loop_cpts

def enumerate_transitions3(card,order,num_examples):
    # generate transition cpts that is a multinomial mixture where components are randomly sampled
    if order == 2:
        weights_generator = ((0.5,0.5),(0.6,0.4),(0.7,0.3),(0.8,0.2),(0.9,0.1),(1.0,0.0))
    elif order == 3:
        weights_generator = ((0.3,0.3,0.4),(0.4,0.3,0.3),(0.5,0.3,0.2),(0.6,0.3,0.1),(0.7,0.3,0.0))
    else:
        weights_generator = ((1.0/order)*order,)
        #each parent equally contribute to the transition cpt
    for _ in range(num_examples):
        components = []
        #state of each parent equally contribute to the transition cpt
        #low, high = 0.0, 1.0 # sample weights from uniform(0.01,1)
        for _ in range(order):
            #component = np.random.uniform(low=low,high=high,size=(card,card)) 
            #component = component/np.sum(component,axis=-1,keepdims=True)
            component = cpt.random(card,[card])
            components.append(component)
        # get mixture transiion cpt
        for weights in weights_generator:
            transition_cpt = gen_mixed_transition_cpt(card,order,components,weights)
            yield transition_cpt

def enumerate_transitions4(card,order,step_size):
    # generate transition cpts that is a multinomial mixture where each component is almost deterministic
    if order == 2:
        weights_generator = ((0.5,0.5),(0.6,0.4),(0.7,0.3),(0.8,0.2),(0.9,0.1),(1.0,0.0))
    elif order == 3:
        weights_generator = ((0.3,0.3,0.4),(0.4,0.3,0.3),(0.5,0.3,0.2),(0.6,0.3,0.1),(0.7,0.3,0.0))
    else:
        weights_generator = ((1.0/order)*order,)
        #each parent equally contribute to the transition cpt
    components_generator = [enumerate_cpts(card,[card],step_size,is_deterministic=True) for _ in range(order)]
    for components in product(*components_generator):
        for weights in weights_generator:
            transition_cpt = gen_mixed_transition_cpt(card,order,list(components),weights)
            yield transition_cpt

def enumerate_transitions5(card,order):
    # generate mixture transition cpts where each subcpt is a unique loop path
    if order == 2:
        weights_generator = ((0.5,0.5),(0.6,0.4),(0.7,0.3),(0.8,0.2),(0.9,0.1),(1.0,0.0))
    elif order == 3:
        weights_generator = ((0.8,0.2,0.0),(0.4,0.3,0.3),(0.5,0.3,0.2),(0.6,0.3,0.1),(0.7,0.3,0.0))
    else:
        weights_generator = ((1.0/order)*order,)
        #each parent equally contribute to the transition cpt
    pos_rate = 1.0
    for loop_cpts in gen_loop_cpts(card,order,pos_rate):
        for weights in weights_generator:
            transition_cpt = gen_mixed_transition_cpt(card,order,loop_cpts,weights)
            yield transition_cpt


def gen_group_cpts(card,order,pos_rate):
    # generate transition cpts where different parent states in the same group maps to as different group as possible
    assert order >= 2
    false_pos_rate = (1.0-pos_rate) / (card-1)
    sub_cpt_length = card**(order-1)
    sub_cpt = list(range(card)) * (sub_cpt_length // card)
    def make_sub_cpt(sub_cpt):
        # convert sub cpt for each group into factors
        sub_cpt = np.array(sub_cpt).reshape((card,)*(order-1))
        res = np.ones((card,)*order) * false_pos_rate
        for index, value in np.ndenumerate(sub_cpt):
            res[index][value] = pos_rate
        res = np.expand_dims(res,axis=-2)
        return res
    # enumerate sub_cpt for each group
    sub_cpt_generators = [map(make_sub_cpt, iter.permutations(sub_cpt)) for _ in range(card)]
    for sub_cpts in product(*sub_cpt_generators):
        cpt = np.concatenate(sub_cpts,axis=-2)
        yield cpt


def enumerate_transitions6(card,order):
    # enumerate transition cpts where states in the same group maps to as different group as possible
    pos_rate = 1.0
    yield from gen_group_cpts(card,order,pos_rate)

def enumerate_one_transition():
    cpt1 = np.zeros((5,5))
    cpt1[(0,1,2,3,4),(1,2,3,4,0)] = 1.0
    yield cpt1
    cpt2 = np.zeros((5,5))
    cpt2[(0,1,2,3,4),(1,2,0,4,3)] = 1.0
    yield cpt2

def enumerate_sensors2(card, step_size):
    # enumerate almost deterministic emission cpts
    max_pos_rate = 1.0
    min_pos_rate = 0.8
    for pos_rate in np.arange(max_pos_rate,min_pos_rate-step_size,-step_size):
        if pos_rate < min_pos_rate:
            pos_rate = min_pos_rate
        false_pos_rate = (1.0-pos_rate)/(card-1)
        cpt = [[pos_rate if i == j else false_pos_rate for j in range(card)] for i in range(card)]
        cpt = np.array(cpt)
        yield cpt


def run_first_order_markov_chain(card,init_probs,transition,step_size):
    # compute prob of states for all time steps over chain
    u.input_check(init_probs.size == card, "Invalid size of initial probs")
    u.input_check(transition.shape == (card,card), "Invalid size of transition matrix")
    states = []
    state_curr = init_probs
    states.append(state_curr)
    for _ in range(step_size):
        # compute prob of next states 
        state_next = np.matmul(state_curr,transition)
        states.append(state_next)
        state_curr = state_next
    states = np.stack(states)
    return states

def __run_Nth_order_markov_chain_baseline(card,order,init_probs,transition,step_size):
    # for n order chain, compute prob of states (X1,X2,...,Xn) by reducing to first order chain 
    assert order >= 1
    u.input_check(init_probs.shape == (card,)*order, "Invalid size of initial probs")
    u.input_check(transition.shape == (card,)*(order+1), "Invalid size of transition matrix")
    card_flattened = card**order
    init_probs_flattened = init_probs.flatten()
    transition_flattened = np.zeros((card,)*order*2)
    for index,value in np.ndenumerate(transition):
        index_flattened = index[:-1] + index[1:]
        transition_flattened[index_flattened] = value
        # get equivalent first order transition
    transition_flattened = transition_flattened.reshape(card_flattened,card_flattened)
    states_flattened = run_first_order_markov_chain(card_flattened,init_probs_flattened,transition_flattened,step_size)
    states = states_flattened.reshape((-1,)+(card,)*order)
    return states

def run_Nth_order_markov_chain(card,order,init_probs,transition,step_size):
    # for n order chain, compute prob of states (X1,X2,...,Xn) by transition step by step
    assert order >= 1
    u.input_check(init_probs.shape == (card,)*order, "Invalid size of initial probs")
    u.input_check(transition.shape == (card,)*(order+1), "Invalid size of transition matrix")
    states = []
    state_curr = init_probs
    states.append(state_curr)
    for _ in range(step_size):
        state_next = np.sum(np.expand_dims(state_curr,axis=-1)*transition,axis=0)
        states.append(state_next)
        state_curr = state_next
        # pr(X2X3X4) = sum_{X1}pr(X1X2X3)pr(X4|X1X2X3)
    states = np.stack(states)
    return states

def __entropy(dist):
    safe_dist = np.where(np.equal(dist, 0.0), EPSILON, dist)
    entropy = np.sum(-safe_dist*np.log2(safe_dist))
    return entropy
    
def __cond_mutual_info(joint, dist1, dist2):
    #print("joint: %s dist1: %s dist2: %s" %(joint.shape,dist1.shape,dist2.shape))
    u.input_check(joint.shape == dist1.shape and dist1.shape[-2:] == dist2.shape, "Invalid size of distributions")
    safe_dist1 = np.where(np.equal(dist1, 0.0), EPSILON, dist1)
    safe_dist2 = np.where(np.equal(dist2, 0.0), EPSILON, dist2)
    # replace zero prob in dists with very small value
    log_diff = np.log(safe_dist1) - np.log(np.expand_dims(safe_dist2,axis=0))
    res = np.sum(joint*log_diff)
    # loss = sum pr(x1x2x3)(log(pr(X3|X1X2) - log(pr(X3|X2))))
    return res

def __cond_mutual_info_from_transition(prior,transition):
    u.input_check(prior.shape == transition.shape[:-1], "Invalid size of distributions")
    joint = np.expand_dims(prior,axis=-1) * transition # pr(X1X2X3) = pr(X1X2)pr(X3|X1X2)
    dist2 = np.sum(joint,axis=tuple(np.arange(joint.ndim-2))) # pr(X2X3) = sum pr(X1X2X3)
    norm = np.sum(dist2,axis=-1)
    safe_norm = np.where(np.equal(norm,0.0), EPSILON, norm)
    dist2 = dist2 / np.expand_dims(safe_norm,axis=-1) # pr(X3|X2) = pr(X2X3)/pr(X2)
    mutual_info = __cond_mutual_info(joint, transition, dist2)
    return mutual_info

def get_cond_mutual_info_for_markov_chain(size,card,order,transition):
    assert order >= 1
    assert size > order
    #u.input_check(init_probs.shape == (card,)*order, "Invalid size of initial probs")
    u.input_check(transition.shape == (card,)*(order+1), "Invalid size of transition matrix")
    step_size = size - order
    # use uniform initial probabilities
    init_probs = np.ones((card,)*order) * 1.0/(card**order)
    states_prior = run_Nth_order_markov_chain(card,order,init_probs,transition,step_size)
    states_prior = np.mean(states_prior, axis=0)
    # compute average prob of states over all time steps
    mutual_info = __cond_mutual_info_from_transition(states_prior,transition)
    return mutual_info

def __cond_mutual_info_from_group(card,prior,transition,group_states):
    # compute individual mutual information for a group containing certain states
    # group states is list of state indices in this group 
    u.input_check(prior.size == card, "Invalid size of prior prob") 
    u.input_check(transition.shape == (card,card), "Invalid size of transition matrix")
    joint = np.expand_dims(prior,axis=-1)*transition
    group_prior = np.sum(prior[group_states])  # pr(X1={v1,v2})=pr(X1=v1)+pr(X1=v2)
    group_joint = np.sum(joint[tuple(group_states),:],axis=0) #pr(X2X1={v1,v2})
    group_transition = group_joint / group_prior
    safe_transition = np.where(np.equal(transition,0.0), EPSILON, transition)
    safe_group_transition = np.where(np.equal(group_transition,0.0), EPSILON, group_transition)
    mutual_info = joint * (np.log(safe_transition) - np.log(np.expand_dims(safe_group_transition,axis=0)))
    mutual_info = np.sum(mutual_info[[group_states]])
    return mutual_info

def get_cond_mutual_info_for_missing_one_card_chain(size,card,transition):
    # for first order chain missing one state, compute mutual info for all possible groups of two states
    # find the group with min mutual info
    u.input_check(transition.shape == (card,card), "Invalid size of transition matrix")
    uniform_dist = np.ones(card)*(1.0/card)
    states_prior = run_first_order_markov_chain(card,uniform_dist,transition,step_size=size-1)
    states_prior = np.mean(states_prior,axis=0)
    mutual_infos = []
    groups = []
    for group in iter.combinations(range(card),2):
        # group any two states
        mutual_info = __cond_mutual_info_from_group(card,states_prior,transition,list(group))
        groups.append(group)
        mutual_infos.append(mutual_info)
    # find the minimum mutual info value
    min_mutual_info = min(mutual_infos)
    # find groups whose mutual info is close to minimum
    MUTUAL_INFO_TOL = 1e-3
    min_groups = []
    for group,mutual_info in zip(groups,mutual_infos):
        if abs(mutual_info-min_mutual_info) <= MUTUAL_INFO_TOL:
            min_groups.append(group)
    # find the minimum mutual info and groups
    return min_mutual_info,min_groups

def enumerate_transitions7(size,card,order,threshold,step_size):
    # enumerate the whole CPT space and choose cpts with mutual info above the threshold
    cards = (card,)*order
    transition_cpts = enumerate_cpts_shuffled(card,cards,step_size=step_size,is_deterministic=False) 
    count, cpt_count = 0, 0
    for transition_cpt in transition_cpts:
        count += 1
        mutual_info = get_cond_mutual_info_for_markov_chain(size,card,order,transition_cpt)
        if mutual_info >= threshold:
            cpt_count += 1
            yield transition_cpt,mutual_info
            # choose cpts with mutual info exceeds the threshold
        if count % 100000 == 0:
            logging.info("Find %d CPTs after enumerating %d CPTs"%(cpt_count,count))

    
def enumerate_transitions8(size,card,order,threshold,step_size):
    # enumerate CPT space with dists of certain determinism and choose cpts with mutual info above the threshold
    cards = (card,)*order
    uniform_dist = np.ones(card)*(1.0/card)
    max_entropy = __entropy(uniform_dist)
    ENTROPY_RATIO = 0.5
    dist_filter = lambda dist: __entropy(dist) <= ENTROPY_RATIO*max_entropy
    transition_cpts = enumerate_cpts_shuffled(card,cards,step_size=step_size,is_deterministic=False,dist_fun=dist_filter) 
    count, cpt_count = 0, 0
    for transition_cpt in transition_cpts:
        count += 1
        mutual_info = get_cond_mutual_info_for_markov_chain(size,card,order,transition_cpt)
        if mutual_info >= threshold:
            cpt_count += 1
            yield transition_cpt,mutual_info
            # choose cpts with mutual info exceeds the threshold
        if count % 1000 == 0:
            logging.info("Find %d CPTs after enumerating %d CPTs"%(cpt_count,count))


def enumerate_transitions9(size,card,order,threshold,step_size):
    # enumerate CPT space for chain missing one state and choose cpts with mutual info above the threshold
    uniform_dist = np.ones(card)*(1.0/card)
    max_entropy = __entropy(uniform_dist)
    ENTROPY_RATIO = 0.25
    dist_filter = lambda dist: __entropy(dist) <= ENTROPY_RATIO*max_entropy
    transition_cpts = enumerate_cpts_shuffled(card,[card],step_size=step_size,is_deterministic=False,dist_fun=dist_filter) 
    count, cpt_count = 0, 0
    for transition_cpt in transition_cpts:
        count += 1
        mutual_info,group = get_cond_mutual_info_for_missing_one_card_chain(size,card,transition_cpt)
        if mutual_info >= threshold:
            cpt_count += 1
            yield transition_cpt,mutual_info,group
            # choose cpts with mutual info exceeds the threshold
        if count % 1000 == 0:
            logging.info("Find %d CPTs after enumerating %d CPTs"%(cpt_count,count))

def test():
    cpts = enumerate_cpts_shuffled(2,[3,3],0.05,False)
    for cpt in cpts:
        print(cpt)




















