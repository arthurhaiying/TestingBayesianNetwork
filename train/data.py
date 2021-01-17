import numpy as np
from random import uniform, shuffle
from math import ceil, sqrt

"""
Provides functions for:
    # validating data
    # splitting data into training and validation
    # generating data batches
    # generating random subsets of data
    # generating evidence (grid and random)
    # generating data from a function
    # coverting between column-based and row-based evidence
    
TAC data is (evidence,marginals).

Evidence is input to a TAC and pertains to evidence nodes of tbn.
Marginals are output of a TAC and pertain to the query node of tbn.

Evidence is a list of 2D numpy arrays (one array for each evidence variable).
Marginals is a 2D numpy array.

More precisely, let:
--n: number of TAC inputs (number of evidence nodes)
--m: number of data records (number of lambdas per evidence node)
--lambda(i,j): evidence on node j in record i

 # Evidence is represented column-based: [col_1,...,col_n], where 
    -col_j is np array [lambda(1,j),...,lambda(m,j)].
    -col_j has shape (m,k), where k is cardinality of evidence node j.

 # Marginals are represented as: np array [marginal_1,...,marginal_m],
   which has shape (m,k) where k is the cardinality of query node.
"""
        
""" splitting data """
     
# shuffles data then splits it into training and validation
# size of validation data is percentage of size of data
def random_split(evidence,marginals,percentage):
    d_size              = len(marginals)           # data size
    v_size              = round(d_size*percentage) # validation data size
    t_size              = d_size - v_size          # training data size
    evidence, marginals = __shuffle(evidence,marginals)
    training_data       = (evd_slice(evidence,0,t_size),marginals[:t_size],t_size)
    validation_data     = (evd_slice(evidence,t_size),marginals[t_size:],v_size)
    assert d_size == t_size + v_size and t_size > 0 and v_size > 0
    return training_data, validation_data
    

""" generating data and evidence batches """

# returns a (batch generator, batch count) based on given batch size
# batch generator yields (evidence batch, marginals batch)
# last batch may be smaller than others

# shuffle data before returning batches
def random_data_batches(evidence,marginals,batch_size):
    evidence, marginals = __shuffle(evidence,marginals)
    return data_batches(evidence,marginals,batch_size)
    
# does not shuffle data before returning batches  
def data_batches(evidence,marginals,batch_size):
    data_size   = len(marginals)
    batch_count = ceil(data_size/batch_size)
    generator   = __batch_generator(evidence,marginals,data_size,batch_size)
    return generator, batch_count
    
def __batch_generator(evidence,marginals,data_size,batch_size):
    for start in range(0,data_size,batch_size):
        stop            = start + batch_size
        evidence_batch  = evd_slice(evidence,start,stop)
        marginals_batch = marginals[start:stop]
        yield evidence_batch, marginals_batch
        
# returns evidence batches based on given batch size
# last batch may be smaller
def evd_batches(evidence,batch_size):
    size = evd_size(evidence)
    for start in range(0,size,batch_size):
        stop      = start + batch_size
        evd_batch = evd_slice(evidence,start,stop)
        yield evd_batch
    

""" generating random subsets of data """

# returns a random subset of data
# size of subset is percentage of size of data
def random_subset(evidence,marginals,percentage):
    evidence, marginals = __shuffle(evidence,marginals)
    d_size              = len(marginals)           # data size
    s_size              = round(d_size*percentage) # subset size
    evidence_subset     = evd_slice(evidence,0,s_size)
    marginals_subset    = marginals[0:s_size]
    return evidence_subset, marginals_subset
        

# randomly shuffle evidence and marginals simultaneously
def __shuffle(evidence,marginals):
    data_size   = len(marginals)
    permutation = np.random.permutation(data_size)
    evidence    = [col[permutation] for col in evidence]
    marginals   = marginals[permutation]
    return evidence, marginals

""" marginals utilities """

# checks whether marginals are formatted correctly
def is_marginals(marginals,one_hot=False):
    if not ( 
        type(marginals) is np.ndarray and \
        len(marginals) > 0 and \
        len(marginals.shape) == 2):
        return False
    # must be normalized
    sum = np.sum(marginals,axis=-1)
    if not np.all(np.isclose(sum,1)):
        return False
    # check one-hot
    if one_hot:
        max = np.max(marginals,axis=-1)
        return np.all(np.isclose(max,1) | np.equal(max,0))
    return True
  
# checks whether marginals match var  
def mar_matches_output(marginals,var):
    try:
        return marginals.shape[1] == var.card
    except IndexError:
        print("marginals shape error: %s var card: %s" % (marginals.shape,var.card))

def mar_matches_output2(marginals,vars):
    vcards = tuple(var.card for var in vars)
    #print("cards: %s marginals shape: %s" % (vcards, marginals.shape,))
    try:
        return marginals.shape[1:] == vcards
    except IndexError:
        print("marginals shape error: %s vars cards: %s" % (marginals.shape,vcards))

# checks sanity of predicted marginals (output of tac evaluation)    
def mar_is_predictions(marginals):
    sum = np.sum(marginals,axis=-1)
    # must be normalized or all 0s (zero-probability evidence)  
    # allows tolerance
    return np.all(np.isclose(sum,1) | np.equal(sum,0))

# checks sanity of predicted marginals (output of tac evaluation)    
def mar_is_predictions2(marginals):
    sum = np.sum(marginals,axis=tuple(np.arange(1,marginals.ndim)))
    # must be normalized or all 0s (zero-probability evidence)  
    # allows tolerance
    return np.all(np.isclose(sum,1) | np.equal(sum,0))


# -expand marginals so it is over all values of var (if var lost values due to pruning)
def mar_expand(marginals,var):
    assert var._for_inference
    return __insert_pruned(marginals,var)
    
# project marginals on feasible values of var if var lost values due to pruning
def mar_project(marginals,var):
    assert var._for_inference
    return __remove_pruned(marginals,var)
    
# -col is np array with shape (batch,var.card)
# -insert zeros in axis=1 at the locations of pruned values of var
# -resulting col will have shape (batch,var.card_org)
def __insert_pruned(col,var):
    lost_values = (var.card < var._card_org)
    if lost_values:
        col     = np.copy(col)
        indices = set(var._values_idx) # indices of unpruned values
        for i in range(var.card_org):  # iterate over indices of original values
            if i not in indices:       # value at index i was pruned
                col = np.insert(col,0,i,axis=1)
    return col
    
# -col is np array with shape (batch,var.card_org)
# -remove entries of axis=1 which correspond to pruned values of var 
# -resulting col will have shape (batch,var.card)
def __remove_pruned(col,var):
    lost_values = (var.card < var._card_org)
    if lost_values:
        return np.take(col,var._values_idx,axis=1)
    return col

""" evidence utilities """

# col-based evidence: 
#   a list of 2D np arrays, each representing the lambdas of a variable
# row-based evidence: 
#  a list of lists, each representing a singleton batch (lambdas for all variables)
# default is col-based evidence 

# checks whether evidence is formatted correctly (col based)
# must have evidence on at least one variable
def is_evidence(evidence):
    if not (
        type(evidence) is list and \
        len(evidence) > 0 and \
        all(type(col) is np.ndarray and len(col.shape) == 2 for col in evidence)):
        return False
    size = evd_size(evidence)
    if not (size > 0 and all(col.shape[0] == size for col in evidence)):
        return False
    for col in evidence:
        # lambdas must be normalized
        sum = np.sum(col,axis=-1)
        if not np.all(np.isclose(sum,1)): 
            return False
    return True
        
# checks whether evidence is hard
# evidence is col-based
def evd_is_hard(evidence,vars,hard_vars):
    hard_vars = set(hard_vars)
    for col, var in zip(evidence,vars):
        if var not in hard_vars: continue
        # every lambda must sum to 1
        sum = np.sum(col,axis=-1)
        if not np.all(np.equal(sum,1)):
            return False
        # every lambda must have 1 as its max
        max = np.max(col,axis=-1)
        if not np.all(np.equal(max,1)):
            return False
    return True
    
# checks whether evidence matches evidence nodes
def evd_matches_input(evidence,vars):
    return all(col.shape[1] == var.card for col,var in zip(evidence,vars))
    
# -project evidence on feasible values of vars that lost values due to pruning
# -projection changes evidence at it removed lambda entries of pruned values
def evd_project(evidence,vars):
    assert all(var._for_inference for var in vars)
    lost_values = any(var.card < var._card_org for var in vars)
    if lost_values:
        return [__remove_pruned(col,var) for col,var in zip(evidence,vars)]
    return evidence
    
# returns a slice of col-evidence
def evd_slice(evidence,start,end=None):
    if end==None: return [e[start:]    for e in evidence]
    else:         return [e[start:end] for e in evidence]
    
# converts row-based evidence to col-based evidence
def evd_row2col(rows):
    batch_size = len(rows)
    assert batch_size > 0   # at least one batch
    var_count  = len(rows[0])
    assert var_count > 0   # at least one evidence variables
    cols = [[] for _ in range(var_count)]
    for row in rows:
        for i,lambda_ in enumerate(row):
            cols[i].append(lambda_)
    return [np.array(col) for col in cols]

# converts col-based evidence to row-based evidence
def evd_col2row(cols):
    var_count  = len(cols)
    assert var_count > 0  # at least one evidence var
    batch_size = len(cols[0])
    assert batch_size > 0 # at least one batch
    rows = [[] for _ in range(batch_size)]
    for col in cols:
        for i, lambda_ in enumerate(col):
            rows[i].append(lambda_)
    return rows

# convert index-based marginals to tac outputs
# index-based marginals: a list representing values of query
# tac ouptput: 2D numpy array
def marg_hard2lambdas(marginals,card):
    num_examples = np.array(marginals).size
    lambdas = np.zeros((num_examples,card))
    lambdas[np.arange(num_examples), np.array(marginals)] = 1.0
    return lambdas

# convert index-based evidence to tac inputs
# index-based evidences: a list of list, each representing one instantion for all evidence variables
# tac inputs: a list of 2D numpy arrays, each representing one evidence variable for all records
def evd_hard2lambdas(evidences,cards):
    assert np.array(evidences).shape[1] == len(cards)
    rows = []
    for evid in evidences:
        # for each record
        row = [] 
        for i,index in enumerate(evid):
            # for evidence node E_i
            card_i = cards[i]
            lambda_ = np.zeros(card_i)
            lambda_[index] = 1.0
            row.append(lambda_)
            # convert index to one-hot lambda 
        rows.append(row)
    cols = evd_row2col(rows)
    return cols
    
# returns the number of evidence variables
# evidence is col-based
def evd_var_count(evidence):
    return len(evidence)
    
# returns number of lambda vectors in evidence
# evidence is col-based
def evd_size(evidence):
    return len(evidence[0])    
        
    
""" generating evidence """

# returns grid evidence
def evd_grid(size): # size is for one dimension of grid
    grid_size = int(sqrt(size))
    assert size == grid_size**2 # size is a perfect square
    tics = [p/(grid_size-1) for p in range(grid_size)]
    E1 = []
    E2 = []
    for p1 in tics: 
        for p2 in tics:
            E1.append([p1,1-p1])
            E2.append([p2,1-p2])
    return [np.array(E1), np.array(E2)]
    
# returns random evidence over len(cards) variables whose cardinalities are cards
def evd_random(size,cards,hard_evidence=False): 
    if hard_evidence:
        return [ np.array([__hard_lambda(card) for _ in range(size)]) for card in cards]
    else:
        return [ np.array([__soft_lambda(card) for _ in range(size)]) for card in cards]         
    
# returns a random soft lambda (soft evidence on a tbn node)
def __soft_lambda(size):
    vector   = [uniform(0,1) for _ in range(size)]
    constant = sum(vector)
    return [r/constant for r in vector]
                
# returns a random hard lambda (hard evidence on a tbn node)
def __hard_lambda(size):
    vector = [0.]*(size-1)
    vector.append(1.)
    shuffle(vector)
    return vector
    
    
""" converting data to a function: extract first element of each lambda and marginal """

def data2fn(evidence,marginals):
    inputs  = [col[:,0] for col in evidence]
    outputs = marginals[:,0]
    return (inputs,outputs)

""" generating data from function(x,y) """

# assumes function has two inputs so uses grid evidence
def simulate_fn2(fn,size):
    evidence  = evd_grid(size)
    marginals = []
    for lambdas in evd_col2row(evidence):
        x = lambdas[0][0]
        y = lambdas[1][0]
        z = fn(x,y)
        marginals.append([z,1-z])
    marginals = np.array(marginals)
    return (evidence,marginals)