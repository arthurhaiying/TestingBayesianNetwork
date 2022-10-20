from functools import lru_cache

import utils.precision as p
import compile.opsgraph as og
import utils.utils as u

cache_size = 1024
"""
Var (variable) is an object that abstracts a tbn node: defined in opsgraph.py

A dimension (axis) is an ordered set of variables.
A simple dimension has a single variable.
A trivial dimension has no variables.

Dims (dimensions) is an object representing an ordered set of dimensions.
The dimensions of a Dims must be unique, except that the trivial dimension can be repeated.

A tensor is defined over a Dims object. That is, each axis of a tensor corresponds
to a set of variables (not one). Hence, tensors represent 'structured' factors
instead of classical factors. This allows a more efficient implementation of factor
operations using tensor operations.

Dims objects are immutable.

For multiply and multiply-project operations, the input/output factors will each 
have at most 4 dimensions (3 + batch). This is ensured by structuring each input
factor according to an appropriate Dims object.
"""

# for debugging
def ppd(dvars):
    vars2str1 = lambda vars: ' '.join([str(var.id) for var in vars])
    vars2str2 = lambda vars: ' '.join([str(var.name) for var in vars])
    print(' '.join([f'({vars2str2(vars)})' for vars in dvars]))
    
# dvars is a tuple of (1) variables (simple dimensions) or (2) variable tuples (dimensions)
# returns a Dims object corresponding to dvars
# this function ensures a canonical representation of Dims objects 
# (Dims() should not be called directly)
def get_dims(dvars):
    simple = all(type(var) is og.Var for var in dvars)
    if simple: dvars = tuple((var,) for var in dvars)
    # lookup or create
    key   = dvars
    cache = Dims.cache
    if key in cache:
        return cache[key]
    dims       = Dims(dvars)
    cache[key] = dims
    return dims
    

class Dims:
    
    cache = {} # maps dvars to dims objects (so we have a unique dims for each dvars)
    
    def __init__(self,dvars):
        assert type(dvars) is tuple
        
        vars           = []
        unique         = set()
        self.batch_var = None
        for vars_ in dvars: 
            # dimensions are tuples of non-overlapping vars 
            # a trivial dimension may occur multiple times in dvars
            assert type(vars_) is tuple
            for var in vars_: 
                assert type(var) is og.Var
                assert var not in unique
                unique.add(var)
                if var.is_batch:
                    assert self.batch_var == None # only one batch var
                    self.batch_var = var
            vars.extend(vars_)
        # batch var (if any) sits in its own first dimension
        assert self.batch_var == None or dvars[0] == (self.batch_var,)
        """ the following test should fail for scalar tensors (arises in testing BNs) """
        assert vars # at least one var
        
        self.vars  = tuple(vars) # a tuple of variables
        self.dvars = dvars       # a tuple of (tuples of variables)
        self.rank  = len(dvars)  # number of dimensions 
        self.flat  = all(len(vars)==1 for vars in dvars)
        
        # save ordered vars
        vars.sort()
        self.ordered_vars = tuple(vars)
                    
        # shape
        def card(vars):
            card = 1
            for var in vars: card *= var.card # var.card -1 if var is batch
            assert card == -1 or card > 0
            return card

        self.shape = tuple(card(vars) for vars in self.dvars)
        
        # memory estimate for tensor with this shape (in MB)
        no_batch = self.vars[1:] if self.batch_var else self.vars
        self.mem = round(card(no_batch)*p.float_size/(1024**2))
                
        # for efficient implementation of various dims operations
        self.vars_set  = set(self.vars)
        self.dvars_set = set(self.dvars) # one occurence of trivial dimensions
                
    def __str__(self):   
        vars2str = lambda vars: '(' + ' '.join([str(var) for var in vars]) + ')'
        return ', '.join([vars2str(vars) for vars in self.dvars])
            
            
    @staticmethod
    def print_cache_info():
        u.show('  restructure',Dims.restructure_into.cache_info())
        u.show('  re-multiply',Dims.restructure_for_multiply.cache_info())
        u.show('  re-mulpro  ',Dims.restructure_for_mulpro.cache_info())
        
        
    # whether self has no variables
    def trivial(self):
        return self.vars == tuple() # same as self.dvars == tuple()
        
    # whether self has trivial dimensions
    def has_trivial(self):
        return tuple() in self.dvars_set
        
    # whether vars is a dimension of self
    def has_dim(self,vars):
        return vars in self.dvars_set
    
    # whether self and dims have the same variables
    def same_vars(self,dims):
        return self.vars_set == dims.vars_set
        
    # whether self and dims have the same shape
    def same_shape(self,dims):
        return self.shape == dims.shape
        
    # whether self and dims have the same dimensions (may be ordered differently)
    def same_dims(self,dims):
        # checking rank is necessary due to multiple trivial dimensions
        return self.rank == dims.rank and self.dvars_set == dims.dvars_set
        
    # whether self and dims have the same variable order (stronger than same-vars)
    def same_order(self,dims):
        return self.vars == dims.vars
        
    # whether self is ordered
    def ordered(self):
        return self.vars == self.ordered_vars
        
    # returns an ordered version of self (assumed to be flat)
    def order(self):
        assert self.flat
        return get_dims(self.ordered_vars)
        
    # returns a flatted dims of self
    def flatten(self):
        return get_dims(self.vars)
    
    # returns a version of self with no trivial dimensions
    def squeeze(self):
        return get_dims(tuple(vars for vars in self.dvars if vars != tuple()))
        
    # whether the dimensions of self are a subset of the dimensions of dims
    def subset(self,dims):
        return self.dvars_set <= dims.dvars_set
        
    # returns indices of trivial dimensions
    def trivial_axes(self):
        return tuple(i for i,vars in enumerate(self.dvars) if vars==tuple())
        
    # returns a permutation of self's dimension that turns it into dims
    def transpose_axes(self,dims):
        assert self.same_dims(dims)
        # need to be careful as we may have multiple trivial dimensions
        trivial_axes = [i      for i,vars in enumerate(self.dvars) if vars == tuple()]
        index        = {vars:i for i,vars in enumerate(self.dvars) if vars != tuple()}
        axis         = lambda vars: index[vars] if vars != tuple() else trivial_axes.pop() 
        axes         = tuple(axis(vars) for vars in dims.dvars)
        assert set(axes) == set(range(self.rank))
        return axes
        
    # returns indices of self dimensions not in dims (used to project self on dims)
    def project_axes(self,dims):
        assert not dims.has_trivial() # otherwise, projection is ambiguous
        assert dims.subset(self)
        return tuple(i for i,vars in enumerate(self.dvars) if not dims.has_dim(vars))
        
    # returns a map from each variable of self to its successor in the variable order
    # None is mapped to first variable and last variable is mapped to None
    def succ_fn(self):
        succ = {}
        prev = None
        for var in self.vars:
            succ[prev] = var
            prev = var
        succ[prev] = None # last var of dims
        return succ
    
    # self and dims have the same variables
    # whether each dimension of self is a subsequence of the variable order of dims
    # for example, ((4 5) (2 3) (1)) is congruent with ((1 2 3) (4 5))
    # if self is congruent with dims, we can turn it into dims using a transpose+reshape
    def congruent_with(self,dims):
        assert self.same_vars(dims)
        succ = dims.succ_fn() # successor function for dims.vars
        def subseq(vars):     # whether vars is a subsequence in dims.vars
            return all(succ[vars[i]] == vars[i+1] for i in range(len(vars)-1))
        return all(subseq(vars) for vars in self.dvars)
        
        
    """
    self and dims have the same variables (but perhaps ordered differently)
    we need to turn self into dims using reshape and transpose operations 
     
    returns dims1, dims2 such that self can be reshaped into dims1, dims1
    can be transposed into dims2, and finally dims2 can be reshaped into dims 
    
    the key is to do the absolute minimum work in terms of the amount of 
    reshaping and transposition (the time of reshape and transpose depend 
    on the specific shape and perm used)
    
    this algorithm is based on the following observations:
    (a) we can make dims1 congruent with dims2 (see above) using one reshape 
         (flattening dims1 will do but is inefficient and not necessary)
    (b) if dims1 is congruent with dims2, we can make it have the same variable
        order as dims2 using one transpose
    (c) if dims1 has the same variable order as dims2, we can turn it into dims2
        using one reshape 
    """
    @lru_cache(maxsize=cache_size)
    def restructure_into(self,dims):
        assert self.same_vars(dims)

        if self.same_order(dims): return self, self # reshaping self into dims   will do
        if self.same_dims(dims):  return self, dims # transposing self into dims will do
    
        # segment each dimension of self into the largest sub-dimensions that
        # are subsequences of dims.vars (and get rid of trivial dimensions)
        # this will make self congruent with dims
        succ  = dims.succ_fn()    # capturing variable order of dims
        dvars = []                # list of sub-dimensions for self
        for vars in self.dvars:   # vars is a dimension of self
            if not vars: continue # skip trivial dimensions
            # segment dimension vars into sub-dimensions vars_
            prev  = vars[0]
            vars_ = [prev]
            for var in vars[1:]:
                if succ[prev] == var:
                    vars_.append(var)
                else: # discontinuity in order, must segment
                    dvars.append(tuple(vars_))
                    vars_ = [var]
                prev = var
            assert vars_
            dvars.append(tuple(vars_))
        dvars = tuple(dvars)
        # each dimension of dvars is now a subsequence of dims.vars (congruent with dims)
        dvars2 = tuple(vars for vars in dims.dvars if vars != tuple)
        assert dvars != dvars2
            
        # returns a permutation of dvars leading to the same variable order as dims.vars
        def permutation(dvars): 
            pos = {var:i for i,var in enumerate(dims.vars)}
            seq = [(pos[vars[0]],i) for i,vars in enumerate(dvars)]
            key = lambda pair: pair[0]
            seq.sort(key=key)
            return tuple(i for _,i in seq)
        
        #print('\n',self,' --> ',dims)
        
        # reshape self into dims1 so it becomes congruent with dims
        dims1  = get_dims(dvars)
        # permute dims1 into dims2 so we have the same variable order as dims
        perm   = permutation(dvars)
        dvars_ = tuple(dvars[i] for i in perm)
        dims2  = get_dims(dvars_)
        # we finally reshape dims2 into dims

        # sanity checks
        assert self.same_order(dims1)
        assert dims1.congruent_with(dims2)
        assert dims2.same_order(dims)     
        
        # reshape self into dims1, transpose dims1 into dims2, reshape dims2 into dims
        # some of these may be no-ops (that is, self == dims1, dims1==dims2 or dims2=dims)
        return dims1, dims2
        
    """
    vars2 is a subset of dims1 variables
    restructures dims1 into dims1_ and structures vars2 into a dims2 so that
    (1) dims1_ has the same variable order as dims1
    (2) the dimensions of dims2 are dimensions in dims1_
    (3) dims1_ has the smallest rank with properties (1) and (2)
    if dims1 has a batch variable, it will be first in dims1_ (and dims1)
    """
    @staticmethod
    def restructure_for_project(dims1,vars2):
        s2 = set(vars2)
        assert s2 <= dims1.vars_set
        
        dvars1 = []
        dvars2 = []
        for vars1 in dims1.dvars:
            s1 = set(vars1)
            if not s1 & s2:
                dvars1.append(vars1)
                continue
            # partition vars1 so each element is either in vars2 or disjoint from vars2
            # add elements of the partition to dvars1
            vars = [] # element of partition
            comm = None # whether vars are in vars2
            for var in vars1:
                c = var in s2
                if vars and c != comm: # discontinuity: vars is now element of partition
                    vars = tuple(vars)
                    dvars1.append(vars)
                    if comm: dvars2.append(vars) # element var belongs to vars2
                    vars = [var] # start a new element of partition
                    comm = c
                else: 
                    vars.append(var)
                    comm = c  
            vars = tuple(vars) # last element of parition
            dvars1.append(vars)
            if comm: dvars2.append(vars) # last element belongs to vars
            s2 -= s1
            
        dims1_ = get_dims(tuple(dvars1))
        dims2  = get_dims(tuple(dvars2))
   
        assert dims1.same_order(dims1_)
        assert dims2.vars_set == set(vars2)
        assert dims2.subset(dims1_)

        return dims1_, dims2 
        
    # utility function for structure_for_multiply() and structure_for_mulpro()
    # the batch variable sits in its own (first) dimension, so is handled specially
    # add batch or trivial dimensions to dvars1, dvars2 and dvars so they are aligned
    # we have four cases: 
    #   dims1, dims2 have batch: dvars1, dvars2, dvars all get batch
    #   dims1, dims2 have no batch: dvars1, dvars2, dvars do not get batch (no change)
    #   dims1, but not dims2, has batch: dvars1, dvars get batch, dvars2 gets trivial
    #   dims2, but not dims1, has batch: dvars2, dvars get batch, dvars1 gets trivial
    @staticmethod
    def insert_batch(dims1,dims2,dvars1,dvars2,dvars):
        batch1, batch2 = dims1.batch_var, dims2.batch_var
        batch_var      = dims1.batch_var or dims2.batch_var
        b, e           = (batch_var,), tuple()
        
        # b is batch dimension and e is trivial dimension
        if batch1:   dvars1.insert(0,b)
        elif batch2: dvars1.insert(0,e)
        
        if batch2:   dvars2.insert(0,b)
        elif batch1: dvars2.insert(0,e)
        
        if batch1 or batch2: dvars.insert(0,b)
        # no change to dvars1, dvars2 and dvars if batch1 and batch2 are None
        
    """
    dims1 and dims2 are for tensors that we wish to multiply
    
    restructures these dims so factor multiply can be performed using standard tensor
    multiplication (with broadcasting). returns also the dims of resulting tensor
    
    the code below orders the components to try to keep dimensions sorted if possible,
    which reduces the amount of tensor transpositions that we will need later
    """
    @staticmethod
    @lru_cache(maxsize=cache_size)
    def restructure_for_multiply(dims1,dims2):

        (x, y, c), e = Dims.decompose_for_multiply(dims1,dims2), tuple()
        # dims1 = (x c), dims2 = (y c) and the result of multiply is (x y c)
        # e is trivial dimension (to allow broadcasting)
        # c, x, y are mutually disjoint and any of them can be empty
        
        dvars_ = [x,y,c]
        key    = lambda vars: vars[0].id if vars else -1
        dvars_.sort(key=key)
        
        dvars1, dvars2, dvars = [], [], []
        for vars in dvars_:
            if vars==e: continue # skipping trivial components
            if vars==x:
                dvars1.append(x)
                dvars2.append(e) 
            elif vars==y:
                dvars1.append(e)
                dvars2.append(y)
            else:
                assert vars==c
                dvars1.append(c)
                dvars2.append(c)
            dvars.append(vars)
        
        # add batch/trivial dimension to dvars1, dvars2 and dvars if needed 
        # (when either dims1 or dims2 has batch)
        Dims.insert_batch(dims1,dims2,dvars1,dvars2,dvars)
            
        dvars1, dvars2, dvars = tuple(dvars1), tuple(dvars2), tuple(dvars)    
        # vars1, vars2 and vars have the same length
        # vars1 and vars2 are compatible: corresponding dimensions are either equal 
        # or one of them is trivial (this is the requirement for broadcasting)
                 
        # return dims object will have at most 3 dimensions (plus batch if any)
        # dims will not have trivial dimensions
        dims1_, dims2_, dims  = get_dims(dvars1), get_dims(dvars2), get_dims(dvars)
        assert not dims.has_trivial()
        assert dims1_.rank <= 4 and dims2_.rank <= 4 and dims.rank <= 4
        
        return ((dims1_,dims2_), dims)
    
    # utility function for structure_for_multiply()
    #
    # decomposes variables of dims1 and dims2, excluding the batch variable,
    # into components x,y,c such that
    #
    # x: variables in dims1 but not dims2
    # y: variables in dims2 but not dims1
    # c: variables in both dims1 and dims2
    #
    # returns the components as sorted tuples (some may be empty)
    @staticmethod
    def decompose_for_multiply(dims1,dims2):
        s1, s2 = set(dims1.vars_set), set(dims2.vars_set)
        s1.discard(dims1.batch_var)
        s2.discard(dims2.batch_var)
        x, y, c = list(s1-s2), list(s2-s1), list(s1&s2)
        for component in (x,y,c): component.sort()
        return tuple(x), tuple(y), tuple(c)
            
             
    """
    dims1 and dims2 are for tensors that we wish to multiply, then project on vars
    
    restructures these dims so multiply-project can be performed using matrix 
    multiplication matmul (without having to construct a tensor over vars1+vars2)
    returns also the dims of resulting tensor
    
    dims1, dims2 and vars correspond to the separators of a cluster so they satisfy:
    if a variable appears in one of them, it must appear in at least another
    hence, variables that are summed out by mulpro must be in both dims1 and dims2
    
    the code below orders the components to try to keep dimensions sorted if possible,
    which reduces the amount of explicit tensor transpositions that we will need later
    """
    @staticmethod
    @lru_cache(maxsize=cache_size)
    def restructure_for_mulpro(dims1,dims2,vars):
        assert u.sorted(u.map('id',vars))
        
        x, y, c, s = Dims.decompose_for_mulpro(dims1,dims2,vars)
        # dims1 has vard (c x s), dims2 has vars (c y s)
        # result of multiply-project will have vars (c x y)
        # s are variables that will be summed out (in dims1 and dims2 but not vars)
        # c are common to dims1, dims2 and vars
        # x are in dims1 but not in dims2 (must be im vars)
        # y are in dims2 but not in dims1 (must be in vars)
        # c, x, y, s are mutually disjoint; c, x, y can be empty
        # s cannot be empty otherwise dims1, dims2 and vars all have the
        # same variables and we would have used multiply instead of mulpro
        assert s

        # matmul requires c be first, s be last or before last (summed out)
        key    = lambda vars: vars[0].id if vars else -1
        less   = lambda a, b: key(a) < key(b)
        
        dvars1 = [x,s] if less(x,s) else [s,x]
        dvars2 = [y,s] if less(y,s) else [s,y]
        if less(x,y):
            dvars, invert = [x,y], False # matmul(dims1,dims2)
        else:
            dvars, invert = [y,x], True  # matmul(dims2,dims1)
        squeeze = not x and not y # result will two trivial dimensions
        
        if c: 
            for dvars_ in (dvars1,dvars2,dvars): dvars_.insert(0,c)
        
        # matrix multiplication (matmul) requires that the dimensions of dvars1
        # and dvars2 be ordered as follows: dvars1=(c,*,s) and dvars2=(c,s,+) 
        # to yield vars=(c,*,+). the current ordering of dvars1 and dvars2 only
        # ensures that c is first so it may violate this pattern, but we can
        # instruct matmul to transpose the last two dimensions on the fly if needed
        s_index1   = -1 if not invert else -2
        s_index2   = -2 if not invert else -1
        transpose1 = dvars1[s_index1] != s # transpose last two dimensions of dims1_
        transpose2 = dvars2[s_index2] != s # transpose last two dimensions of dims2_
        assert not transpose1 or not transpose2 # at most one tensor will be transposed
                       
        # add batch/trivial dimension to dvars1, dvars2 and dvars if needed 
        # (when either dims1 or dims2 has batch)
        Dims.insert_batch(dims1,dims2,dvars1,dvars2,dvars)
        
        dvars1, dvars2, dvars = tuple(dvars1), tuple(dvars2), tuple(dvars)
        
        # returned dims object will have at most 3 dimensions (plus batch if any)
        # dims may have up to two trivial dimensions (when x and y are empty)
        dims1_, dims2_, dims  = get_dims(dvars1), get_dims(dvars2), get_dims(dvars)
        assert dims1_.rank <= 4 and dims2_.rank <= 4 and dims.rank <= 4
        
        return (((dims1_,transpose1), (dims2_,transpose2)), dims, invert, squeeze)
        
        
    # utility function for structure_for_mulpro()
    #
    # vars(dims1), vars(dims2) and vars are three separators connected to the same 
    # jointree node, hence, they satisfy the following:
    #  --if a variable appears in one set, it must appear in at least another set
    #  --each set is a subset of the union of the other two
    #
    # we want to decompose vars(dims1) and vars(dims2), excluding the batch variable,
    # into components x,y,c,s such that
    #
    # x: variables in dims1 but not dims2 (x must be a subset of vars)
    # y: variables in dims2 but not dims1 (y must be a subset of vars)
    # c: variables in dims1, dims2 and vars
    # s: variables in dims1 and dims2 but not in vars
    #
    # returns the components as sorted tuples (some may be empty)
    @staticmethod
    def decompose_for_mulpro(dims1,dims2,vars):
        s1, s2 = set(dims1.vars_set), set(dims2.vars_set)
        s1.discard(dims1.batch_var) # s1 may not have a batch var
        s2.discard(dims2.batch_var) # s2 may not have a batch var
        s3 = set(var for var in vars if not var.is_batch)
        assert s1 <= (s2|s3) and s2 <= (s1|s3) and s3 <= (s1|s2) 
        x, y, c, s = list(s1-s2), list(s2-s1), list(s1 & s2 & s3), list((s1 & s2)-s3)
        for component in (x,y,c,s): component.sort()
        # if s is empty, then s1=s2=s3: we would have called multiply not mulpro
        assert s 
        return tuple(x), tuple(y), tuple(c), tuple(s)