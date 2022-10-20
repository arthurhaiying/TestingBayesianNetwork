import numpy as np
from time import perf_counter
from inspect import stack
from datetime import datetime
from psutil import virtual_memory

"""
Various utilities
"""

verbose = True # can be set by command line

# verbose and silent modes
# code uses show() instead of print() which adheres to verbose/silent modes
def set_verbose():
    global verbose
    verbose = True
    
def set_silent():
    global verbose
    verbose = False
    
# print only if verbose is True
def show(*args,**kwargs):
    if verbose: print(*args,**kwargs)

# pause execution until user confirms continuation
def pause():
    input('continue?')
    
# prints an error message if function input fails a test
# identifies the calling function and quits
def input_check(assertion,message):
    if not assertion:
        frame        = stack()[2] # skipping frame for 'input_check'
        fname        = frame[1]
        line_no      = frame[2]
        code_context = frame[4]
        curent_line  = code_context[frame[5]]
        current_line = curent_line.strip()     # white space at start/end
        current_line = curent_line.strip('\n')
        print(f'\nINPUT ERROR:\n  File \'{fname}\', line {line_no}')
        print(f'  {current_line}')
        print(f'  {message}')
        print( 'QUITTING PyTAC')
        quit()
        
# prints an error message and quits if assertion fails
def check(assertion,message,type):
    if not assertion:
        print(f'\nERROR in {type}:')
        print(f'  {message}')
        print( 'QUITTING PyTAC')
        quit()
        
# prints a warning message
def warning(cond,message):
    if verbose and cond: print(f'warning: {message}')
    
# returns system memory in GB
def system_RAM_GB():
    mem = virtual_memory()
    return mem.total/(1024**3)
    
# maps sequence to a list of element.attr or fn(element)
def map(attr_or_fn,seq):
    if type(attr_or_fn) is str:
        return [getattr(i,attr_or_fn) for i in seq]
    else:
        return [attr_or_fn(i) for i in seq]
    
# seq_of_seq could be a list of lists: flatten it
def flatten(seq_of_seq):
    flat = []
    for seq in seq_of_seq:
        flat.extend(seq)
    return flat
    
# joins elements of itr with s separators
# if attr is supplied, joins element.attr instead
def unpack(itr,attr=None,s=' '):
    if attr:
        return s.join(str(getattr(i,attr)) for i in itr)
    else:
        return s.join(str(i) for i in itr)
    
# checks that the sequence of numbers are unique and sorted from small to large    
def sorted(numbers):
    return all(numbers[i] < numbers[i+1] for i in range(len(numbers)-1))
    

# prints sets of vars using their ids (for debugging)
# emulates print() otherwise
def ppi(*args,**kwargs):
    def sl(vars):
        l = list(vars)
        l.sort()
        return unpack(l,'id')
    args = [(sl(a) if type(a) in (set,list,tuple) else a) for a in args]
    print(*args,**kwargs)
    
# prints sets of vars using their names (for debugging)
# emulates print() otherwise
def ppn(*args,**kwargs):
    def sl(vars):
        l = list(vars)
        l.sort(key=lambda v: v.name)
        return unpack(l,'name')
    args = [(sl(a) if type(a) in (set,list,tuple) else a) for a in args]
    print(*args,**kwargs)

# stamps a file name with current time    
def time_stamp(fname,extension):
    now = datetime.now().strftime('day_%d_%m_%Y_time_%I_%M_%S_%p')
    return f'{fname}_{now}.{extension}'
    
# used to log experimental data to file, also echos output to standard output
# this is used with -s command line option (silent)
# first member of args is a file
def echo(*args,**kwargs):
    file = args[0]
    args = args[1:]
    print(*args,**kwargs)
    file.write(*args)
    if not 'end' in kwargs: file.write('\n')
    
# applies a softmax to a list (np array) of weights (for debugging)
def normalize_weight(w):
    return np.exp(w)/sum(np.exp(w))

# validate that marginals are equal
def equal(marginals1,marginals2,tolerance=False):
    equal = np.isclose(marginals1,marginals2) if tolerance else np.equal(marginals1,marginals2)
    ok = np.all(equal) 
    if not ok:      
        print('\nMismatch between marginals1 and marginals2:\n')
        print('  marginals1\n  ',marginals1[np.logical_not(equal)])
        print('  marginals2\n  ',marginals2[np.logical_not(equal)])
    return ok
    
# validate the accuracy reported by the tac algorithm (for computing metrics)
def accuracy(reported_accuracy,one_hot_marginals,predictions):
    
    max_p   = np.max(predictions,axis=-1,keepdims=True)
    equal   = np.equal(one_hot_marginals,predictions==max_p)
    match   = np.product(equal,axis=-1)
    correct = np.count_nonzero(match)
    total   = len(match)
    
    computed_accuracy = correct/total
  
    if computed_accuracy != reported_accuracy:
        print('\nError: reported accuracy',reported_accuracy,'computed',computed_accuracy)