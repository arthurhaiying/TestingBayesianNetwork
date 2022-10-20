import time
import math
import numpy as np
import statistics as s

import tac
import verify.AC
import verify.FJT as FJT
import train.data as data
import compile.decouple as decouple
import utils.utils as u
import utils.paths as paths
import utils.rbn as rbn
import utils.VE as VE
  
"""
Tests for verifying correctness of 

 -- tac compilation algorithm: test_functional() and test_suite()
 -- jointree algorithm that exploits functional cpts: test_functional_mockup()
 
Tests for measuring reduction in treewidth due to functional cpts: test_tw_reduction()

Tests for comparing evaluation times of various AC implementations: test_eval_time()

All tests are on random networks, except for test_suite()
"""

# collection of tests
def run_tests():
    # verify tac compilations on random networks with functional cpts
    # this does compile ACs
    test_functional(ssize=25,esize=8,vcount=45,scount=3,pcount=4)
    
    # verify tac compilations on a suite of networks
    test_suite(esize=25,hard_evidence=True)
    test_suite(esize=25,hard_evidence=False)
    
    # verify algorithm for shrinking separators on random networks and random jointrees
    # this does not compile ACs, just runs a direct implementation
#    test_functional_mockup()
    
    
# collection of evaluations
def run_evals():
    # measures reduction in tw due to shrinking separators
    test_tw_reduction(ssize=10)
    
    # measures evaluation time using different representations of ACs
    # writes log files to logs/exp
    ssize = 25
    for bsize in (16,32,64,128):
        for min_ac, max_ac in ((30,50),):
            for scount in (3,5):
                test_eval_time(ssize,bsize,min_ac,max_ac,vc=100,sc=scount,pc=5)

""" verifying correctness of tac compilations on random functional networks (against VE) """

# ssize : how many networks to try (in pairs)
# esize : evidence size (how many input vectors to evaluate on)
# vcount: number of variables in bn
# scount: max number of values per node
# pcount: max number of parents per node

def test_functional(ssize,esize,vcount,scount,pcount):
    saved = u.verbose
    u.verbose = False
    print('\n===Testing random networks with functional cpts')
    print('   (tac compilations against VE)')
    print(f'  networks {ssize}, batch {esize}, vars {vcount}, values <= {scount}, parents <= {pcount}')
    back = vcount - 1
    for testing in (False,True):
        for hard_evidence in (True,False):
            for fcount in (vcount,vcount//2,0): # number of vars with functional CPTs
                print(f'testing {testing}, hard_evidence {hard_evidence}, fcount {fcount}')
                for _ in range(ssize):
                    result = rbn.get(vcount,scount,pcount,fcount,back,testing)
                    bn, inputs, outputs, tbn = result if testing else (*result,None)
                    # inputs/outputs are roots/leafs of bn
                    # possible for a node to be both root and leaf (no parents or children)
                    for o in outputs:
                        if o not in inputs:  __VE_vs_TAC(bn,inputs,o,esize,hard_evidence,tbn)
                    for i in inputs:
                        if i not in outputs: __VE_vs_TAC(bn,outputs,i,esize,hard_evidence,tbn)
                print('\n')
    print('\n===All good!\n')
    u.verbose = saved


""" measure reduction in treewidth and rank on random networks with functional cpts """

# ssize: number of networks to try
def test_tw_reduction(ssize):

    saved = u.verbose
    u.verbose = False
    
    # vcount: number of network vars
    # scount: max number of values per var
    # pcount: max number of parents per var
    
    counts = (75,2,4), (100,3,5) # vcount, scount, pcount
    fperct = (1/4,1/2,2/3,4/5)   # percentage of functional vars
    
    fname = f'TW{ssize}_C{counts}_P{fperct}'
    fname = paths.exp / u.time_stamp(fname,'txt')
    f     = open(fname,'w+')
    
    u.echo(f,'\n===Reduction in TreeWidth')
    u.echo(f,f'sample size {ssize}')
    u.echo(f,'output logged into logs/exp/')
    
    start_time = time.perf_counter()
    
    for vcount, scount, pcount in counts:
        back = vcount - 1
        for functional_fraction in fperct: 
            fcount = int(vcount * functional_fraction)
            w1_sample = []
            w2_sample = []
            for _ in range(ssize):
                bn, _, _ = rbn.get(vcount,scount,pcount,fcount,back,testing=False)
                bn1      = bn.copy_for_inference()
                bn2, _, (w1,w2) = decouple.get(bn1,[],False,'minfill',None)
                w1_sample.append(w1)
                w2_sample.append(w2)
                
            reduction = [w1-w2 for w1,w2 in zip(w1_sample,w2_sample)]

            rd_mean, rd_stdev = s.mean(reduction), s.stdev(reduction)
            w1_mean, w1_stdev = s.mean(w1_sample), s.stdev(w1_sample)
            w2_mean, w2_stdev = s.mean(w2_sample), s.stdev(w2_sample)
            
            u.echo(f,f'\n== vcount {vcount}, scount {scount}, pcount {pcount}, fcount {functional_fraction:.2f}, ')
            u.echo(f,f'before mean {w1_mean:.1f} stdev {w1_stdev:.1f}')
            u.echo(f,f'after  mean {w2_mean:.1f} stdev {w2_stdev:.1f}')
            u.echo(f,f'reduce mean {rd_mean:.1f} stdev {rd_stdev:.1f}')
    
    all_time = time.perf_counter() - start_time
    u.echo(f,f'\n===Total Time: {all_time:.3f} sec') 
    f.close()
    v.verbose = saved
     

""" compare evaluation times on random networks using different implementation of AC """

# ssize: number of networks to try
# bsize: batch size
# min_ac: smallest AC size to try
# max_ac: largest  AC size to try
# vc    : number of network vars
# sc    : max number of values
# pc    : max number of parents

def test_eval_time(ssize,bsize,min_ac,max_ac,vc,sc,pc): 
    saved = u.verbose
    u.verbose = False
       
    fcount = vc // 2 # number of vars with functional cpt
    back   = vc - 1          
    
    fname = (f'RBN_S{ssize}_B{bsize}_'
             f'C{min_ac}_{max_ac}_BN_'
             f'{vc}_{sc}_{pc}_{fcount}_{back}')
    fname = paths.exp / u.time_stamp(fname,'txt')
    f     = open(fname,'w+')
    u.echo(f,f'\n===Evaluation time for random bayesian networks===\n')
    u.echo(f,f'{vc} vars, {sc} values, {pc} parents, '
             f'{fcount} functional vars (no roots), {back} back'
             f'\n{ssize} circuits, '
             f'size {min_ac}-{max_ac}M'
             f'\n{bsize} examples')
    u.echo(f,'output logged into logs/exp/')
    start_time = time.perf_counter()
    
    # stats
    s_AC, r_AC, s_SAC            = [], [], []
    t_AC, t_numpy, t_tf, t_array = [], [], [], []
    b_AC, b_numpy, b_tf, b_array = [], [], [], []
    def process(result):
        s, r, s2, tac, tnumpy, ttf, tarray, bac, bnumpy, btf, barray = result
        
        s_AC.append(s)
        r_AC.append(r)
        s_SAC.append(s2)
        
        t_AC.append(tac)
        t_numpy.append(tnumpy)
        t_tf.append(ttf)
        t_array.append(tarray)
        
        b_AC.append(bac)
        b_numpy.append(bnumpy)
        b_tf.append(btf)
        b_array.append(barray)
    
    success = 0
    while success < ssize:
        bn, inputs, outputs = rbn.get(vc,sc,pc,fcount,back,testing=False)
        i = np.random.choice(inputs)
        o = np.random.choice(outputs)
        
        result = __posterior_time(f,bn,inputs,o,bsize,min_ac,max_ac,success) # causal
        if result:
            success += 1
            process(result)
        result = __posterior_time(f,bn,outputs,i,bsize,min_ac,max_ac,success) # evidential
        if result:
            success += 1
            process(result)
    assert len(s_AC) == ssize
            
    # summary stats
    # eval time for ac per one million nodes and one example
    ac_per_mill = [1000*t/bsize//(s/1000000) for t,s in zip(t_AC,s_AC)]
    # size of largest tensor in ac (2** max binary rank)
    ac_max_rank = r_AC
    # comparing tensor and scalar ac size
    sac_ac      = [s1/s2 for s1,s2 in zip(s_SAC,s_AC)]
    # comparing ac eval time with others
    numpy_ac    = [t1/t2 for t1,t2 in zip(t_numpy,t_AC)]
    tf_ac       = [t1/t2 for t1,t2 in zip(t_tf,t_AC)]
    array_ac    = [t1/t2 for t1,t2 in zip(t_array,t_AC)]
    
    u.echo(f,f'\n==\nsummary stats ({ssize} circuits, {bsize} examples, '
             f'size {min_ac}-{max_ac}M)')
    u.echo(f,f'  ac  size: mean {int(s.mean(s_AC)):,}, stdev {int(s.stdev(s_AC)):,}, '
             f'min {min(s_AC):,}, max {max(s_AC):,}')
    u.echo(f,f'  ac brank: mean {s.mean(ac_max_rank):.1f}, stdev {s.stdev(ac_max_rank):.1f}')
    u.echo(f,f'  sac/ac size: mean {s.mean(sac_ac):.2f}, stdev {s.stdev(sac_ac):.2f}')
    
    # used batch size may be different from evidence size due to memory limitations
    u.echo(f,f'\nused batch size')
    u.echo(f,f'  ac   : mean {s.mean(b_AC):.1f}, stddev {s.stdev(b_AC):.1f}')
    u.echo(f,f'  numpy: mean {s.mean(b_numpy):.1f}, stddev {s.stdev(b_numpy):.1f}')
    u.echo(f,f'  tf   : mean {s.mean(b_tf):.1f}, stddev {s.stdev(b_tf):.1f}')
    u.echo(f,f'  array: mean {s.mean(b_array):.1f}, stddev {s.stdev(b_array):.1f}')
    
    u.echo(f,f'\neval time')
    u.echo(f,f'  ac / 1M : mean {s.mean(ac_per_mill):,}, stdev {s.stdev(ac_per_mill):.1f}')
    u.echo(f,f'  numpy/ac: mean {s.mean(numpy_ac):.1f}, stdev {s.stdev(numpy_ac):.1f}')
    u.echo(f,f'  tf/ac   : mean {s.mean(tf_ac):.1f}, stdev {s.stdev(tf_ac):.1f}')
    u.echo(f,f'  array/ac: mean {s.mean(array_ac):.1f}, stdev {s.stdev(array_ac):.1f}')
        
    all_time = time.perf_counter() - start_time
    u.echo(f,f'\n===Total Time: {all_time:.3f} sec (includes skipped circuits)') 
    f.close()
    u.verbose = saved
        
def __posterior_time(f,bn,inputs,output,bsize,min_ac,max_ac,counter):
    s_time = time.perf_counter()
    AC = tac.TAC(bn,inputs,output)
    t = time.perf_counter()-s_time
    
    if AC.size < min_ac*1000000 or AC.size > max_ac*1000000: 
        return None
    
    u.echo(f,f'\n== {counter} ==\nTensor AC:',end='')
    u.echo(f,f' {t:.1f} sec')
    u.echo(f,f'  size {AC.size:,}, max binary rank {AC.binary_rank:0.1f}')
    
    # get evidence
    cards          = tuple(bn.node(input).card for input in inputs)
    evidence       = data.evd_random(bsize,cards)
    
    # evaluate AC as tf graph with batch
    u.echo(f,f'(tf full) eval:',end='',flush=True)
    tac_posteriors, t_AC, b_AC = AC.evaluate(evidence,report_time=True)    
    u.echo(f,f' {t_AC:.2f} sec'
             f'\n  {1000*t_AC/bsize:.0f} ms per example, used batch size {b_AC}'
             f'\n  {1000*t_AC/bsize/(AC.size/1000000):.0f} ms per 1M nodes (one example)')
   
    # check classical AC and numpy
    AC_size  = AC.size
    AC_brank = AC.binary_rank
    opsgrapy = AC.ops_graph
    del AC # no longer needed

    u.echo(f,'\nScalar AC:',end='')
    s_time = time.perf_counter()
    SAC = verify.AC.ScalarAC(opsgrapy)
    t = time.perf_counter()-s_time
    u.echo(f,f' {t:.1f} sec')
    u.echo(f,f'  size {SAC.size:,}')
    u.echo(f,f'  {SAC.size/AC_size:.2f} scalar ac/tensor ac')

    def v(eval_func,type):
        u.echo(f,f'({type}) eval:',end='',flush=True)
        t_SAC, b_SAC = eval_func(evidence,tac_posteriors)
        u.echo(f,f' {t_SAC:.2f} sec'
                 f'\n  {1000*t_SAC/bsize:.0f} ms per example, used batch size {b_SAC}'
                 f'\n  {t_SAC/t_AC:.2f} {type}/ac ')
        return t_SAC, b_SAC

    t_numpy, b_numpy = v(SAC.verify_numpy,'numpy batch')
#    t_tf, b_tf    = v(SAC.verify_tf,'tf batch')
    t_tf, b_tf    = 0, 0
#    t_array, b_array = v(SAC.verify_array,'array')
    t_array, b_array = 0, 0
    
    return (AC_size, AC_brank, SAC.size, t_AC, t_numpy, t_tf, t_array, b_AC, b_numpy, b_tf, b_array)
    
        
"""
Verifies the jointree algorithm that exploits functional dependencies by replicating
functional cpts and shrinking separators. The networks are random and the jointrees
are also random. Verification is done against a crude implementation of VE. 

The jointree algorithm being testing is a direct (mockup) implementation of the
one that constructs tensor graphs (this way we remove tensorflow from the testing
loop). To test the tensorflow implementation, use test_functional_mockup())
"""

def test_functional_mockup():
    saved = u.verbose
    u.verbose = False
    print('\n===Testing random networks with functional cpts')
    print('   (jointree algorithm against crude VE)')
    for vcount in range(4,15): # number of variables
        for q in (2,3):        # controls number of functional var
            for d in (2,3,4):  # max number of duplicates for functional cpts
                for seed in (0,None):
                    for maxp in (3,4,5): # max number of parents
                        fcount = max(vcount//q,2)
                        maxp   = min(maxp,vcount-1)
                        net    = FJT.Net(vcount=vcount,fcount=fcount,max_pcount=maxp)
                        net.verify(max_duplicates=d,seed=seed)
    print('\n===All good!\n')
    u.verbose = saved


"""
Verifies correctness of tac compilation and evaluation on a selected set of networks.
The verification is done against a crude implementation of the VE algorithm.
"""    
    
# esize: size of evidence (number of input vectors)
def test_suite(esize,hard_evidence):
    import examples.networks.model as m1
    import examples.rectangles.model as m2
    import examples.digits.model as m3
    
    saved = u.verbose
    u.verbose = False
    
    print(f'\n===Verifying TAC compilations against VE: {esize} random evidence')
    """
    print('digits bns...',end='',flush=True)
    bn, inputs, output = m3.get(size=7,digits=range(10),testing=False,use_bk=True,tie_parameters=False)
    tbn, inputs, output = m3.get(size=7,digits=range(10),testing=True,use_bk=True,tie_parameters=False)
    __VE_vs_TAC(bn,inputs,output,esize,hard_evidence=hard_evidence) # checking bn
    __VE_vs_TAC(bn,inputs,output,esize,hard_evidence=hard_evidence,tbn=tbn) # checking tbn
    print('done.')
    """
    print('small bns...',end='')
    __VE_vs_TAC(m1.bn0(),('a',),   'b',esize,hard_evidence=hard_evidence)
    __VE_vs_TAC(m1.bn1(),('a','e'),'d',esize,hard_evidence=hard_evidence)
    __VE_vs_TAC(m1.bn2(),('a','b'),'c',esize,hard_evidence=hard_evidence)
    __VE_vs_TAC(m1.bn3(),('a','b'),'c',esize,hard_evidence=hard_evidence)
    __VE_vs_TAC(m1.bn4(),('a','c'),'b',esize,hard_evidence=hard_evidence) 
    print('done.')
    
    print('rectangle bns...',end='',flush=True)
    for size in (5,6,7,8):
        for output in ('label','width','col'):
            for use_bk in (True,False):
                # VE always uses bn, TAC uses bn or tbn
                bn, inputs  = m2.get(size,output,testing=False,use_bk=use_bk,tie_parameters=False)
                tbn, inputs = m2.get(size,output,testing=True,use_bk=use_bk,tie_parameters=False)
                __VE_vs_TAC(bn,inputs,output,esize,hard_evidence=hard_evidence) # checking bn
                __VE_vs_TAC(bn,inputs,output,esize,hard_evidence=hard_evidence,tbn=tbn) # checking tbn
    print('done.')
    
    u.verbose = saved
    print('===All good!\n')
    

# verifies tac against VE by comparing computed marginals on random soft evidence
# inputs are names of tbn node, output is name of tbn node
# if tbn is specified, it should be equivalent to bn
# VE always uses bn, TAC uses tbn if supplied, otherwise bn

def __VE_vs_TAC(bn,inputs,output,evidence_size,hard_evidence=False,tbn=None):
    if tbn==None: tbn = bn
    assert not bn._for_inference
    assert not tbn._for_inference
    
    hinputs  = inputs if hard_evidence else []
    cards    = tuple(bn.node(input).card for input in inputs)
    evidence = data.evd_random(evidence_size,cards,hard_evidence)
    TAC      = tac.TAC(tbn,inputs,output,hard_inputs=hinputs,trainable=False)

    tac_posteriors = TAC.evaluate(evidence,batch_size=16) # 16 to reduce memory  
    del TAC # no longer needed (save memory)
    ve_posteriors  = VE.posteriors(bn,inputs,output,evidence)
        
    close = np.isclose(ve_posteriors,tac_posteriors) # array
    ok    = np.all(close) # boolean
    if not ok:      
        mismatches = np.logical_not(close)
        print(f'\nMismatch between VE and TAC on {tbn.name}:')
        print('  VE \n  ',ve_posteriors[mismatches])
        print('  TAC\n  ',tac_posteriors[mismatches])
        print('\n  VE \n  ',ve_posteriors)
        print('  TAC\n  ',tac_posteriors)
        print('\n***Ouch!!!!\n')
        quit()
    else:
        print('.',end='',flush=True)