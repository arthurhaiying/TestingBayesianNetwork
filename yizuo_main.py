import sys, getopt, os
import random
import logging
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from multiprocessing import Pool
from tqdm import tqdm
import time

import utils.precision as p
import yizuo_play as play
import tensors.tacgraph as tacgraph
import utils.utils as u

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# -e: run exclusion jobs
# -i: run inclusion jobs
# -d: use double precision for tensors
# -f: force profile (fast way to save tacs and profile their executation)
# -h: help
# -s: silent (suppresses printing statistics)

server = None

def main(argv):
    global server
    
    try: opts, args = getopt.getopt(argv,'defihs',[])
    except getopt.GetoptError:
        print('main.py -d -e -f -i -h -m <percentage> -s')
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print('usage: main.py -d -e -f -h -i -m <percent> -s')
            sys.exit()
        elif opt == '-f':
            tacgraph.force_profile = True
        elif opt == '-e':
            server = 'exclusion'
        elif opt == '-i':
            server = 'inclusion'
        elif opt == '-d':
            p.set_double_precision()
        elif opt == '-s':
            u.set_silent()


if __name__ == '__main__':
    main(sys.argv[1:])

    ram = u.system_RAM_GB()

    u.show('\nPyTac v1.2, 2019 © Adnan Darwiche')
    u.show(f'RAM {ram:.1f} GB')
    u.show(f'TF {tf.__version__}, {p.precision} precision')

   
    def call_exclusion(arg):
        job, use_bk, tie = arg
        assert 1 <= job <= 9
        assert not (6 <= job <= 9) or use_bk==True
        # rectangles 
        if   job==1:
            play.play_Recs_Sizes(use_bk,tie)
            print(f'\nDone: Recs_Sizes use_bk {use_bk} tie {tie}')
        elif job==2:
            play.play_Recs_Noise(use_bk,tie)
            print(f'\nDone: Recs_Noise use_bk {use_bk} tie {tie}')
        elif job==3:
            play.play_Recs_Per(use_bk,tie) # diff_noise2 is (1,2) (2,1)
            print(f'\nDone: Recs_Per use_bk {use_bk} tie {tie}')
        elif job==4:
            play.play_Recs_Diff(use_bk,tie,(1,2),(2,1))
            print(f'\nDone: Recs_Diff_12_21 use_bk {use_bk} tie {tie}')
        elif job==5:
            play.play_Recs_Diff(use_bk,tie,(2,1),(1,2))
            print(f'\nDone: Recs_Diff_21_12 use_bk {use_bk} tie {tie}')
        # digits 
        elif job==6:
            play.play_Digits_Sizes(use_bk,tie)
            print(f'\nDone: Digits_Sizes use_bk {use_bk} tie {tie}')
        elif job==7:
            play.play_Digits_Noise(use_bk,tie)
            print(f'\nDone: Digits_Noise use_bk {use_bk} tie {tie}')
        elif job==8:
            play.play_Digits_12_21(use_bk,tie)
            print(f'\nDone: Digits_12_21 use_bk {use_bk} tie {tie}')
        elif job==9:
            play.play_Digits_21_12(use_bk,tie)
            print(f'\nDone: Digits_21_12 use_bk {use_bk} tie {tie}')
        return f'e {arg}'
        
    def call_inclusion(arg):
        job, group, use_bk, tie = arg
        assert 1 <= job <= 4
        assert use_bk==False
        assert not (1 <= job <= 1) or group
        assert not (2 <= job <= 4) or not group
        # digits 
        if   job==1:
            play.play_Digits_Sizes(use_bk,tie,group)
            print(f'\nDone: Digits_Sizes use_bk {use_bk} tie {tie} group {group}')
        elif job==2:
            play.play_Digits_Noise(use_bk,tie,group)
            print(f'\nDone: Digits_Noise use_bk {use_bk} tie {tie} group {group}')
        elif job==3:
            play.play_Digits_12_21(use_bk,tie)
            print(f'\nDone: Digits_12_21 use_bk {use_bk} tie {tie}')
        elif job==4:
            play.play_Digits_21_12(use_bk,tie)
            print(f'\nDone: Digits_21_12 use_bk {use_bk} tie {tie}')
        return f'i {arg}'
    
    
    # spawn processes
        
    if server=='exclusion':
        call       = call_exclusion
        pool_size  = 20
        parameters = []
        for job in range(1,6):
            for use_bk in (True,False):
                for tie in (True,False):
                    parameters.append((job,use_bk,tie))
        for job in range(6,10):
            for tie in (True,False):
                parameters.append((job,True,tie))
#        parameters.sort(key=lambda p: (1-p[1],1-p[2])) # easier jobs first
        print(f'\n===\nexclusion: {len(parameters)} jobs, {pool_size} pool') 
    else:
        assert server=='inclusion'
        call       = call_inclusion
        pool_size  = 8
        parameters = []
        for job in range(1,2):
            for tie in (True,False):
                for group in ('a','b','c','d'):
                    parameters.append((job,group,False,tie))
        for job in range(2,5):
            for tie in (True,False):
                parameters.append((job,None,False,tie))  
#        parameters.sort(key=lambda p: (1-p[2],1-p[3])) # easier jobs first
        print(f'\n===\ninclusion: {len(parameters)} jobs, {pool_size} pool') 
        
    print('===')
    
    start_time = time.perf_counter()
    
    return_codes = [] 
    with Pool(processes=pool_size) as pool:
        for rval in tqdm(pool.imap_unordered(call,parameters), total=len(parameters)):
            return_codes.append(rval)
            
    total_time  = time.perf_counter() - start_time
        
    print('\nFinish Order:',return_codes)
    print(f'Total Time: {total_time/3600:.2f} hrs')
    print('===')
