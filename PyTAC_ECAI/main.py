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

# to find which blas/atlas is installed for python
# np.__config__.show()

tf.config.threading.set_intra_op_parallelism_threads(0)

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# -d: use double precision to tensors
# -f: force profile (fast way to save tacs and profile their executation)
# -h: help
# -s: silent (suppresses printing statistics)

def main(argv):
   
    try: opts, args = getopt.getopt(argv,'dfhs',[])
    except getopt.GetoptError:
        print('usage: main.py -d -f -h -s')
        sys.exit(2)
    
    for opt, arg in opts:
        if   opt == '-f':
            tacgraph.force_profile = True
        elif opt == '-d':
            p.set_double_precision()
        elif opt == '-s':
            u.set_silent()
        else: # covers -h
            print('usage: main.py -d -f -h -s')
            sys.exit()

if __name__ == '__main__':
    main(sys.argv[1:])

    ram = u.system_RAM_GB()

    u.show('\nPyTac Version 1.2.2, 2020 © Adnan Darwiche')
    u.show(f'RAM {ram:.1f} GB')
    u.show(f'TF {tf.__version__}, {p.precision} precision')

    #play.play_Digits_Sizes(use_bk=False,tie_parameters=False)
    play.play_Recs_250(use_bk=False, tie_parameters=False)
    #play.play_Digits_Noise(use_bk = False, tie_parameters = True)
    #play.play_Digits_Sizes(use_bk = False, tie_parameters = True)
    #play.play_Digits_21_12(use_bk = False, tie_parameters = True)
