import numpy as np
import tensorflow as tf

precision  = None
float      = None
float_size = None
eps        = None

def set_single_precision():
    global precision, float, float_size, eps
    
    precision  = 'single'
    float      = tf.float32 # for all tensors
    float_size = 4          # for estimating tac memory    
    info       = np.finfo('float32')
    eps        = info.eps   # smallest representable positive number st 1 + eps != 1
    
def set_double_precision():
    global precision, float, float_size, eps
    
    precision  = 'double'
    float      = tf.float64 # for all tensors
    float_size = 8          # for estimating tac memory    
    info       = np.finfo('float64')
    eps        = info.eps   # smallest representable positive number st 1 + eps != 1
    

# double precision can be set on the command line (-d option)
set_single_precision()