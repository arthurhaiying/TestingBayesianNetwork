import math
import random
import numpy as np

from tbn.tbn import TBN
from tbn.node import Node
import utils.utils as u

"""
Generates a tbn for rendering a rectangle:

  # Evidence nodes correspond to pixels in the image with values (True,False).
  # Query node is specified by the 'output' parameter.
"""

# image has size x size pixels
# pixel_0_0 is upper left pixel in image
# (row=0,col=0) is upper left corner of rendered rectangle

# pixels are always visited row, then column to have consistency
# between data and order of evidence variables in compiled tac

# use_bk: whether to integrate background knowledge (fix functional cpts and zeros)
# tie_parameters: whether to tie parameters of pixel nodes
def get(size,output,testing,use_bk,tie_parameters):
    assert output in ('label','height','width','row','col')

    net    = TBN(f'rectangle_{size}_{size}')
    irange = range(size)     # [0,..,size-1]
    srange = range(1,size+1) # [1,..,size]
    
    ### 1. row and column origins (only roots) are _unconstrained_
    
    # cpts: shape (size)
    uniform = lambda values: [1./len(values)]*len(values)
    
    # nodes
    orn = Node('row', values=irange, parents=[], cpt=uniform(irange))
    ocn = Node('col', values=irange, parents=[], cpt=uniform(irange))
    net.add(orn)
    net.add(ocn)
    
    ### 2. height and width are _constrained_ by row and column origins
    
    # cpts: shape (size, size)
    constraint = lambda p,n,size=size: p+n <= size
    
    # nodes
    h = Node('height', values=srange, parents=[orn], cpt=constraint, fixed_zeros=use_bk)
    w = Node('width',  values=srange, parents=[ocn], cpt=constraint, fixed_zeros=use_bk)
    net.add(h)
    net.add(w)
    
    ### 3. type is determined by height and width (except when height=weight)
    
    # cpt: shape (size,size,2)
    constraint = lambda h,w,t: h >= w if t == 'tall' else w >= h
    
    # nodes
    t = Node('label', values=('tall','wide'), parents=[h,w], cpt=constraint, fixed_cpt=use_bk)
    net.add(t)
    
    ### 4. row(i) is _determined_ by row origin and height: whether row i has an on-pixel
    ###    col(i) is _determined_ by col origin and width:  whether col i has an on-pixel
    
    row = {} # maps row to node
    col = {} # maps col to node
    
    for i in irange:
        fn     = lambda o,s,i=i: (o <= i and i < o+s)
        row[i] = Node(f'r_{i}', parents=[orn,h], cpt=fn, fixed_cpt=use_bk, functional=True)
        col[i] = Node(f'c_{i}', parents=[ocn,w], cpt=fn, fixed_cpt=use_bk, functional=True)
        net.add(row[i])
        net.add(col[i])
    
    ### 5. pixels are _determined_ by row and column
    
    # cpt for pixel (i,j): shape (2,2,2)
    function = lambda r,c: r and c
    
    # nodes
    inputs = []
    pname  = lambda i, j: f'pixel_{i}_{j}' 
        
    # evidence nodes must be generated row, then column to match data generation
    tie = 'pixel' if tie_parameters else None
    for i in irange:
        r = row[i]
        for j in irange: 
            c   = col[j]
            n   = pname(i,j)
            tie = tie if not testing else None
            p   = Node(n, parents=[r,c], cpt=function, testing=testing, cpt_tie=tie)
            net.add(p)
            inputs.append(n)
 
    #net.dot(view=True)
    return (net, inputs)
