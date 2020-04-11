import numpy as np
from functools import reduce
from operator import or_

from tbn.tbn import TBN
from tbn.node import Node
import utils.utils as u

"""

Generates a tbn for rendering 7-segment digits:
(https://en.wikipedia.org/wiki/Seven-segment_display)

  # Evidence nodes correspond to pixels in the image with values (True,False).
  # Query node has name 'digit' with values a subset of (0,1,2,3,4,5,6,7,8,9).

"""
    
# digits: specifies the digits covered by the model
# use_bk: whether to integrate background knowledge (fix functional cpts and zeros)
# tie_parameters: whether to tie parameters of pixel nodes
# remove_common: whether to exclude segments that aappear in all covered digits
def get(size,digits,testing,use_bk,tie_parameters,remove_common=False):
    assert size >= 7
    assert len(digits) >= 2
    assert all(d in (0,1,2,3,4,5,6,7,8,9) for d in digits)
    assert u.sorted(digits)

    # height is multiple of 7: 3 horizontal segments, 4 horizontal spaces
    # width  is multiple of 4: 2 vertical segments,   2 vertical spaces
    h_inc = 7
    w_inc = 4 
    
    net = TBN('digits')
    
    ###
    ### master START
    ###
    
    # nodes:
    # d digit
    # r upper-row (of bounding rectangle)
    # c left-col  (of bounding rectangle)
    # h height    (of bounding rectangle)
    # w width     (of bounding rectangle)
    # t thickness (of lines)
        
    # values
    dvals  = digits                # for digits (root)
    rvals  = range(0,size-h_inc+1) # for upper-row of digit (root)
    cvals  = range(0,size-w_inc+1) # for left-column of digit (root)
    srange = range(1,size+1)       # height, width, thickness (will be pruned)
    
    # constraints and functions
    # w is _constrained_ by c (length of segments)
    uniform = lambda values: [1./len(values)]*len(values)
    wct     = lambda c, w, w_inc=w_inc, size=size: (w % w_inc) == 0 and w <= size-c 
    
    # nodes
    dn  = Node('d', values=dvals,  parents=[],   cpt=uniform(dvals))
    rn  = Node('r', values=rvals,  parents=[],   cpt=uniform(rvals))
    cn  = Node('c', values=cvals,  parents=[],   cpt=uniform(cvals))
    wn  = Node('w', values=srange, parents=[cn], cpt=wct, fixed_zeros=use_bk)
    
    for n in [dn,rn,cn,wn]: net.add(n)
        
    ###
    ### segments
    ###
    
    # seven segments: 0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G
    # 1,2,4,5 are vertical
    # 0,3,6   are horizontal
    # https://en.wikipedia.org/wiki/Seven-segment_display
    
    # each digit corresponds to some segments (activated segments)
    segments  = (0,1,2,3,4,5,6)
    vsegments = (1,2,4,5) # vertical segments
    hsegments = (0,3,6)   # horizontal segments
    # map digits to segments
    dsegments = {0:'012345', 1:'12',     2:'01346', 3:'01236',   4:'1256', 
                 5:'02356',  6:'023456', 7:'012',   8:'0123456', 9:'012356'}
    
    # segments needed for the given digits
    segments  = tuple(s for s in segments if any(str(s) in dsegments[d] for d in digits))
    if remove_common:
        common   = tuple(s for s in segments if all(str(s) in dsegments[d] for d in digits))
        segments = tuple(s for s in segments if s not in common)
        u.show('Removing common segments',common)
    vsegments = tuple(s for s in vsegments if s in segments)
    hsegments = tuple(s for s in hsegments if s in segments)
    
    # nodes:
    # a segment is a rectangle
    # a segment s has upper-row srn[s], left-column scn[s], height shn[s], width swn[s],
    # whether segment is activated san[s], whether it will render in row i, sirn[s,i], 
    # whether it will render in column i, sicn[s,i], and whether it will render in 
    # pixel i,j, spn[s,i,j]
    
    # values
    irange = range(size)     # for segment row and column
    srange = range(1,size+1) # for segment height and width
                       
    # san[s] is whether segment s is activated given digit (True,False)
    # san[s] is a _function_ of node dn
    
    san = {} # maps segment to node
    for s in segments:
        fn   = lambda d, s=s, dsegments=dsegments: str(s) in dsegments[d]
        node = Node(f's{s}', parents=[dn], cpt=fn, fixed_cpt=use_bk, functional=True)
        net.add(node)
        san[s] = node
    
    # srn[s] is upper-row   for segment s
    # scn[s] is left-column for segment s
    # srn[s] and scn[s] are _functions_ of nodes rn and cn of master
    
    srn    = {} # maps segment to node
    scn    = {} # maps segment to node
    shifts = [(0,0), (0,3), (3,3), (6,0), (3,0), (0,0), (3,0)]
    # rshift: shift rn (digit) down  to get srn (segment)
    # cshift: shift cn (digit) right to get scn (segment)
        
    fn   = lambda r: r + 3
    sr3n = Node(f'sr3', values=irange, parents=[rn], cpt=fn, fixed_cpt=use_bk, functional=True)
    fn   = lambda r: r + 6
    sr6n = Node(f'sr6', values=irange, parents=[rn], cpt=fn, fixed_cpt=use_bk, functional=True)
    fn   = lambda c: c + 3
    sc3n = Node(f'sc3', values=irange, parents=[cn], cpt=fn, fixed_cpt=use_bk, functional=True)
    for n in (sr3n,sr6n,sc3n): net.add(n)
    
    rsn = {0:rn, 3:sr3n, 6:sr6n}
    csn = {0:cn, 3:sc3n}
    
    for s in segments: 
        rshift, cshift = shifts[s]
        srn[s] = rsn[rshift]
        scn[s] = csn[cshift]

    # sirn[s,i] is whether segment s will render in row i (True,False)
    # sicn[s,i] is whether segment s will render in col i (True,False)
    # sirn[s,i] is a _function_ of srn[s] and wn
    # sicn[s,i] is a _function_of  scn[s] and wn
    
    sirn = {} # maps (segment,row) to node
    sicn = {} # maps (segment,col) to node
    
    for s in segments:
        for i in irange: 
            name = f'in_r{s}_{i}'
            if s in vsegments: # vertical segment
                pa = [srn[s],wn]
                fn = lambda r, h, i=i: r <= i and i < r+h
            else:
                pa = [srn[s]]
                fn = lambda r, i=i: r == i
            node = Node(name, parents=pa, cpt=fn, fixed_cpt=use_bk, functional=True)
            net.add(node)
            sirn[(s,i)] = node
 
            name = f'in_c{s}_{i}'
            if s not in vsegments: # horizontal
                pa = [scn[s],wn]
                fn = lambda c, w, i=i: c <= i and i < c+w
            else:
                pa = [scn[s]]
                fn = lambda c, i=i: c == i
            node = Node(name, parents=pa, cpt=fn, fixed_cpt=use_bk, functional=True)
            net.add(node)
            sicn[s,i] = node
            
    # spn[s,i,j] is whether segment s will render in pixel i,j (True,False)
    # spn[s,i,j] is a _function_ of san[s] and sirn[s,i] and sicn[s,j]
    
    spn = {} # maps (segment,row,col) to node
    fn  = lambda a, r, c: a and r and c
    
    for s in segments:
        for i in irange:
            for j in irange:
                name = f'p{s}_{i}_{j}'
                pa   = [san[s],sirn[(s,i)],sicn[(s,j)]]
                node = Node(name, parents=pa, cpt=fn, fixed_cpt=use_bk, functional=True)
                net.add(node)
                spn[(s,i,j)] = node

    ###
    ### master END
    ###
    
    # image pixels: whether pixel i,j will render in image (iff some segment renders)
    
    output = dn.name
    inputs = []
    fn     = lambda *inputs: any(inputs) # or-gate
    
    tie = 'pixel' if tie_parameters else None
    for i in irange:
        for j in irange:
            name = f'p_{i}_{j}'
            pa   = [spn[(s,i,j)] for s in segments]
            tie  = tie if not testing else None
            pn   = Node(name, parents=pa, cpt=fn, testing=testing, cpt_tie=tie)
            net.add(pn)
            inputs.append(name)
    
    return (net, inputs, output)