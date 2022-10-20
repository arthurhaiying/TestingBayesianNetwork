from tbn.tbn import TBN
from tbn.node import Node

# chain tbn for learning functions with two inputs
# size is the number of testing layers between roots and output
# card is the cardinality of testing nodes, except the input/output ones
def fn2_chain(size,card=2):
    assert size >= 2 and card >= 2
    
    net     = TBN('fn2_chain')
    bvalues = ('v0','v1')
    values  = tuple('v%d' % i for i in range(card)) 
    
    x1 = Node('x',values=bvalues,parents=[])
    y1 = Node('y',values=bvalues,parents=[])
    z1 = Node('z1',values=values,parents=[x1,y1],testing=True)
    
    net.add(x1)
    net.add(y1)
    net.add(z1)
        
    for i in range(2,size+1):
        last   = (i==size)
        name   = 'z' if last else 'z%d' % i
        values = bvalues if last else values
        p_name = 'z%d' % (i-1)
        parent = net.node(p_name)
        z = Node(name,values=values,parents=[parent],testing=True)
        net.add(z)
        
    #net.dot(view=True)
    return (net,'x','y','z')
    

def chain(testing=False):
    
    n0 = Node('S', parents=[])
    n1 = Node('n1',parents=[n0],testing=testing)
    n2 = Node('n2',parents=[n1],testing=testing)
    n3 = Node('M', parents=[n2],testing=testing)
    n4 = Node('n4',parents=[n3],testing=testing)
    n5 = Node('n5',parents=[n4],testing=testing)
    n6 = Node('E' ,parents=[n5],testing=testing)
    
    net = TBN('chain')
    net.add(n0)
    net.add(n1)
    net.add(n2)
    net.add(n3)
    net.add(n4)
    net.add(n5)
    net.add(n6)
    
    return net
    
# bn0:  a -> b   
# bn1:  a -> b,c  b,c -> d  c -> e
# bn2:  a -> b -> c
# bn3:  a(3) -> b,c
# bn4:  a, b(3) -> c

def bn0():

    #CPTs
    aa = [0.6,0.4]
    bb = [[0.9,0.1],[0.2,0.8]]
    
    #nodes
    a = Node('a',cpt=aa)
    b = Node('b',parents=[a],cpt=bb)

    #TBN
    n = TBN('bn0')
    n.add(a)
    n.add(b)
        
    return n
    
def bn1():

    #CPTs
    aa = [0.6,0.4]
    bb = [[0.2,0.8],[0.75,0.25]]
    cc = [[0.9,0.1],[0.1,0.9]]
    dd = [[[0.95,0.05],[0.9,0.1]],[[0.8,0.2],[0.0,1.0]]]
    ee = [[0.7,0.3],[0.0,1.0]]
    
    #nodes
    a = Node('a',values=('b','g'),cpt=aa)
    b = Node('b',parents=[a],cpt=bb)
    c = Node('c',parents=[a],cpt=cc)
    d = Node('d',parents=[b,c],cpt=dd)
    e = Node('e',parents=[c],cpt=ee)

    #TBN
    n = TBN('bn1')
    n.add(a)
    n.add(b)
    n.add(c)
    n.add(d)
    n.add(e)
        
    return n
    
def tbn1(random=False):

    #CPTs
    aa = [0.6,0.4]
    bb = [[0.2,0.8],[0.75,0.25]]
    cc = [[0.9,0.1],[0.1,0.9]]
    dd = [[[0.95,0.05],[0.9,0.1]],[[0.8,0.2],[0.0,1.0]]]
    ee = [[0.7,0.3],[0.0,1.0]]
    
    if random: bb=dd=None
    
    #nodes
    a = Node('a',values=('b','g'),cpt=aa)
    b = Node('b',parents=[a], testing=True, cpt1=bb, cpt2=bb)
    c = Node('c',parents=[a], testing=True, cpt1=cc, cpt2=cc)
    d = Node('d',parents=[b,c], testing=True, cpt1=dd, cpt2=dd)
    e = Node('e',parents=[c], cpt=ee)

    #TBN
    n = TBN('tbn1')
    n.add(a)
    n.add(c) # intentionally c then b (selecting b can now depend on evidence e)
    n.add(b)
    n.add(d)
    n.add(e)
        
    return n
    
def bn2():

    #CPTs
    aa = [0.6,0.4]
    bb = [[0.9,0.1],[0.2,0.8]]
    cc = [[0.3,0.7],[0.5,0.5]]
    
    #nodes
    a = Node('a',cpt=aa)
    b = Node('b',parents=[a],cpt=bb)
    c = Node('c',parents=[b],cpt=cc)

    #TBN
    n = TBN('bn2')
    n.add(a)
    n.add(b)
    n.add(c)
        
    return n
    
def tbn2(random=False):

    #CPTs
    aa = [0.6,0.4]
    bb = [[0.9,0.1],[0.2,0.8]]
    cc = [[0.3,0.7],[0.5,0.5]]
    
    if random: bb=cc=None
        
    #nodes
    a = Node('a',cpt=aa)
    b = Node('b',parents=[a],testing=True, cpt1=bb, cpt2=bb)
    c = Node('c',parents=[b],testing=True, cpt1=cc, cpt2=cc)

    #TBN
    n = TBN('tbn2')
    n.add(a)
    n.add(b)
    n.add(c)
        
    return n
    
    
def bn3():

    #CPTs
    aa = [0.4,0.6]
    bb = [[0.9,0.1],[0.1,0.9]]
    cc = [[0.3,0.7],[0.8,0.2]]
    
    #nodes
    a = Node('a',values=('t','f'),cpt=aa)
    b = Node('b',parents=[a],cpt=bb)
    c = Node('c',parents=[a],cpt=cc)

    #TBN
    n = TBN('bn3')
    n.add(a)
    n.add(b)
    n.add(c)
        
    return n
    
    
def tbn3(random=False):

    #CPTs
    aa = [0.4,0.6]
    bb = [[0.9,0.1],[0.1,0.9]]
    cc = [[0.3,0.7],[0.8,0.2]]
    
    if random: cc=None
        
    #nodes
    a = Node('a',values=('t','f'),cpt=aa)
    b = Node('b',parents=[a],cpt=bb)
    c = Node('c',parents=[a],testing=True,cpt1=cc,cpt2=cc)

    #TBN
    n = TBN('tbn3')
    n.add(a)
    n.add(b)
    n.add(c)
        
    return n
    
    
def bn4():

    #CPTs
    aa = [0.2,0.8]
    bb = [0.7,0.1,0.2]
    cc = [[[0.3,0.7],[0.5,0.5]], \
          [[0.8,0.2],[0.4,0.6]], \
          [[0.1,0.9],[0.7,0.3]]]
    
    #nodes
    a = Node('a',cpt=aa)
    b = Node('b',values=('r','b','g'),cpt=bb)
    c = Node('c',parents=[b,a],cpt=cc)

    #TBN
    n = TBN('bn4')
    n.add(a)
    n.add(b)
    n.add(c)
        
    return n
    
# for simulation        
def kidney_full():

    #CPTs
    ll = [0.49,0.51]
    tt = [[0.77,.23],[0.24,0.76]]
    ss = [[[0.73,0.27],[0.69,0.31]], [[0.93,0.07],[0.87,0.13]]]
    
    #nodes
    l = Node('L',values=['y','n'],cpt=ll)
    t = Node('T',values=['A','B'],parents=[l],cpt=tt)
    s = Node('S',values=['y','n'],parents=[l,t],cpt=ss)

    #TBN
    n = TBN('kidney true model')
    n.add(l)
    n.add(t)
    n.add(s)
        
    return n

# for fitting
def kidney_bn():

    #nodes
    l = Node('L',values=['y','n'])
    t = Node('T',values=['A','B'])
    s = Node('S',values=['y','n'],parents=[l,t])

    #TBN
    n = TBN('kidney bn')
    n.add(l)
    n.add(t)
    n.add(s)
        
    return n
    
# for fitting
def kidney_tbn():
    
    #nodes
    l = Node('L',values=['y','n'])
    t = Node('T',values=['A','B'])
    s = Node('S',values=['y','n'],parents=[l,t],testing=True)

    #TBN
    n = TBN('kidney tbn')
    n.add(l)
    n.add(t)
    n.add(s)
        
    return n


# get cancer true model
def cancer_bn():
    # cpts
    cpt_a = [0.6, 0.4]
    cpt_b = [[0.2, 0.8], [0.75, 0.25]]
    cpt_c = [[0.8, 0.2], [0.1, 0.9]]
    cpt_d = [[[0.95, 0.05], [0.9, 0.1]], [[0.8, 0.2], [0.0, 1.0]]]
    cpt_e = [[0.7, 0.3], [0., 1.0]]

    # nodes
    bn = TBN("cancer")
    A = Node("A", parents=[], cpt=cpt_a)
    B = Node("B", parents=[A], cpt=cpt_b)
    C = Node("C", parents=[A], cpt=cpt_c)
    D = Node("D", parents=[B, C], cpt=cpt_d)
    E = Node("E", parents=[C], cpt=cpt_e)
    nodes = [A, B, C, D, E]
    for n in nodes:
        bn.add(n)
    return bn