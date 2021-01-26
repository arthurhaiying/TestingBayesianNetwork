import math
import numpy as np

import tac
import train.data as data
import verify
import utils.visualize as visualize
import examples.networks.model as get


""" example bns and tbns: simulate data from bn/tbn then learn parameters back using AC/TAC """

def train_nets():
    __simulate_fit(get.bn1(),'a','e','d')
    __simulate_fit(get.tbn1(random=True),'a','e','d') 
    __simulate_fit(get.bn2(),'a','b','c')
    __simulate_fit(get.tbn2(random=True),'a','b','c')
    __simulate_fit(get.bn3(),'a','b','c')
    __simulate_fit(get.tbn3(random=True),'a','b','c')
    __simulate_fit(get.bn4(),'a','c','b') 
    __simulate_fit(get.chain(),'S','E','M')
    __simulate_fit(get.chain(testing=True),'S','E','M')
        

""" simulate data from a tbn then learn back the tbn parameters using TAC """

def __simulate_fit(tbn,e1,e2,q):
    size = 1024

    # simulate
    TAC = tac.TAC(tbn,(e1,e2),q,trainable=False)

    evidence, marginals = TAC.simulate(size,'grid')
    
    # visualize simulated data
    visualize.plot3D(evidence,marginals,e1,e2,q)
    
    # learn
    TAC = tac.TAC(tbn,(e1,e2),q,trainable=True)
    TAC.fit(evidence,marginals,loss_type='MSE',metric_type='MSE')
    predictions = TAC.evaluate(evidence)
    
    # visualize learned tac
    visualize.plot3D(evidence,predictions,e1,e2,q)


""" train chain networks to fit various functions of the form z = f(x,y) """

def fun1(x,y): return 0.5*math.exp(-5*(x-.5)**2-5*(y-.5)**2)
def fun2(x,y): return .5 + .5 * math.sin(2*math.pi*x)
def fun3(x,y): return 1.0/(1+math.exp(-32*(y-.5)))
def fun4(x,y): return math.exp(math.sin(math.pi*(x+y))-1)
def fun5(x,y): return (1-x)*(1-x)*(1-x)*y*y*y
def fun6(x,y): return math.sin(math.pi*(1-x)*(1-y))
def fun7(x,y): return math.sin((math.pi/2)*(2-x-y))
def fun8(x,y): return.5*x*y*(x+y)

functions = [fun1,fun2,fun3,fun4,fun5,fun6,fun7,fun8]

class train_fn2_fun_wrapper:
    def __init__(self,size,card):
        self.size = size
        self.card = card
    def __call__(self,fn):
        tac = train_fn2(self.size,self.card,fn)
        return True

def train_fn2(size,card,fn):
    print("Start training fn...")
    tbn, e1, e2, q = get.fn2_chain(size,card)
    TAC = tac.TAC(tbn,[e1,e2],q,trainable=True,sel_type="sigmoid",profile=False)
    
    evidence, marginals = data.simulate_fn2(fn,1024)
    #visualize.plot3D(evidence,marginals,e1,e2,q)
    
    TAC.fit(evidence,marginals,loss_type='CE',metric_type='CE')
    predictions = TAC.evaluate(evidence)
    #visualize.plot3D(evidence,predictions,e1,e2,q)
    print("Finish training fn")
    return TAC

from multiprocessing import Pool
def run_train_fn2(size,card):
    NUM_WORKERS = 5
    train_fn2_fun = train_fn2_fun_wrapper(size,card)
    with Pool(NUM_WORKERS) as p:
        tac_list = []
        for ok in p.imap(train_fn2_fun,functions):
            #tac_list.append(tac) # can we get learned tac?
            print("ok: %s" %ok)

    
        

""" train an AC/TAC for data generated from the kidney model (showing Simpson's paradox) """

def train_kidney():
    
    tbn = get.kidney_full()
    tbn.set_select_type('sigmoid')

    e1 = 'L'
    e2 = 'T'
    q  = 'S'
    size = 1024
    
    # simulate
    TAC = tac.TAC(tbn,(e1,e2),q,trainable=False)
    evidence, marginals = TAC.simulate(size,'grid')
    
    # visualize simulated data
    visualize.plot3D(evidence,marginals,e1,e2,q)
    
    # bn
    for tbn in (get.kidney_tbn(),get.kidney_bn()):    
        # learn
        TAC = tac.TAC(tbn,(e1,e2),q,trainable=True)
        TAC.fit(evidence,marginals,loss_type='CE',metric_type='CE')
        predictions = TAC.evaluate(evidence)
    
        # visualize learned tac
        visualize.plot3D(evidence,predictions,e1,e2,q)

