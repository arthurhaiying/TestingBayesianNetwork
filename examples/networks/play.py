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

def train_fn2(size,card):
    
    functions = [
#        lambda x,y: .7,
#        lambda x,y: x,
        lambda x,y: 0.5*math.exp(-5*(x-.5)**2-5*(y-.5)**2),
        lambda x,y: .5 + .5 * math.sin(2*math.pi*x),
        #lambda x,y: 1.0/(1+math.exp(-32*(y-.5))),
        #lambda x,y: math.exp(math.sin(math.pi*(x+y))-1),
        #lambda x,y:  (1-x)*(1-x)*(1-x)*y*y*y,
        #lambda x,y: math.sin(math.pi*(1-x)*(1-y)),
        #lambda x,y: math.sin((math.pi/2)*(2-x-y)),
        lambda x,y: .5*x*y*(x+y)]
    
    tbn, e1, e2, q = get.fn2_chain(size,card)
    tbn.set_select_type('sigmoid')
    TAC = tac.TAC(tbn,[e1,e2],q,trainable=True,profile=False)
    
    for fn in functions:
        evidence, marginals = data.simulate_fn2(fn,1024)
        visualize.plot3D(evidence,marginals,e1,e2,q)
        
        TAC.fit(evidence,marginals,loss_type='CE',metric_type='CE')
        predictions = TAC.evaluate(evidence)
        visualize.plot3D(evidence,predictions,e1,e2,q)
        

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