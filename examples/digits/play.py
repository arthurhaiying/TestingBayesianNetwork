import numpy as np
import time
import statistics as s

import utils.precision as p
import utils.paths as paths
import tac
import train.data as data
import utils.visualize as visualize
import examples.digits.model as dmodel
import examples.digits.data as ddata
import verify.AC
import utils.utils as u

"""
Validating and training ACs/TACs for networks that recognize digits.
"""

# validates ac/tac for digits model by running on clean images (we know the labels)
def validate(size,digits,testing,elm_method='minfill',elm_wait=30):
    assert size >= 7
    assert all(d in range(10) for d in digits)

    # get data (ground truth)
    evidence, labels = ddata.get(size,digits)
    data_size = len(labels)

    circuit_type = 'TAC' if testing else 'AC'    
    u.show(f'\n===Checking {circuit_type} for digits {digits} in {size}x{size} images: {data_size} total')
    
    # get model
    net, inputs, output = dmodel.get(size,digits,testing,
                            use_bk=True,tie_parameters=False,remove_common=False)
    
    # compile model into circuit
    circuit = tac.TAC(net,inputs,output,trainable=False,profile=False,
                elm_method=elm_method,elm_wait=elm_wait)
    
    # evaluate circuit on evidence to get predictions
    predictions = circuit.evaluate(evidence)

    # verify that predictions match labels
    if u.equal(predictions,labels): 
        u.show('\n===All good!\n')
    else:
        u.show('***bumper!!!')
        quit()
        
    #evidence = data.evd_col2row(evidence)
    #for lambdas, prediction in zip(evidence,predictions):
    #    visualize.image_lambdas(lambdas,str(prediction),size)
  

# trains the ac/tac of digits model on noisy images
# testing: whether tac or ac
# use_bk: whether to incorporate background knowledge in tac/ac       
def train(size,digits,data_size,testing,use_bk,tie_parameters,remove_common=False):
    assert size >= 7
    assert all(d in range(10) for d in digits)
    
    circuit_type = 'TAC' if testing else 'AC'    
    u.show(f'\n===Training {circuit_type} for digits {digits} in {size}x{size} images, use_bk {use_bk}, tie {tie_parameters}')
    
    # get model
    net, inputs, output = dmodel.get(size,digits,testing,use_bk,tie_parameters,remove_common)
    
    # get data (ground truth)
    t_evidence, t_labels = ddata.get(size,digits,noisy_image_count=100,noise_count=size)
    v_evidence, v_labels = ddata.get(size,digits,noisy_image_count=200,noise_count=size)
    
    # compile model into circuit
    circuit = tac.TAC(net,inputs,output,trainable=True,profile=False)
    
    # get random subset of dats
    t_percentage = data_size / len(t_labels)
    v_percentage = max(1000,data_size)/len(v_labels) # no less than 1000
    t_evidence, t_labels = data.random_subset(t_evidence,t_labels,t_percentage)
    v_evidence, v_labels = data.random_subset(v_evidence,v_labels,v_percentage)
    
    # fit circuit
    circuit.fit(t_evidence,t_labels,loss_type='CE',metric_type='CA')

    # compute accuracy
    accuracy = circuit.metric(v_evidence,v_labels,metric_type='CA')
    u.show(f'\n{circuit_type} accuracy {100*accuracy:.2f}')
       
    # verify reported accuracy
    #predictions = circuit.evaluate(t_evidence)
    #u.accuracy(accuracy,t_labels,predictions)
       
"""
scripts for logging experiments
"""    

def eval_all(sizes,digits,testing):
    circuit_type = 'TAC' if testing else 'AC'
    fname = paths.exp / u.time_stamp(f'eval_digits_{testing}','txt')
    f     = open(fname,'w+')
    u.echo(f,f'\n===Digits: evaluating {circuit_type} for {digits}===')
    u.echo(f,'output logged into logs/exp/')
    start_time = time.time()
    for size in sizes: eval(f,size,digits,testing)
    all_time = time.time() - start_time
    u.echo(f,f'\nTotal Time: {all_time:.3f} sec') 
    f.close()
        
def eval(f,size,digits,testing):
    circuit_type = 'TAC' if testing else 'AC'
    # get data (ground truth)
    evidence, marginals = ddata.get(size,digits)
    ecount = len(marginals) # number of examples
        
    u.echo(f,f'\n==digits {size}x{size} images: {ecount} total')
    
    # get model
    net, inputs, output = dmodel.get(size,digits,testing,
                            use_bk=True,tie_parameters=False,remove_common=False)
    
    # compile model
    s_time = time.time()
    u.echo(f,f'\ncompiling {circuit_type}:',end='')
    AC = tac.TAC(net,inputs,output,trainable=False,profile=False)
    t = time.time()-s_time
    u.echo(f,f' {t:.1f} sec')
    u.echo(f,f'  {circuit_type} size {AC.size:,}\n  (sep) binary rank {AC.binary_rank:.1f}, rank {AC.rank}')
    
    # evaluate AC on evidence to get predictions
    u.echo(f,f'evaluating {circuit_type}:',end='',flush=True)
    predictions, t1, batch_size = AC.evaluate(evidence,report_time=True)
    u.echo(f,f'  batch_size {batch_size}')
    u.echo(f,f'  {t1:.2f} sec, {1000*t1/ecount:.1f} ms per example')
    
# trains digits model for different data sizes 
# runs a number of tries for each dataset size and reports best accuracy
# results are logged into a time-stamped and descriptively-named file in 'logs/exp/'
def train_all(size,digits,tries,data_sizes,testing,use_bk,tie_parameters,batch_size):
    start_time = time.time()
    
    fname = paths.exp / u.time_stamp(f'train_digit_{size}_{digits}_{tries}_{testing}_{use_bk}_{tie_parameters}','txt')
    f     = open(fname,'w+')
    
    u.echo(f,f'\ndigit {size} x {size}, digits {digits}, data_sizes {data_sizes}, testing {testing}, use_bk {use_bk}, tie {tie_parameters}\n')
    u.echo(f,f'fixed batch size {batch_size}')
    u.echo(f,'output logged into logs/exp/')
        
    def get_data(data_size):
        # full data
        t_evidence, t_labels = ddata.get(size,digits,noisy_image_count=100,noise_count=size)
        v_evidence, v_labels = ddata.get(size,digits,noisy_image_count=200,noise_count=size)
        # random subset
        t_percentage = data_size / len(t_labels)
        v_percentage = max(1000,data_size)/len(v_labels) # no less than 1000
        t_evidence, t_labels = data.random_subset(t_evidence,t_labels,t_percentage)
        v_evidence, v_labels = data.random_subset(v_evidence,v_labels,v_percentage)
        return t_evidence, t_labels, v_evidence, v_labels
    
    # get model
    net, inputs, output = dmodel.get(size,digits,testing,use_bk,tie_parameters,remove_common=False)
    # compile model into circuit
    circuit = tac.TAC(net,inputs,output,trainable=True,profile=False)
    u.echo(f,f'circuit size {circuit.size:,}, paramater count {circuit.parameter_count}\n')
    
    for data_size, count in zip(data_sizes,tries):
        u.echo(f,f'==data size {data_size}')
        t_evidence, t_labels, v_evidence, v_labels = get_data(data_size)
        u.echo(f,f'  train {len(t_labels)}, test {len(v_labels)}')
        u.echo(f,f'  accuracy ({count}):',end='',flush=True)
        sample = []
        for i in range(count):
            circuit.fit(t_evidence,t_labels,loss_type='CE',metric_type='CA',batch_size=batch_size)
            acc = 100*circuit.metric(v_evidence,v_labels,metric_type='CA')
            sample.append(acc)
            u.echo(f,f' {acc:.2f}',end='',flush=True)
        u.echo(f,f'\naccuracy mean {s.mean(sample):.2f}, std {s.stdev(sample):.2f}\n')
    
    all_time = time.time() - start_time
    u.echo(f,f'Total Time: {all_time:.3f} sec') 
    f.close()