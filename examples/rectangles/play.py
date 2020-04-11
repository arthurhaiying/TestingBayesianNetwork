import numpy as np
import time
import statistics as s

import tac
import utils.paths as paths
import utils.precision as p
import train.data as data
import examples.rectangles.model as rmodel
import examples.rectangles.data as rdata
import utils.utils as u


# validates ac/tac for rectangle model by running on clean images (we know the labels)
def validate(size,output,testing,elm_method='minfill',elm_wait=30):
    
    circuit_type = 'TAC' if testing else 'AC'
    
    # get data (ground truth)
    evidence, labels = rdata.get(size,output)
    
    u.show(f'\n===Checking {circuit_type} for rectangle {output} in {size}x{size} images: {len(labels)} total')
    
    # get model
    bn, inputs = rmodel.get(size,output,testing=testing,use_bk=True,tie_parameters=False)
    
    # compile model
    AC = tac.TAC(bn,inputs,output,trainable=False,profile=False,
            elm_method=elm_method,elm_wait=elm_wait)

    # evaluate TAC on evidence to get predictions
    predictions = AC.evaluate(evidence)

    # verify that predictions match one_hot_marginals
    if u.equal(predictions,labels): 
        u.show('\n===All good!')
    else:
        u.show('***bumper!!!')
        quit()
     
    
# trains the ac/tac of rectangle model on noisy images
# testing: whether tac or ac
# use_bk: whether to incorporate background knowledge in tac/ac
def train(size,output,data_size,testing,use_bk,tie_parameters):
    circuit_type = 'TAC' if testing else 'AC'
    u.show(f'\n===Training {circuit_type} for rectangle {output} in {size}x{size} images, use_bk {use_bk}, tie {tie_parameters}')

    # get training and testing data (labels are one-hot)
    t_evidence, t_labels = rdata.get(size,output,noisy_image_count=size,noise_count=size)
    v_evidence, v_labels = rdata.get(size,output,noisy_image_count=2*size,noise_count=2*size)
    
    # get model
    bn, inputs = rmodel.get(size,output,testing,use_bk,tie_parameters)
    
    # compile model to circuit
    circuit = tac.TAC(bn,inputs,output,trainable=True,profile=False)
    
    # use a random subset of the generated data
    t_percentage = data_size / len(t_labels)
    v_percentage = max(1000,data_size)/len(v_labels) # no less than 1000
    t_evidence, t_labels = data.random_subset(t_evidence,t_labels,t_percentage)
    v_evidence, v_labels = data.random_subset(v_evidence,v_labels,v_percentage)
  
    # train AC
    circuit.fit(t_evidence,t_labels,loss_type='CE',metric_type='CA')
    
    # compute accuracy
    accuracy = circuit.metric(v_evidence,v_labels,metric_type='CA')
    u.show(f'\n{circuit_type} accuracy {100*accuracy:.2f}')
    
    return (100*accuracy, circuit)
    
    # verify reported accuracy
    #predictions = circuit.evaluate(v_evidence)
    #u.accuracy(accuracy,v_labels,predictions)
    
"""
scripts for logging experiments
"""

def eval_all(sizes,output,testing):
    circuit_type = 'TAC' if testing else 'AC'
    fname = paths.exp / u.time_stamp(f'eval_rect_{output}_{testing}','txt')
    f     = open(fname,'w+')
    u.echo(f,f'\n===Rectangle: evaluating {circuit_type} for {output}===')
    u.echo(f,'output logged into logs/exp/')
    start_time = time.time()
    for size in sizes: 
        eval(f,size,output,testing)
    all_time = time.time() - start_time
    u.echo(f,f'\nTotal Time: {all_time:.3f} sec') 
    f.close()
        
def eval(f,size,output,testing):
    circuit_type = 'TAC' if testing else 'AC'
    # get data (ground truth)
    evidence, marginals = rdata.get(size,output)
    ecount = len(marginals) # number of examples
        
    u.echo(f,f'\n==rectangle {size}x{size} images: {ecount} total')
    
    # get model
    bn, inputs = rmodel.get(size,output,testing=testing,use_bk=True,tie_parameters=False)
    
    # compile model
    s_time = time.time()
    u.echo(f,f'\ncompiling {circuit_type}:',end='')
    AC = tac.TAC(bn,inputs,output,trainable=False,profile=False)
    t = time.time()-s_time
    u.echo(f,f' {t:.1f} sec')
    u.echo(f,f'  {circuit_type} size {AC.size:,}\n  (sep) binary rank {AC.binary_rank:.1f}, rank {AC.rank}')
    
    # evaluate AC on evidence to get predictions
    u.echo(f,f'evaluating {circuit_type}:\n',end='',flush=True)
    predictions, t1, batch_size = AC.evaluate(evidence,report_time=True)
    u.echo(f,f'  batch size {batch_size}')
    u.echo(f,f'  {t1:.2f} sec, {1000*t1/ecount:.1f} ms per example')


# trains rectangle model for different data sizes 
# runs a number of tries for each dataset size and reports best accuracy
# results are logged into a time-stamped and descriptively-named file in 'logs/exp/'
def train_all(size,output,tries,data_sizes,testing,use_bk,tie_parameters,batch_size):
    start_time = time.time()
    
    fname = paths.exp / u.time_stamp(f'train_rect_{size}_{output}_{tries}_{testing}_{use_bk}_{tie_parameters}','txt')
    f     = open(fname,'w+')
    
    u.echo(f,f'\nrectangle {size} x {size}, output {output}, data_sizes {data_sizes}, testing {testing}, use_bk {use_bk}, tie {tie_parameters}\n')
    u.echo(f,f'fixed batch size {batch_size}')
    u.echo(f,'output logged into logs/exp/')
        
    def get_data(data_size):
        # full data
        t_evidence, t_labels = rdata.get(size,output,noisy_image_count=size,noise_count=size)
        v_evidence, v_labels = rdata.get(size,output,noisy_image_count=2*size,noise_count=2*size)
        # random subset
        t_percentage = data_size / len(t_labels)
        v_percentage = max(1000,data_size)/len(v_labels) # no less than 1000
        t_evidence, t_labels = data.random_subset(t_evidence,t_labels,t_percentage)
        v_evidence, v_labels = data.random_subset(v_evidence,v_labels,v_percentage)
        return t_evidence, t_labels, v_evidence, v_labels
    
    # get model
    net, inputs = rmodel.get(size,output,testing,use_bk,tie_parameters)
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