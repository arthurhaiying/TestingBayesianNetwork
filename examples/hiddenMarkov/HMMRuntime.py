from examples.hiddenMarkov.HMMExperiment import *
from multiprocessing import Process, Pipe, Queue
import utils.utils as u
import itertools as iter
import numpy as np
import os.path
import shutil
import csv
import sys

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# experiment settings
NUM_WORKERS = 25

def run_single(size,card,order,transition,emission,missing_card,train_card,num_trial):
    # Given cpt of n order testing hmm, compare tac and ac performance
    # define first order HMM and testing HMM models
    if not missing_card:
        # if not missing card, learn the forward query Pr(H_n|E[0:n-1])
        hmm = HiddenMarkovModel(size,card,order=1,testing=False) 
        thmm = HiddenMarkovModel(size,card,order=1,testing=True,sel_type=SELECT_TYPE,gamma_opt=GAMMA_OPTION)
        true_hmm = HiddenMarkovModel(size,card,order=order,testing=False,param=True,transition1=transition,emission=emission)
        # the true model will also have nonzero loss
    else:
        # if missing card, learn to predict the next evidence Pr(E_n|E[0:n-1])
        inputs = ['e_'+str(i) for i in range(size-1)]
        output = 'e_'+str(size-1)
        hmm = HiddenMarkovModel(size,card=train_card,evid_card=card,order=1,testing=False,inputs=inputs,output=output) 
        thmm = HiddenMarkovModel(size,card=train_card,evid_card=card,order=1,testing=True,inputs=inputs,output=output,
            sel_type=SELECT_TYPE,gamma_opt=GAMMA_OPTION)
        true_hmm = HiddenMarkovModel(size,card,order=order,testing=False,inputs=inputs,output=output,param=True,
            transition1=transition,emission=emission)
    logging.info("Start trial {}...".format(num_trial))
    # sample the training set
    evidences,marginals = direct_sample_training_set(size,card,order,transition,emission,missing_card,num_trial=num_trial)
    # rememeber to use original card for sampling data
    loss_ac = cross_validate(hmm,evidences,marginals,num_folds=5,num_trial=num_trial) # train hmm 
    loss_tac = cross_validate(thmm,evidences,marginals,num_folds=5,num_trial=num_trial) # train testing hmm
    loss_true = true_hmm.metric(evidences,marginals,metric_type='CE') 
    # the true model will also have nonzero loss since the labels are zero/one
    gain = loss_tac / loss_ac
    print("Trial {0}: the AC loss is {1:.5f} and the TAC loss is {2:.5f}".format(num_trial,loss_ac,loss_tac))
    logging.info("Finish trial {}.".format(num_trial))
    result = (num_trial,loss_ac,loss_tac,loss_true,gain)
    return result

def run_worker(size,card,order,missing_card,train_card,conn,queue):
    # worker: receive hmm cpts from master and do experiments
    # need to redirest stdout
    dirname = os.path.abspath(os.getcwd())
    filename = "out.txt"
    filepath = os.path.join(dirname,'examples','hiddenMarkov','output',filename)
    sys.stdout = open(filepath,'w+')
    while 1:
        num_trial,transition,emission = conn.recv()
        if num_trial == "Finished":
            break
        else:
            u.input_check(np.array(transition).shape == (card,)*(order+1), "wrong size for transition matrix")
            u.input_check(np.array(emission).shape == (card, card), "wrong size for emission matrix")
            num_trial = int(num_trial)
            result = run_single(size,card,order,transition,emission,missing_card,train_card,num_trial)
            # send result to queue
            queue.put(result)

def report(filepath,queue):
    fields = ['Trial','AC loss','TAC loss', 'True loss', 'TAC/AC']
    with open(filepath,mode='w+') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        count = 0
        while 1:
            if not queue.empty():
                # get new report
                num_trials, loss_ac, loss_tac, loss_true, gain = queue.get()
                if isinstance(num_trials,str) and num_trials == "End":
                    break
                else:
                    writer.writerow([num_trials, loss_ac, loss_tac, loss_true, gain])
                    count += 1
                    if count % NUM_WORKERS == 0:
                        f.flush()        

def run_master(size,card,order,missing_card=False,train_card=None,num_workers=NUM_WORKERS):
    if missing_card:
        u.input_check(train_card is not None, "Please provide cardinality of hidden nodes for training")
        u.input_check(order==1, "Must use first order hmm if miss cardinality")
    # set opsgraph options
    queue = Queue()
    # collect experiment reports from workers
    # enumerate possible cpts
    # initialize workers
    workers, pipes = [],[]
    for i in range(num_workers):
        parent_conn,child_conn = Pipe()
        p_i = Process(target=run_worker,args=(size,card,order,missing_card,train_card,child_conn,queue))
        workers.append(p_i)
        pipes.append(parent_conn)
    # start running the workers
    for worker in workers:
        worker.start()
    # save report
    if not missing_card:
        filename = 'HMM_size_{}_order_{}_report.csv'.format(size,order)
    else:
        filename = 'HMM_size_{}_card_{}_train_card_{}_report.csv'.format(size,card,train_card)
    # clear the cpt logs
    dirname = os.path.abspath(os.getcwd())
    cpts_dirname = os.path.join(dirname,'logs','cpts')
    try:
        shutil.rmtree(cpts_dirname)
    except:
        print("Error removing logs/cpts directory")
        exit(1)
    filepath = os.path.join(dirname,'examples','hiddenMarkov',filename)
    reporter = Process(target=report,args=(filepath,queue))
    reporter.start()
    # start enumerating hmms
    num_trials = 0
    worker_ids = iter.cycle(range(num_workers)) # visit workers in cycle
    fpath = os.path.join(dirname,'examples','hiddenMarkov','models.txt')
    f = open(fpath,mode='w')
    for transition,emission in enumerate_HMM(card,order,STEP_SIZE):
        num_trials += 1
        if num_trials > 1:
            break
        worker_id = next(worker_ids)
        pipe = pipes[worker_id]
        data = (str(num_trials), transition, emission)
        pipe.send(data) # send job to worker_i
        def print_cpt():
            # helper function to print cpt
            f.write('%d transition:\n' %num_trials)
            f.write(np.array2string(transition,formatter={'float_kind': lambda x:"%.5f"%x}))
            f.write('\n\n')
            f.write('%d emission:\n' %num_trials)
            f.write(np.array2string(emission, formatter={'float_kind': lambda x:"%.5f"%x}))
            f.write('\n\n')
            if num_trials % NUM_WORKERS == 0:
                f.flush()
        print_cpt()
        # finish enumerating hmms
    f.close()
    for pipe in pipes:
        pipe.send(("Finished", None, None))
        # wait workers to terminate
    for worker in workers:
        worker.join()
    # save reports
    queue.put(("End",None,None,None,None))
    reporter.join()
    print("Finish HMM Experiment.")


        