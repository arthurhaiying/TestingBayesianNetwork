from examples.hiddenMarkov.HMMExperiment import *
from multiprocessing import Process, Pipe, Queue
import utils.utils as u
import itertools as iter
import os.path
import csv
import sys

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# experiment settings
NUM_WORKERS = 25

def run_single(size,card,order,transition,emission,num_trial):
    # Given cpt of n order testing hmm, compare tac and ac performance
    # define first order HMM and testing HMM models
    hmm = HiddenMarkovModel(size,card,order=1,testing=False) 
    thmm = HiddenMarkovModel(size,card,order=1,testing=True,sel_type=SELECT_TYPE,gamma_opt=GAMMA_OPTION) 
    logging.info("Start trial {}...".format(num_trial))
    # sample the training set
    evidences,marginals = direct_sample_training_set(size,card,order,transition,emission,num_trial=1)
    loss_ac = cross_validate(hmm,evidences,marginals,num_folds=5) # train hmm 
    loss_tac = cross_validate(thmm,evidences,marginals,num_folds=5) # train testing hmm
    print("Trial {0}: the AC loss is {1:.5f} and the TAC loss is {2:.5f}".format(num_trial,loss_ac,loss_tac))
    logging.info("Finish trial {}.".format(num_trial))
    result = (num_trial,loss_ac,loss_tac)
    return result

def run_worker(size,card,order,conn,queue):
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
            result = run_single(size,card,order,transition,emission,num_trial)
            # send result to queue
            queue.put(result)

def report(filepath,queue):
    fields = ['Trial','AC loss','TAC loss']
    with open(filepath,mode='w+') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        count = 0
        while 1:
            if not queue.empty():
                # get new report
                num_trials, loss_ac, loss_tac = queue.get()
                if isinstance(num_trials,str) and num_trials == "End":
                    break
                else:
                    writer.writerow([num_trials, loss_ac, loss_tac])
                    count += 1
                    if count % NUM_WORKERS == 0:
                        f.flush()        

def run_master(size,card,order,num_workers=NUM_WORKERS):
    queue = Queue()
    # collect experiment reports from workers
    # enumerate possible cpts
    # initialize workers
    workers, pipes = [],[]
    for i in range(num_workers):
        parent_conn,child_conn = Pipe()
        p_i = Process(target=run_worker,args=(size,card,order,child_conn,queue))
        workers.append(p_i)
        pipes.append(parent_conn)
    # start running the workers
    for worker in workers:
        worker.start()
    # save report
    filename = 'HMM_size_{}_order_{}_report.csv'.format(size,order)
    dirname = os.path.abspath(os.getcwd())
    filepath = os.path.join(dirname,'examples','hiddenMarkov',filename)
    reporter = Process(target=report,args=(filepath,queue))
    reporter.start()
    # start enumerating hmms
    num_trials = 0
    worker_ids = iter.cycle(range(num_workers)) # visit workers in cycle
    for transition,emission in enumerate_HMM(card,order,STEP_SIZE):
        num_trials += 1
        if num_trials > 256:
            break
        worker_id = next(worker_ids)
        pipe = pipes[worker_id]
        data = (str(num_trials), transition, emission)
        pipe.send(data) # send job to worker_i
        # finish enumerating hmms
    for pipe in pipes:
        pipe.send(("Finished", None, None))
        # wait workers to terminate
    for worker in workers:
        worker.join()
    # save reports
    queue.put(("End",None,None))
    reporter.join()
    print("Finish HMM Experiment.")


        







