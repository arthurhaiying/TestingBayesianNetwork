import numpy as np

class LookUpTable:
    # lookup table that maps posterior value to cpt
    # Use average cpt for posteriors that map to the same interval
    epsilon = np.finfo('float32').eps 
    def __init__(self,size,min_value=0.0,max_value=1.0,num_intervals=10):
        assert min_value < max_value
        self._min = min_value
        self._max = max_value
        self._size = size
        self._num_intervals = num_intervals
        self._delta = (max_value - min_value) / num_intervals
        self._storage = [(np.zeros(size), 0)] * self._num_intervals
        # for each interval keep track of (avg_value, num_entries)
    def __setitem__(self,key,value):
        if key < self._min or key > self._max:
            raise KeyError("Key out of range: {.5f}".format(key))
        if len(value) != self._size:
            raise ValueError("Invalid value size: {} should be {}".format(len(value),self._size))
        index = int((key-self._min-self.epsilon) / self._delta) 
        # becareful if key is very close to self._min or self._max
        avg_value, count = self._storage[index]
        new_avg_value = (avg_value * count + value) / (count + 1)
        new_count = count + 1
        self._storage[index] = new_avg_value, new_count
    def __getitem__(self,key):
        if key < self._min or key > self._max:
            raise KeyError("Key out of range: {.5f}".format(key))
        index = min(int((key - self._min) / self._delta), self._num_intervals-1)
        return self._storage[index][0]
    def getNumConflicts(self):
        res = 0
        for _, count in self._storage:
            if count > 1:
                res += count-1
        return res

    def thresholds(self):
        return np.arange(self._delta,self._max,self._delta)

    def cond_cpts(self):
        conds = []
        for cond,count in self._storage:
            if count == 0:
                cond = 1.0/self._size*np.ones(self._size)
            conds.append(cond)
        return np.array(conds)


class LookUpTableV2:
    # lookup table that map posterior to cpt
    # use weighted average cpt for posteriors that map to the same interval

    # In each interval, keep track of current average cpt, num of entries, and weights sum
    class Record:
        def __init__(self,cpt,count,weights_acc):
            self.cpt = cpt
            self.count = count
            self.weights_acc = weights_acc

    epsilon = np.finfo('float32').eps 
    def __init__(self,size,min_value=0.0,max_value=1.0,num_intervals=10):
        assert min_value < max_value
        self._min = min_value
        self._max = max_value
        self._size = size
        self._num_intervals = num_intervals
        self._delta = (max_value - min_value) / num_intervals
        self._storage = [self.Record(np.zeros(size),0,0.0)] * self._num_intervals
        # for each interval keep track of (avg_value, num_entries)
    def __setitem__(self,key,value):
        if key < self._min or key > self._max:
            raise KeyError("Key out of range: {.5f}".format(key))
        index = int((key-self._min-self.epsilon) / self._delta) 
        # becareful if key is very close to self._min or self._max
        weight,cpt = value
        if len(cpt) != self._size:
            raise ValueError("Invalid cpt size: {} should be {}".format(len(value),self._size))
        
        record = self._storage[index]
        #weight,cpt = value
        avg_cpt, count, weights_acc = record.cpt, record.count, record.weights_acc
        new_avg_cpt = (avg_cpt*weights_acc + weight*cpt) / (weights_acc + weight) # compute new average cpt
        # increment count
        count += 1
        weights_acc += weight
        self._storage[index] = self.Record(new_avg_cpt,count,weights_acc)
    def __getitem__(self,key):
        if key < self._min or key > self._max:
            raise KeyError("Key out of range: {.5f}".format(key))
        index = min(int((key - self._min) / self._delta), self._num_intervals-1)
        return self._storage[index].cpt
    def getNumConflicts(self):
        res = 0
        for record in self._storage:
            if record.count > 1:
                res += record.count-1
        return res

    def thresholds(self):
        return np.arange(self._delta,self._max,self._delta)

    def cond_cpts(self):
        conds = []
        for record in self._storage:
            if record.count == 0:
                cond = 1.0/self._size*np.ones(self._size)
            else:
                cond = record.cpt
            conds.append(cond)
        return np.array(conds)
