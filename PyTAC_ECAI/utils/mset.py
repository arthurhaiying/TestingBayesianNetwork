"""
Multi-Set class
"""

class mset:
    def __init__(self,elements=[]):
        # maps each element to its number of occurrences (must be > 0)
        # if an element does not appear in members keys, then it occurs 0 times
        self.members = {} 
        for e in elements:
            self.add(e)
            
    def __str__(self):
        s = ', '.join([f'{e}:{c}' for e,c in self.members.items()])
        return f'({s})'
        
    # make multi-set into an iterator
    def __iter__(self):
        self.list = list(self.members.keys())
        return self
    def __next__(self):
        if self.list:
            return self.list.pop()
        else:
            raise StopIteration
        
    # overloading operators
    
    # =, !=, <, <=, >, >= operate as if we have regular sets
    def __eq__(self,other): # =
        return self.members.keys() == other.members.keys()
    
    def __ne__(self,other): # !=
        return self.members.keys() != other.members.keys()
        
    def __lt__(self,other): # <
        return self.members.keys() < other.members.keys()
    
    def __le__(self,other): # <=
        return self.members.keys() <= other.members.keys()
    
    def __gt__(self,other): # >
        return self.members.keys() > other.members.keys()
    
    def __ge__(self,other): # >=
        return self.members.keys() >= other.members.keys()
    
    def __or__(self,other): # |
        return self.union(other)
        
    def __and__(self,other): # &
        return self.intersection(other)
        
    # number of elements with > 0 occurrences in multi-set
    def __len__(self):
        return len(self.members.keys())
        
    # whether multi-set contains element e (> 0 occurrences)
    def __contains__(self,e): # in
        return self.contains(e)
        
    # interface
    
    # returns union of two multi-sets: adds their occurrences 
    def union(self,mset1):
        members2 = {e:c for e,c in self.members.items()}
        for e,c in mset1.members.items():
            if e in members2:
                members2[e] += c
            else:
                members2[e]  = c
        mset2 = mset()
        mset2.members = members2
        return mset2     
        
    # returns intersection of two multi-sets: takes min of their occurrences 
    def intersection(self,s):
        s1, s2 = (self,s) if len(self) <= len(s) else (s,self)
        members2 = {}
        for e,c in s1.members.items():
            if e in s2.members:
                members2[e] = min(c,s2.members[e])
        mset2 = mset()
        mset2.members = members2
        return mset2
    
    # whether multi-set contains element e (> 0 occurrences)
    def contains(self,e):
        return e in self.members
        
    # number of occurrences of element e in multi-set
    def count(self,e):
        if e in self.members:
            return self.members[e]
        return 0
       
    # adds element e to multi-set 
    def add(self,e):
        if e in self.members:
            self.members[e] += 1
        else:
            self.members[e]  = 1
            
    # adds elements to multi-set
    def extend(self,elements):
        for e in elements: 
            self.add(e)