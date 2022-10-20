import examples.digits.play as dplay
import examples.networks.play as nplay
import examples.rectangles.play as rplay

def play():
    
    """ networks: see networks/play.py for what the functions below do """
    # nplay.train_nets()
    # nplay.train_kidney()
    # nplay.train_fn2(size=8, card=3)


    """ rectangles: see rectangles/play.py for what the functions below do"""

    """ validate """
    rplay.validate(size=10,output='label',testing=False,elm_method='minfill')
    rplay.validate(size=12,output='label',testing=False,elm_method='tamaki heuristic',elm_wait=60)
    for s in (8,10,12,14):
        rplay.validate(size=s,output='label',testing=False)
        rplay.validate(size=s,output='label',testing=True)

    #eval_rectangles_all() # run with -s command line option
    
    """ train """
    #rplay.train(size=10,output='label',data_size=1000,testing=False,use_bk=True,tie_parameters=False)
    #rplay.train(size=10,output='label',data_size=1000,testing=False,use_bk=False,tie_parameters=False)

    #train_rectangle_all(10,use_bk=True,tie_parameters=True) # run with -s command line option
    #train_rectangle_all(10,use_bk=False,tie_parameters=True)
    #train_rectangle_all(10,use_bk=True,tie_parameters=False)
    #train_rectangle_all(10,use_bk=False,tie_parameters=False)


    """ digits: see digits/play.py for what the functions below do """
    
    """ validate """
    #dplay.validate(size=12,digits=range(10),testing=False,elm_method='minfill')
    #dplay.validate(size=12,digits=range(10),testing=False,elm_method='tamaki heuristic',elm_wait=60)
    #for s in (8,10,12,14):
    #    dplay.validate(size=s,digits=range(10),testing=False)
    #    dplay.validate(size=s,digits=range(10),testing=True)

    #eval_digits_all() # run with -s command line option
    
    """ train """ 
    #dplay.train(size=10,digits=range(10),data_size=1000,testing=False,use_bk=True,tie_parameters=False)
    #dplay.train(size=8,digits=range(10),data_size=1000,testing=False,use_bk=False,tie_parameters=False)
  
    #train_digits_all(10,use_bk=True,tie_parameters=True) # run with -s command line option
    #train_digits_all(10,use_bk=True,tie_parameters=False)


def eval_rectangles_all():
    sizes   = (10,12,14,16) #(8,10,12,14,16,20)
    output  = 'label'
    testing = False
    
    rplay.eval_all(sizes,output,testing)
    
def eval_digits_all():
    sizes   = (10,12,14,16) #(8,10,12,14,16)
    digs    = range(10)
    testing = False
    
    dplay.eval_all(sizes,digs,testing)

def train_rectangle_all(size,use_bk,tie_parameters):
    output     = 'label'
    tries      = (5,)*6
    data_sizes = (25,50,100,250,500,1000)
    batch_size = 32
    
    rplay.train_all(size,output,tries,data_sizes,testing=False,
        use_bk=use_bk,tie_parameters=tie_parameters,batch_size=batch_size)
    #rplay.train_all(size,output,tries,data_sizes,testing=True,
    #    use_bk=use_bk,tie_parameters=tie_parameters,batch_size=batch_size)
  
def train_digits_all(size,use_bk,tie_parameters):
    digs       = range(10)
    tries      = (5,)*6
    data_sizes = (25,50,100,250,500,1000)
    batch_size = 32
    
    dplay.train_all(size,digs,tries,data_sizes,testing=False,
        use_bk=use_bk,tie_parameters=tie_parameters,batch_size=batch_size)
    #dplay.train_all(size,digs,tries,data_sizes,testing=True,
    #    use_bk=use_bk,tie_parameters=tie_parameters,batch_size=batch_size)