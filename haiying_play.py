import utils.VE as VE
import train.data as data
import examples.networks.model as model





def testcase1():
    bn = model.cancer_bn()
    map_vars = ['A']
    evd_vars = ['D', 'E']
    cards = (2, 2)
    evidence = [[0, 0], [0, 1], [1, 0], [1, 1]]
    evidence = data.evd_id2col(evidence, cards)
    post_prob = VE.posteriors(bn, evd_vars, map_vars[0], evidence)
    map_prob = VE.VE_MAP(bn, map_vars, evd_vars, evidence)
    print("posteriors: {}".format(post_prob))
    print("map probabilities: {}".format(map_prob))

def testcase2():
    bn = model.cancer_bn()
    map_vars = ['A']
    evd_vars = ['D', 'E']
    cards = (2, 2)
    evidence = [[0, 0], [0, 1], [1, 0], [1, 1]]
    evidence = data.evd_id2col(evidence, cards)
    post_prob = VE.posteriors(bn, evd_vars, map_vars[0], evidence)
    map_prob, map_inst = VE.VE_MAP2(bn, map_vars, evd_vars, evidence, return_inst=True)
    print("posteriors: {}".format(post_prob))
    print("map probabilities: {} instantiations: {}".format(map_prob, map_inst))

def play():
    testcase1()
    testcase2()













