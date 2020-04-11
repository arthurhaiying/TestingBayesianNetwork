import compile.prune as prune
import utils.precision as p
import utils.utils as u

"""
Constructing a TAC using a trace of jointree inference.

This file exposes only one function, trace(), which adds operations to an ops_graph 
that are sufficient to construct a TAC when the ops_graph is executed (see opsgraph.py).

In particular, the function adds operations to the ops_graph that construct tensors for 
evidence, cpts, selected cpts and posterior over the given query node. 

The ops_graph is an abstraction of a tensorflow graph (tf_graph). When the ops_graph 
operations are 'executed', they construct tensors in a tf_graph that represents a TAC.

Inference (to construct ops_graph) and execution (to construct tf_graph) happen when 
a TAC object is constructed. The TAC object is simply a wrapper of the tf_graph. 
The TAC object can be used to simulate data from the TAC, train the TAC based 
on labeled data, and evaluate the TAC at some given evidence. All these functions are 
done on the wrapped tf_graph (see tac.py).

Evidence vars will always exist in the ops_graph and tf_graph even if they are pruned
during jointree inference (pruned evidences node will have disconnected tensors so their
values in user evidence will have no effect on TAC training or evaluation).

In the following code:
-- We use i, c1, c2, p, root, host for jointree nodes.
-- We use var for tbn nodes.
"""


""" 
Adds operations to og that will construct a tac when the ops are executed.
"""

# query_var: tbn node
# evidence_vars: tbn nodes
# jt: Jointree
# og: OpsGraph

def trace(query_var,evidence_vars,tbn,jt,og):
    assert tbn._for_inference
    assert tbn == jt.tbn
    # the following need to be relaxed
    assert not query_var.has_pruned_values() 
    assert not any(var.has_pruned_values() for var in evidence_vars)
    
    ops = og.add_evidence_ops(evidence_vars) # ops that construct tensors for evidence
    jt.declare_evidence(evidence_vars,ops)   # save ops in jointree for later lookup
    
    # qcontext: captures the pruned tbn used to compute posterior on query_var
    qcontext = prune.for_node_posterior(query_var,evidence_vars,tbn)
    
    # add ops that will create tensors for selected cpts (if any)
    for var in qcontext.testing_nodes:       # top-down
        __selected_cpt(var,qcontext,jt,og)   # also prunes
        
    # add ops that will create tensor for the posterior over query_node
    __node_posterior(query_var,qcontext,jt,og)
            
    hit_rate     = jt.hits*100/jt.lookups if jt.lookups > 0 else 0
    all_count    = len(qcontext.testing_nodes)
    live_count   = qcontext.live_count 
    sval_count   = sum(1 for n in tbn.nodes if len(n.values) == 1)
    pruned_count = len(tbn.nodes)-len(qcontext.nodes)
    pruned_perct = pruned_count*100/len(tbn.nodes)
    u.show(f'  Tracing posterior for \'{query_var.name}\':\n'
           f'    og-cache lookups {jt.lookups}, hits {jt.hits}, rate {hit_rate:.1f}%\n'
           f'    selected cpts: all {all_count}, live {live_count}\n'
           f'    single-value nodes: {sval_count}\n'
           f'    pruned nodes: {pruned_count}, percentage {pruned_perct:.1f}%')
    
         
""" 
Adds ops that construct a tensor for the selected cpt of tbn node.
"""

def __selected_cpt(var,qcontext,jt,og): # var is a tbn node
    assert var.testing
    assert jt.lookup_sel_cpt_op(var) is None
    
    # qcontext captures the pruned tbn used to compute the posterior on query var
    # scontext captures the pruned tbn used to compute the posterior on parents of var
    # the tbn captured by scontext is a subset of the one captured by qcontext
    scontext             = prune.for_selection(var,qcontext)
    ppost_op, host, view = __parents_posterior(var,scontext,jt,og)
    
    cpt1_op    = og.add_cpt_op(var,var.cpt1,'cpt1')
    cpt2_op    = og.add_cpt_op(var,var.cpt2,'cpt2')
    sel_cpt_op = og.add_selected_cpt_op(var,cpt1_op,cpt2_op,ppost_op)
    jt.save_sel_cpt_op(var,sel_cpt_op) # cache it, looked up by _cpt_evd()
                

"""              
Returns an op that constructs a tensor for the posterior over parents of tbn node,
which is used to select a cpt for the tbn node.

scontext: captures the pruned tbn that is used for computing the parents posterior,
see prune.py
"""

def __parents_posterior(var,scontext,jt,og): # var is a tbn node
    view    = jt.view_for_query(var,scontext,verbose=False)
    host    = view.host        # jointree node that hosts cpt of var
    parents = set(var.parents) # we need the posterior on these vars
    if view.empty:             # pruned tbn has only one node (var)
        assert not parents
        # -testing node lost its parents as they have one value
        # -posterior over parents is constant 1 and the testing node is dead
        op = og.add_scalar_op(1.)
    else: # pruned tbn has more than one node
        assert parents
        op, evd, sep = __separator_marginal(view,jt,og) # for edge (host,root)
        # sep = parents unless some parents are clamped
        if parents != sep: # separator is a strict subset of parents
            assert evd and parents > sep
            vars = set(sep)
            for p in parents - sep:
                assert p._clamped
                vars.add(p)
                ope = jt.get_evd_op(p)
                op  = og.add_multiply_op(op,ope,vars)
        if evd: op  = og.add_normalize_op(op,parents)
    return op, host, view
    

"""
Adds ops that construct a tensor for the posterior over a tbn node.

qcontext: captures the pruned tbn that is used for computing the node posterior,
see prune.py
"""

def __node_posterior(var,qcontext,jt,og): # var is a tbn node
    view    = jt.view_for_query(var,qcontext)
    host    = view.host # jointree node hosting the cpt of var
    h_cls   = view.cls(host)
    v_cls   = set([var])
    op, evd = __cpt_evd(host,view,jt,og) # op over cluster h_cls
    
    # we don't use mulpro here as it requires three separators connected to a node
    if not view.empty: # pruned tbn has more than one node
        mess_op, mess_evd, _ = __separator_marginal(view,jt,og) # for edge (root,host)
        evd = evd or mess_evd
        op = og.add_multiply_op(op,mess_op,h_cls)
    if h_cls > v_cls: 
        op = og.add_project_op(op,v_cls)
    if evd: 
        og.add_normalize_op(op,v_cls)
       
            
""" 
Returns an op that constructs a tensor for the cpt at view leaf i and its evidence.
"""

def __cpt_evd(i,vw,jt,og): # i is a jointree view node (leaf)
    var    = i.var
    op_evd = jt.lookup_cpt_evd_op(var) # could be a selected cpt op
    if op_evd: return op_evd
    
    if var.testing: # cpt op already computed and cached
        cpt_op = jt.lookup_sel_cpt_op(var)
        assert cpt_op
    else:    
        cpt_op = og.add_cpt_op(var,var.cpt,'cpt')
        
    # -add evidence (if any) to cpt
    # -evidence could be on var or its (clamped) parents
    # -hence, it is possible evd=false yet cpt_evd_op != cpt_op
    #  (when no evidence on var but some parent has hard evidence and is clamped)
    op_evd = __inject_evd(i,cpt_op,vw,jt,og)
    jt.save_cpt_evd_op(var,op_evd) # cache it
    return op_evd # tuple (op,boolean)


"""
Multiplies cpt at view leaf with evidence on the cpt var and 'clamped' parents.

Adding evidence on clamped parents amounts to conditioning the cpt on these 
parents since we sum out these parents from the cluster when computing the 
message to its neighbor (see set_separators_and_clusters() in separators.py).

A functional var may have multiple (replicated) cpts: soft evidence is injected
into exactly _one_ of these cpts, but hard evidence is injected into _all_ of them.

The evd flag is used to decide whether we need to normalize or not. Evidence 
on clamped parents is not integrated into this flag since such evidence only
serves to select a subset of the cpt (does not actually lead to a posterior).
"""

def __inject_evd(i,cpt_op,vw,jt,og): # i is a jointree view node (leaf)
    cls = vw.cls(i)
    var = i.var                     # cpt at i is for var  
    evd = vw.has_evidence_at(var,i) # whether var has evidence that is assigned to i
    # check if we are entering evidence on var at leaf i
    if evd:
        evd_op     = jt.get_evd_op(var)
        cpt_evd_op = og.add_multiply_op(cpt_op,evd_op,cls)
    else:
        cpt_evd_op = cpt_op
    # check if some parents of var are clamped (have evidence)
    for p in var.parents:
        if p._clamped:
            assert vw.has_evidence(p)
            evd_op     = jt.get_evd_op(p)
            cpt_evd_op = og.add_multiply_op(cpt_evd_op,evd_op,cls)
    return cpt_evd_op, evd
    

""" 
Adds ops that construct a tensor for the marginal over separator(host,root) of view. 
"""

def __separator_marginal(vw,jt,og):
    assert not vw.empty
    
    # returns messages i --> p, which is a pair (op, evd): 
    # --op will create a tensor over separator(i,p)
    # --evd is whether message depends on evidence
    def message(i,p):
        # lookup message from cache
        signature = vw.signature(i,p)
        oe = jt.lookup_message_op(signature)
        if oe: return oe # found it
        # compute message
        cls = vw.cls(i)
        sep = vw.sep(i)
        assert sep <= cls
        if vw.leaf(i):
            op, evd = __cpt_evd(i,vw,jt,og)
            if cls > sep: 
                op = og.add_project_op(op,sep)
        else:
            c1, c2    = vw.children(i)
            op1, evd1 = message(c1,i)
            op2, evd2 = message(c2,i)
            evd       = evd1 or evd2
            if cls > sep: # otherwise, we don't need to project
                op = og.add_mulpro_op(op1,op2,sep)
            else: # multiply is sufficient
                op = og.add_multiply_op(op1,op2,cls)
        jt.save_message_op(signature,(op,evd)) # cache it
        #show('message %d -> %d' % (i.id,p.id))
        return (op, evd)
        
    host    = vw.host
    root    = vw.root
    op, evd = message(root,host)
    sep     = vw.sep(root)
    
    return (op, evd, sep)
    