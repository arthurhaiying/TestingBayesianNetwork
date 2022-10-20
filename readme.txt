PyTAC Version 1.22
==================

Usage
=====

PyTAC is called as follows:

python main.py

It will load play.py which has a number of examples. Try out some of these examples 
by commenting them out.

Options can be passed to main.py: -h (help), -s (silent), -d (double precision), 
-f (saves TACs for view in tensorboard)

Any file saved by PyTAC is in the 'logs/' directory.

Below are the new features of Version 1.22.

Training
========
--Data format has been updated. Please see data.py for a description of the new format.

--The training algorithm has been updated and is now completely independent of 
tensorflow (see train.py). Among the improvements to the algorithm: more refined weight 
initialization, trying different initial weights before starting gradient descent, 
a more refined stopping criteria, and a modular design with a sharper and minimal 
interface to the data manager (data.py) and tensorflow (tacgraph.py).

--The training algorithm can be fine-tuned (customized) through various parameters
that can be found in train.py.

--Training now supports "tied CPTS," which allow CPTs of different TBN nodes to be
identical during training. Simply set the "cpt_tie" field of a TBN to some string. 
All TBN nodes with the same value of "cpt_tie" will have the same learned CPT. 
Tied CPTs must have the same shape and "cpt_tie".

--Training now supports "fixed CPTs," which allow CPTs to be carried into the TAC
and not trained (exploiting background knowledge). Simply set the "fixed_cpt" field 
of a TBN node to True.

--Training now supports "fixed zeros," which allows zero probabilities to be carried
into the TAC and not trained (exploiting background knowledge). Simply set the
"fixed_zeros" field of a TBN node to True (this is more refined than "fixed_cpt"
and should not be set together with "fixed_cpt").

--The notion of "metric" was introduced and can be used for both training (stopping
criteria) and evaluation (to assess the quality of TACs). Three metrics are supported: 
  --cross entropy
  --mean squared error
  --classification accuracy (assumes one-hot labels)
  

Performance
===========
--TACs now employ 'dynamic' tensor shapes, which allows the batch size to be specified
at evaluation/learning time, instead of compile time. This simplified the code 
significantly, is more efficient and is standard practice for neural networks.

--The implementation of factor operations using tensor operations has been significantly
changed, through the employment of the Dims object (dims.py). The biggest changes are
a new implementation of factor multiplication, which relies on "broadcasting." The
second major change is the introduction of a multiply-project operations based on
matrix multiplication (see ops.py and dims.py).


Compilation
===========
--A new inference technique ("decoupling") was introduced for TBNs with functional 
CPTs, allowing the efficient compilation of TBNs with very large treewidth in some
cases (see decouple.py). This technique is applied automatically in two cases:
 --the "fixed_cpt" flag is set and the cpt contains only 0/1 probabilities, or
 --the "functional" flag is set and the tac is trainable

--When compiling a tbn into a TAC, one can now specify whether evidence will be hard.
The compilation algorithm exploits this to produce smaller TACs (could be exponentially
smaller depending on the tbn topology and location of evidence nodes in the tbn). 
Just specify the "hard_inputs" parameter in the constructor tac.TAC() to invoke
this technique.


Specifying TBNs
===============
(see the Node class in node.py for these new features)

--The "values" attribute of a tbn node can now be any python Sequence. This includes 
lists, tuples, strings, and xrange.

--The values of a tbn node can now be arbitrary python objects (e.g., integers, tuples, 
lists, strings, etc). Hence, the "values" attribute is a Sequence of python objects.

--Deterministic CPTs can now be specified by passing a python function that computes 
the value of a tbn node based on the values of its parents. Just pass the function
to the "cpt" parameter of the constructor Node().

--CPTs that encode logical constraints can now be specified by passing a python function 
that returns True/False depending on whether a family instantiation satisfies the
constraints. Just pass the function to the "cpt" parameter of the constructor Node().
The CPT will be uniform for values that satisfy the constraints.

--Impossible values of a tbn node are now pruned automatically.

--The above features allow one to specify complex TBNs very compactly. For examples,
see the rectangles-tbn in 'rectangles/model.py' and the digits-tbn in 'digits/model.py'

--Pruning values can lead to significantly smaller TACs in some cases, as it prunes
values that are guaranteed to always have a zero probability. This will also prune
edges when a node end up having only one possible value.

Impossible Evidence
===================

--Improved handling of tiny and zero probabilities, which cause problems during 
normalization, computing logs for cross entropy and training. This includes:
  --A factor scaling operation which is applied after a certain number of multiplies.
  --Handling zero-probability evidence gracefully by outputting all-zero distributions.
  
--float32 or float64 precision can be selected for tensors using a flag (see precision.py).
One may need to use float64 when evidence has very small probabilities.


Architecture
============

--The only two files that are tensorflow dependent (ops.py and tacgraph.py) were
redesigned to improve modularity and the interface to other parts of the code.

--The data manager was redesigned and expanded (see data.py).

--The jointree compilation algorithm has been refactored (see jointree.py and view.py).

Visualization
=============

--The visualization of TACs using tensorboard has been redesigned to allow hierarchical
viewing (use the "profile" flag when compiling a TBN into a TAC). Each factor operation 
will now appear as a node in tensorboard graph: clicking the node will expand it into
a subgraph, showing the corresponding tensor operations that implement it.

--Added dot visualization for TBNs, Jointrees and views (see .dot() functions in tbn.py,
jointree.py, and view.py)