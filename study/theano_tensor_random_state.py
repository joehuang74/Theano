###################################################################
# Refer to http://deeplearning.net/software/theano/tutorial       #
###################################################################

from __future__ import print_function
import theano
import numpy
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams

class Graph():
     def __init__(self, seed=123):
         self.rng = RandomStreams(seed)
         self.y = self.rng.uniform(size=(1,))
         
g1 = Graph(seed=123)
random1 = theano.function([], g1.y)
g2 = Graph(seed=987)
random2 = theano.function([], g2.y)
# By default, the two functions are out of sync.
print("=== Two different random number generators ===")
print("random1(): " + str(random1()))
print("random1(): " + str(random1()))
print("random2(): " + str(random2()))
print("random2(): " + str(random2()))


def copy_random_state(g1, g2):
     if isinstance(g1.rng, MRG_RandomStreams):
         g2.rng.rstate = g1.rng.rstate
     for (su1, su2) in zip(g1.rng.state_updates, g2.rng.state_updates):
         su2[0].set_value(su1[0].get_value())

# We now copy the state of the theano random number generators.
copy_random_state(g1, g2)
print("=== Copy the random state from one graph to the other ===")
print("random1() and random2() have the same random number sequence now")
print("random1(): " + str(random1()))
print("random1(): " + str(random1()))
print("random2(): " + str(random2()))
print("random2(): " + str(random2()))




