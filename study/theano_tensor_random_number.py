###################################################################
# Refer to http://deeplearning.net/software/theano/tutorial       #
###################################################################

from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
#srng = RandomStreams(seed=234)
srng = RandomStreams()
#rv_u = srng.uniform((2,2))
rv_u = srng.uniform()
#rv_n = srng.normal((2,2))
rv_n = srng.normal()
uniform_random = function([], rv_u)
normal_random = function([], rv_n)
uniform_random_no_updates = function([], rv_u, no_default_updates=True)
normal_random_no_updates = function([], rv_n, no_default_updates=True)
nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)
print "uniform_random(): " + str(uniform_random())
print "uniform_random(): " + str(uniform_random())
print "uniform_random(): " + str(uniform_random())
print "uniform_random(): " + str(uniform_random())
print "uniform_random(): " + str(uniform_random())
print "uniform_random_no_updates(): " + str(uniform_random_no_updates())
print "uniform_random_no_updates(): " + str(uniform_random_no_updates())
print "uniform_random_no_updates(): " + str(uniform_random_no_updates())
print "uniform_random_no_updates(): " + str(uniform_random_no_updates())
print "uniform_random_no_updates(): " + str(uniform_random_no_updates())
print "normal_random(): " + str(normal_random())
print "normal_random(): " + str(normal_random())
print "normal_random(): " + str(normal_random())
print "normal_random(): " + str(normal_random())
print "normal_random(): " + str(normal_random())
print "normal_random_no_updates(): " + str(normal_random_no_updates())
print "normal_random_no_updates(): " + str(normal_random_no_updates())
print "normal_random_no_updates(): " + str(normal_random_no_updates())
print "normal_random_no_updates(): " + str(normal_random_no_updates())
print "normal_random_no_updates(): " + str(normal_random_no_updates())
print "nearly_zeros(): " + str(nearly_zeros())
print "nearly_zeros(): " + str(nearly_zeros())
print "nearly_zeros(): " + str(nearly_zeros())
print "nearly_zeros(): " + str(nearly_zeros())
print "nearly_zeros(): " + str(nearly_zeros())
print "nearly_zeros(): " + str(nearly_zeros())
print "nearly_zeros(): " + str(nearly_zeros())
print "nearly_zeros(): " + str(nearly_zeros())
print "nearly_zeros(): " + str(nearly_zeros())
print "nearly_zeros(): " + str(nearly_zeros())
print "nearly_zeros(): " + str(nearly_zeros())
print "nearly_zeros(): " + str(nearly_zeros())
print "nearly_zeros(): " + str(nearly_zeros())
print "nearly_zeros(): " + str(nearly_zeros())
print "nearly_zeros(): " + str(nearly_zeros())

rng_val = rv_u.rng.get_value(borrow=True)   # Get the rng for rv_u
print "srng: " + str(srng)
print "rv_u: " + str(rv_u)
print "rv_u.rng: " + str(rv_u.rng)
print "rng_val: " + str(rng_val)
rng_val.seed(89234)                         # seeds the generator
rv_u.rng.set_value(rng_val, borrow=True)    # Assign back seeded rng


