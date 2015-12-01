###################################################################
# Refer to http://deeplearning.net/software/theano/tutorial       #
###################################################################

import theano.tensor as T
from theano import function
from theano import numpy
from theano import Param

# Function for scalar addition
# In Theano, all symbols must be typed. 
# In particular, T.dscalar is the type we assign to "0-dimensional arrays (scalar) of doubles (d)". 
# It is a Theano Type. dscalar is not a class. Therefore, neither x nor y are actually instances of dscalar.
# They are instances of TensorVariable
print "===== Define Scalar x, y and z=x+y ====="
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
print "Class type of declared variable x: " + str(type(x))
print "x.type: " + str(x.type)
print "T.dscalar: " + str(T.dscalar)
print "x.type is T.dscalar: " + str(x.type is T.dscalar)

# Define and test function scalar_add
scalar_add = function([x, y], z)
scalar_add_one_by_default = function([x, Param(y, default=1)], z)
print "===== Function scalar_add tests ====="
print "scalar_add(2, 3) = " + str(scalar_add(2, 3))
print "scalar_add(16.3, 12.1) = " + str(scalar_add(16.3, 12.1))
print "scalar_add_one_by_default(16.3, 12.1) = " + str(scalar_add_one_by_default(16.3, 12.1))
print "scalar_add_one_by_default(16.3) = " + str(scalar_add_one_by_default(16.3))
print "z.eval({x : 16.3, y : 12.1}) = " + str(z.eval({x : 16.3, y : 12.1}))


# Function for matrix addition
print ""
print "===== Define matrix A, B and Z=A+B ====="
A = T.dmatrix('A')
B = T.dmatrix('B')
Z = A + B
print "Class type of declared variable A: " + str(type(A))
print "A.type: " + str(A.type)
print "T.dmatrix: " + str(T.dmatrix)
print "A.type is T.dmatrix: " + str(A.type is T.dmatrix)

# Define and test function matrix_add
matrix_add = function([A, B], Z)
print "===== Function matrix_add tests ====="
print matrix_add([[1, 2], [3, 4]], [[10, 20], [30, 40]])
# print matrix_add(numpy.array([[1, 2], [3, 4]]), numpy.array([[10, 20], [30, 40]]))


# Logistic function
# Logistic is performed elementwise because all of its operations
# division, addition, exponentiation, and division are themselves elementwise operations.
print ""
print "===== Define matrix X and logistics function S = 1 / (1 + T.exp(-X)) ====="
X = T.dmatrix('X')
S = 1 / (1 + T.exp(-X))

# Define and test function matrix_add
logistics = function([X], S)
print "===== Function logistics tests ====="
input = [[1, 2], [3, 4]]
print "input: "
print input
print "Function logistics output: "
print logistics(input)


# Computing More than one Thing at the Same Time
print ""
print "===== Define matrices a,b and function diffs with three outputs ====="
a, b = T.dmatrices('a', 'b')
diff = a - b
absdiff = abs(diff)
squareddiff = diff ** 2
# Define and test function diffs
diffs = function([a, b], [diff, absdiff, squareddiff])
print "===== Function diffs tests ====="
input1 = [[1, 1], [1, 1]]
print "input1: "
print input1
input2 = [[0, 1], [2, 3]]
print "input2: "
print input2
print "Function diffs output: "
print diffs(input1, input2)


# Function for vector addition
print ""
print "===== Define vectors V1, V2 and function V=V1+V2 ====="
V1 = T.vector('V1')
V2 = T.vector('V2')
V = V1 + V2
print "Class type of declared variable V1: " + str(type(V1))
print "V1.type: " + str(V1.type)
print "T.vector: " + str(T.vector)
print "V1.type is T.vector: " + str(V1.type is T.vector)
# Define and test function vector_add
vector_add = function([V1, V2], V)
print "===== Function vector_add tests ====="
print vector_add([1, 2], [10, 20])


# Using Shared Variables to make a function with an internal state.
from theano import shared
state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state, state+inc)])
print ""
print "===== Function accumulator tests (Using shared variables as internal state) ====="
print "state.get_value(): " + str(state.get_value())
print "accumulator(1): " + str(accumulator(1))
print "state.get_value(): " + str(state.get_value())
print "accumulator(200): " + str(accumulator(200))
print "state.get_value(): " + str(state.get_value())
print "accumulator(1): " + str(accumulator(1))
print "state.get_value(): " + str(state.get_value())

# The type of foo must match the shared variable we are replacing
# with the ``givens``
fn_of_state = state * 2 + inc
foo = T.scalar(dtype=state.dtype)
print "===== Function skip_shared tests (Skip shared variables) ====="
skip_shared = function([inc, foo], fn_of_state, givens=[(state, foo)])
print "skip_shared(1, 3): " + str(skip_shared(1, 3))  # we're using 3 for the state, not state.value
print "state.get_value(): " + str(state.get_value())  # old state still there, but we didn't use it

