###################################################################
# Refer to http://deeplearning.net/software/theano/tutorial       #
###################################################################

# from theano import numpy
import numpy

# Matrix conventions for machine learning
examples1 = numpy.asarray([[1., 2], [3, 4], [5, 6]])
examples1_shape = numpy.asarray([[1., 2], [3, 4], [5, 6]]).shape
print "A 3 by 2 array represents three examples(vectors) with dimension 2" 

# Broadcasting
# The smaller array (or scalar) is broadcasted across the larger array so that they have compatible shapes
a = numpy.asarray([1.0, 2.0, 3.0])
b = 2.0
print a * b
print b * a