###################################################################
# Refer to http://deeplearning.net/software/theano/tutorial       #
###################################################################

# from theano import numpy
import numpy

# Various ways to create a ndarray
#a = numpy.asarray([[1., 2], [3, 4], [5, 6]])
#a = numpy.array([[1.,2], [3,4], [5,6]])
#a = numpy.array([[1.,2], [3,4], [5,6]], dtype=complex)
#a = numpy.arange(6).reshape(3,2) # from 0 to 6
#a = numpy.arange(start=1, stop=7).reshape(3,2)
#a = numpy.zeros(6).reshape(3,2)
#a = numpy.zeros((3,2))
#a = numpy.ones(6).reshape(3,2)
#a = numpy.linspace(1, 6, 6).reshape(3,2)
#a = numpy.random.random(6).reshape(3,2)
def f(x,y):
    return 10*x+y
a = numpy.fromfunction(f, (3,2))
print "Demo1: ndarray creation"
print "       A 3 by 2 array represents three examples(vectors) with dimension 2" 
print "a = \n" + str(a)
print "type of a: " + str(type(a))
print "dtype of a: " + str(a.dtype)
print "shape of a: " + str(a.shape)


# Broadcasting
# The smaller array (or scalar) is broadcasted across the larger array so that they have compatible shapes
a = numpy.asarray([1.0, 2.0, 3.0])
b = 2.0
print ""
print "Demo2: Broadcasting of smaller array (or scalar) b: " + str(b) 
print "       across the larger array a: " + str(a)
print "a*b = " + str(a*b)
print "b*a = " + str(b*a)

# Indexing, slicing and iterating
a = numpy.arange(12)**3
print ""
print "Demo3: Indexing, slicing and iterating"
print "a = " + str(a)
print "a[2] = " + str(a[2])
print "a[2:5] = " + str(a[2:5])
print "a[0:6:2] = " + str(a[0:6:2]) # from position 0 to 6, every 2nd element
print "a[:6:2] = " + str(a[:6:2]) # Same as a[:6:2]

# Multidimensional arrays can have one index per axis. 
# These indices are given in a tuple separated by commas:
print ""
print "# Now let A = a.reshape(4,3)"
A = a.reshape(4,3)
print "A = \n" + str(A)
print "A[2,1] = " + str(A[2,1]) + "  (row 2 and column 1)"
print "A[0:2,1] = " + str(A[0:2,1]) + "  (row 0 and 1; column 1)" # row 0,1; column 1
print "A[:,1] = " + str(A[:,1]) + "  (column 1)" # All rows, column 1
print "A[0,:] = " + str(A[0,:]) + "  (row 0)" # All columns, row 0
print "A[1,:] = " + str(A[1,:]) + "  (row 1)" # All columns, row 1
print "A[0] = " + str(A[0]) + "  (row 0)" # All columns, row 0
print "A[1] = " + str(A[1]) + "  (row 1)" # All columns, row 1
print "A[-1] = " + str(A[-1]) + "  (The last row)" # All columns, the last row
print "# Iteration over A with respect to the first axis(ie. rows for 2-D array):"
for row in A:
    print "Row: " + str(row)

print ""
print "# Now let A3D = a.reshape(2,2,3)"
A3D = a.reshape(2,2,3)
print "A3D = \n" + str(A3D)
print "# Iteration over A3D with respect to the first axis:"
for array in A3D:
    print "Array: \n" + str(array)

print ""
print "Flattening of A3D array:"
print "A3D.ravel() = " + str(A3D.ravel())




