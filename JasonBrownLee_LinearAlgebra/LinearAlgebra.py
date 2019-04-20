import numpy as np
import pandas as pn

# 4 Introduction to NumPy Arrays
# create array
from numpy import array
# create array
l = [1.0, 2.0, 3.0]
a = array(l)
# display array
print(a)
# display array shape
print(a.shape)
# display array data type
print(a.dtype)

# create empty array
from numpy import empty
a = empty([3,3])
print(a)

# create zero array
from numpy import zeros
a = zeros([3,5])
print(a)

# create one array
from numpy import ones
a = ones([5])
print(a)

# create array with vstack
from numpy import array
from numpy import vstack
# create first array
a1 = array([1,2,3])
print(a1)
# create second array
a2 = array([4,5,6])
print(a2)
# vertical stack
a3 = vstack((a1, a2))
print(a3)
print(a3.shape)

# create array with hstack
from numpy import array
from numpy import hstack
# create first array
a1 = array([1,2,3])
print(a1)
# create second array
a2 = array([4,5,6])
print(a2)
# create horizontal stack
a3 = hstack((a1, a2))
print(a3)
print(a3.shape)


# 5 Index, Slice and Reshape NumPy Arrays
# create one-dimensional array
from numpy import array
# list of data
data = [11, 22, 33, 44, 55]
# array of data
data = array(data)
print(data)
print(type(data))
# create two-dimensional array
from numpy import array
# list of data
data = [[11, 22],
[33, 44],
[55, 66]]
# array of data
data = array(data)
print(data)
print(type(data))
# index a one-dimensional array
from numpy import array
# define array
data = array([11, 22, 33, 44, 55])
# index data
print(data[0])
print(data[4])
# index array out of bounds
from numpy import array
# define array
data = array([11, 22, 33, 44, 55])
# index data
print(data[5])
# negative array indexing
from numpy import array
# define array
data = array([11, 22, 33, 44, 55])
# index data
print(data[-1])
print(data[-5])
# index two-dimensional array
from numpy import array
# define array
data = array([
[11, 22],
[33, 44],
[55, 66]])
# index data
print(data[0,0])
# index row of two-dimensional array
from numpy import array
# define array
data = array([
[11, 22],
[33, 44],
[55, 66]])
# index data
print(data[0,])
# slice a one-dimensional array
from numpy import array
# define array
data = array([11, 22, 33, 44, 55])
print(data[:])
# slice a subset of a one-dimensional array
from numpy import array
# define array
data = array([11, 22, 33, 44, 55])
print(data[0:1])
# negative slicing of a one-dimensional array
from numpy import array
# define array
data = array([11, 22, 33, 44, 55])
print(data[-2:])
# split input and output data
from numpy import array
# define array
data = array([
[11, 22, 33],
[44, 55, 66],
[77, 88, 99]])
# separate data
X, y = data[:, :-1], data[:, -1]
print(X)
print(y)
# split train and test data
from numpy import array
# define array
data = array([
[11, 22, 33],
[44, 55, 66],
[77, 88, 99]])
# separate data
split = 2
train,test = data[:split,:],data[split:,:]
print(train)
print(test)

# shape of one-dimensional array
from numpy import array
# define array
data = array([11, 22, 33, 44, 55])
print(data.shape)

# shape of a two-dimensional array
from numpy import array
# list of data
data = [[11, 22],
[33, 44],
[55, 66]]
# array of data
data = array(data)
print(data.shape)

# row and column shape of two-dimensional array
from numpy import array
# list of data
data = [[11, 22],
[33, 44],
[55, 66]]
# array of data
data = array(data)
print('Rows: %d' % data.shape[0])
print('Cols: %d' % data.shape[1])

# reshape 1D array to 2D
from numpy import array
# define array
data = array([11, 22, 33, 44, 55])
print(data.shape)
# reshape
data = data.reshape((data.shape[0], 1))
print(data.shape)

# reshape 2D array to 3D
from numpy import array
# list of data
data = [[11, 22],
[33, 44],
[55, 66]]
# array of data
data = array(data)
print(data.shape)
# reshape
data = data.reshape((data.shape[0], data.shape[1], 1))
print(data.shape)


# 6 NumPy Array Broadcasting
# broadcast scalar to one-dimensional array
from numpy import array
# define array
a = array([1, 2, 3])
print(a)
# define scalar
b = 2
print(b)
# broadcast
c = a + b
print(c)

# broadcast scalar to two-dimensional array
from numpy import array
# define array
A = array([[1, 2, 3],[1, 2, 3]])
print(A)
# define scalar
b = 2
print(b)
# broadcast
C = A + b
print(C)

# broadcast one-dimensional array to two-dimensional array
from numpy import array
# define two-dimensional array
A = array([[1, 2, 3],[1, 2, 3]])
print(A)
# define one-dimensional array
b = array([1, 2, 3])
print(b)
# broadcast
C = A + b
print(C)

# broadcasting error
from numpy import array
# define two-dimensional array
A = array([[1, 2, 3],[1, 2, 3]])
print(A.shape)
# define one-dimensional array
b = array([1, 2])
print(b.shape)
# attempt broadcast
C = A + b
print(C)

# 7 Vectors and Vector Arithmetic
# create a vector
from numpy import array
# define vector
v = array([1, 2, 3])
print(v)

# vector addition
from numpy import array
# define first vector
a = array([1, 2, 3])
print(a)
# define second vector
b = array([1, 2, 3])
print(b)
# add vectors
c = a + b
print(c)

# vector subtraction
from numpy import array
# define first vector
a = array([1, 2, 3])
print(a)
# define second vector
b = array([0.5, 0.5, 0.5])
print(b)
# subtract vectors
c = a - b
print(c)

# vector multiplication
from numpy import array
# define first vector
a = array([1, 2, 3])
print(a)
# define second vector
b = array([1, 2, 3])
print(b)
# multiply vectors
c = a * b
print(c)

# vector division
from numpy import array
# define first vector
a = array([1, 2, 3])
print(a)
# define second vector
b = array([1, 2, 3])
print(b)
# divide vectors
c = a / b
print(c)

# vector dot product
from numpy import array
# define first vector
a = array([1, 2, 3])
print(a)
# define second vector
b = array([1, 2, 3])
print(b)
# multiply vectors
c = a.dot(b)
print(c)

# vector-scalar multiplication
from numpy import array
# define vector
a = array([1, 2, 3])
print(a)
# define scalar
s = 0.5
print(s)
# multiplication
c = s * a
print(c)

# 8  Vector Norms
# vector L1 norm
from numpy import array
from numpy.linalg import norm
# define vector
a = array([1, 2, 3])
print(a)
# calculate norm
l1 = norm(a, 1)
print(l1)

# vector L2 norm
from numpy import array
from numpy.linalg import norm
# define vector
a = array([1, 2, 3])
print(a)
# calculate norm
l2 = norm(a)
print(l2)

# vector max norm
from math import inf
from numpy import array
from numpy.linalg import norm
# define vector
a = array([1, 2, 3])
print(a)
# calculate norm
maxnorm = norm(a, inf)
print(maxnorm)

