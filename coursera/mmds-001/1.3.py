import numpy

M = numpy.matrix([[0, 0, 1],
                 [1.0 / 2.0, 0, 0],
                 [1.0 / 2.0, 1, 0]])
e = numpy.matrix([[1], [1], [1]])
v = numpy.matrix([[1.0], [1.0], [1.0]])
print v

for i in range(50):
    v = M * v
    print 'iteration %d' % (i + 1), v

print v
