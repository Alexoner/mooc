import numpy

# M = numpy.matrix([[0, 0, 1],
                 #[1.0 / 2.0, 0, 0],
                 #[1.0 / 2.0, 1, 0]])
M = numpy.matrix([[0, 0, 0],
                  [1.0 / 2.0, 0, 0],
                  [1.0 / 2.0, 1.0, 1.0]])
e = numpy.matrix([[1.0 / 3.0], [1.0 / 3.0], [1.0 / 3.0]])
v = numpy.matrix([[1.0], [1.0], [1.0]])
print v

for i in range(50):
    v = 0.7 * M * v + (1 - 0.7) * e
    print 'iteration %d' % (i + 1), v
