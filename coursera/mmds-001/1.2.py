import numpy

M = numpy.matrix([[0, 0, 1],
                 [1.0 / 2.0, 0, 0],
                 [1.0 / 2.0, 1, 0]])
e = numpy.matrix([[1.0 / 3.0], [1.0 / 3.0], [1.0 / 3.0]])
# v = numpy.matrix([[1.0 / 3.0], [1.0 / 3.0], [1.0 / 3.0]])
v = numpy.matrix([[1.0], [1.0], [1.0]])
print v

for i in range(50):
    v = 0.85 * M * v + (1 - 0.85) * e

print v
print '\n'
print(v[1, 0] - 0.475 * v[0, 0] - 0.05 * v[2, 0])
print(0.85 * v[2, 0] - v[1, 0] - 0.575 * v[0, 0])
print(0.95 * v[1, 0] - 0.475 * v[0, 0] - 0.05 * v[2, 0])
print(v[2, 0] - 0.9 * v[1, 0] - 0.475 * v[0, 0])
