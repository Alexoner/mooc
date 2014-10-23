M = matrix(c(0,1/2,1/2,0,0,1,0,0,1),ncol = 3)
e = matrix(c(1,1,1),ncol = 1)
v1 = matrix(c(1,1,1),ncol = 1)
v1 = v1 / 3
n = 3
for (i in 1:5) {
    v1 = ((0.7 * M )  %*% v1) + (((1-0.7)*e)/n)
}

v1 = v1 * 3
