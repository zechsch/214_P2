#!/usr/bin/python
#recommender.py
#Matrix factorization for an example recommender system.
#Zechariah Schneider (zechsch@umich.edu)
#Jonathan Lipworth (lipworth@umich.edu)
#Kevin Rabideau (kevrab@umich.edu)
#Samuel Poznak (poznak@umich.edu)
#Created: 4.3.16
#Last Modified: 4.5.16

#Reads in a file containing a matrix of M ratings from N users
#and factors it such that the dot product of the two resulting
#matrices fill in 0's in the original matrix as recommendations

#Takes inputs steps, alpha, beta.
#steps: the maximum number of optimization cycles
#alpha: a *very* small constant to represent the learning rate
#beta: the regularization constant

try:
    import numpy, sys, csv
except:
    print "Error importing module."
    print "Required: numpy, sys, csv"
    exit(1)

def matrix_factorization(R, P, Q, K, steps, alpha, beta):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T

def getInput():
    try:
        steps = input("Number of steps,\nnothing for default: ")
    except:
        steps = 5000
    try:
        alpha = input("Learning rate,\nnothing for default: ")
    except:
        alpha=0.0002
    try:
        beta = input("Regularization,\nnothing for default: ")
    except:
        beta=0.02

    return steps, alpha, beta

def getMatrix(name):
    R = []
    try:
        with open(name, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    R.append(map(int, line.split(',')))
    except:
        print "Error reading file: \"" + sys.argv[1] + "\", exiting."
        exit(1)
    return R


if __name__ == "__main__":

    #Rows are users, columns are ratings
    if len(sys.argv) != 2:
        print "Error. Wrong number of arguments."
        print "Usage: python recommender.py INPUT_FILE"
        exit(1)

    R = getMatrix(sys.argv[1])

    steps, alpha, beta = getInput()

    R = numpy.array(R)

    N = len(R)
    M = len(R[0])
    K = 2

    P = numpy.random.rand(N,K)
    Q = numpy.random.rand(M,K)

    pResult, qResult = matrix_factorization(R, P, Q, K, steps, alpha, beta)
    recommenationMatrix = numpy.dot(pResult, qResult.T)

    with open('output.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(recommenationMatrix)
