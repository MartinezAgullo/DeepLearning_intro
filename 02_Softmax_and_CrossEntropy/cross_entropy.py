import numpy as np

# Task: Write a function that takes as input two lists Y, P,
#       and returns the float corresponding to their cross-entropy.

# Initial ideal
def cross_entropy(Y, P):
    CE = 0 # Cross Entropy
    i = 0  # Run over the list
    while i < len(Y):
        tmp = Y[i]*np.log(P[i])+(1-Y[i])*np.log(1-P[i])
        CE = CE - tmp
        i = i+1
    return CE

# More efficient approach
def cross_entropy(Y, P):
    Y = np.float_(Y)  # converts all elements in the list into floats
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))