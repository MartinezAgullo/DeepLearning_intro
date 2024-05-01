import numpy as np

# SoftmaxFunction  // Sigmoid
# P(class i) = exp(Z_i) / Sum_j (Z_j)
# Z_i = LienarFunctionScore

def softmax(L):
    expL = np.exp(L)        # Exp to every elemment of L
    sumExpL = sum(expL)     # Sum of all ellements of expL
    result = []             # List of probabilities for each class
    for i in expL:
        result.append(i*1.0/sumExpL)
    return result


# Tests
# def softmax(L):
#     expL = np.exp(L)
#     return np.divide (expL, expL.sum())

#def softmax(L):
#    P = []
#    sum = 0
#    for Z_i in L:
#        p_i = np.exp(Z_1)
#        P.append(p_1)
#        sum += p_i
#    P = [x / sum for x in P]
#    pass