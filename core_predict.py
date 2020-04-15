import numpy as np

def sigmoid(z):
    """
    Sigmoid activation function
    
    :param z:  Accepts the z value that is calculated through linear equation. z could be a
               matrix of real number, or could be a single real number.
    
    :returns:  Matrix/Real number (based on input), where the values have been aplied through
               sigmoid function. Range: 0 < g(z) < 1.
    """
    return 1/(1+np.exp(-z))

def predict(X, params):
    L = int(len(list(params.keys()))/2)
    cache = {"A0": X}
    for i in range(L):
        cache["Z"+str(i+1)] = np.dot(params["W"+str(i+1)], cache["A"+str(i)]) + params["b"+str(i+1)]
        cache["A"+str(i+1)] = sigmoid(cache["Z"+str(i+1)])

    return np.where(cache["A"+str(L)]==np.max(cache["A"+str(L)], axis=0),1,0)