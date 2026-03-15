import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    N_batch = x.shape[0]
    C_out = W.shape[0]
    n = x.shape[2]
    m = W.shape[2]
    X_news = np.zeros((N_batch,C_out,n-m+1,n-m+1))
    for z in range(N_batch):
        for k in range(C_out): 
            for i in range(n - m + 1):
                for j in range(n - m + 1):
                    X_news[z,k,i,j] = np.sum(x[z,:,i:i+m,j:j+m] * W[k]) + b[k]
    return X_news