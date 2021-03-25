import numpy as np

class num_kernel():
    def __init__(self, k_type='gaussian', alpha=1):
        self.kernel_type = k_type
        self.alpha = alpha

    def kernel(self,x1,x2):
        if self.kernel_type == 'gaussian':
            return np.exp(-(1 / (2 * self.alpha)) * np.linalg.norm(x1 - x2) ** 2)


    def gram_matrix(self,X):
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                x1 = X[i, :]
                x2 = X[j, :]
                val = self.kernel(x1, x2,)
                K[i, j] = val
                K[j, i] = val
        return K

    def test_gram_matrix(self,X,X_test):
        n = len(X)
        p = len(X_test)
        K = np.zeros((p, n))
        for i in range(p):
            for j in range(n):
                x_tst = X_test[i, :]
                x_tr = X[j, :]
                val = self.kernel(x_tr, x_tst)
                K[i, j] = val
        return K