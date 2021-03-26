from mismatch_utils import mismatch_matrix
import os.path
from scipy import sparse
import pandas as pd
import cvxpy as cp
import numpy as np


class MKL_SVM():

    def __init__(self, dataset_id, k_l, m_l, max_iter=10):
        """ Takes as argument a dataset id,  a list of k and the corresponding m
        Performs the multiple kernel learning  """

        assert (dataset_id in [0, 1, 2])
        assert (len(k_l) == len(m_l))

        self.k_l = k_l
        self.m_l = m_l
        self.dataset_id = dataset_id
        self.max_iter = max_iter
        self.X = None
        self.Y = None

        self.create_data()
        self.res = None
        self.training_acc = None
        self.eta = None
        self.mism_l = None
        self.acc_val = None

    def create_data(self):
        """ Loads the given dataset and preprocesses it """

        if self.dataset_id == 0:
            X = pd.read_csv('data/Xtr0.csv', index_col=0)
            Y = pd.read_csv('data/Ytr0.csv', index_col=0)

        elif self.dataset_id == 1:
            X = pd.read_csv('data/Xtr1.csv', index_col=0)
            Y = pd.read_csv('data/Ytr1.csv', index_col=0)

        else:
            X = pd.read_csv('data/Xtr2.csv', index_col=0)
            Y = pd.read_csv('data/Ytr2.csv', index_col=0)

        Y.loc[Y.Bound == 0, 'Bound'] = -1
        self.X = X
        self.Y = Y.values.squeeze()

    def compute_mismatch_matrice_multi(self, norm=False):
        """ For given kernel lists, computes/loads  sparse feauture matrices
         with possible normalization and stores them in mism_l"""

        if not os.path.exists('mismatch_matrices'):
            os.makedirs('mismatch_matrices')

        nb_k = len(self.k_l)
        mism_l = []
        for i in range(nb_k):
            k = self.k_l[i]
            m = self.m_l[i]
            path = 'mismatch_matrices/data{}_k{}_m{}.npz'.format(self.dataset_id, k, m)
            if os.path.isfile(path):
                # If matrix already exists, load it
                print("Loading mismatch matrix")
                mism = sparse.load_npz(path)
                if norm:
                    mism = sparse.csr_matrix(
                        mism / (sparse.linalg.norm(mism, axis=1)[:, None]))

                mism_l.append(mism)
            else:
                # If matrice doesn't exist, computes and stores it
                print("Computing mismatch matrix")
                mism = mismatch_matrix(self.X, k, m)
                sparse.save_npz(path, mism)
                mism = sparse.load_npz(path)
                if norm:
                    # Normalization
                    mism = sparse.csr_matrix(
                        mism / (sparse.linalg.norm(mism, axis=1)[:, None]))

                mism_l.append(mism)

        self.mism_l = mism_l

    def svm_accuracy(self, mism_train=None, mism_val=None, y_train=None, y_val=None, lmbda=3):
        """ For given list of feature matrices and a given lambda,
        performs th iterative algorithm of MKL SVM described in  [1]
        updates the final weights eta as well ass the dual coefs res """

        self.eta = (1 / len(self.k_l)) * np.ones((len(self.k_l)))
        self.acc_val = 0

        kernel_train_l = [m @ m.T for m in mism_train]
        kernel_val_l = [m_val @ m.T for (m_val, m) in zip(mism_val, mism_train)]

        w = np.zeros(len(self.k_l))

        # Iterative algorithm that performs SVM and learns eta at each iteration
        for it in range(self.max_iter):

            # Construct Gram matrices using weights eta
            K = np.sum([eta * k for eta, k in zip(self.eta, kernel_train_l)])

            # Construct the problem.
            n = K.shape[0]
            C = 1 / (2 * lmbda * n)
            x = cp.Variable(n)
            objective = cp.Minimize((1 / 2) * cp.quad_form(x, K) - x.T @ y_train)
            constraints = [0 <= cp.multiply(y_train, x), cp.multiply(y_train, x) <= C]
            prob = cp.Problem(objective, constraints)
            prob.solve()
            res = x.value
            self.res = res

            for i, (eta, k) in enumerate(zip(self.eta, kernel_train_l)):
                w[i] = (eta ** 2) * res @ (k @ res).T
            w = np.sqrt(w)
            etas = w / np.sum(w)
            etas[etas < 1e-7 * etas.max()] = 0
            self.eta = etas

            # Compute train accuracy
            f = K @ res
            pred = np.where(f >= 0, 1, -1)
            score = pred * y_train
            acc_train = score[score == 1].sum() / len(score.T)

            # Compute val accuracy
            K_val = np.sum([eta * k for eta, k in zip(self.eta, kernel_val_l)])
            f = K_val @ res
            pred = np.where(f >= 0, 1, -1)
            score = pred * y_val
            self.acc_val = score[score == 1].sum() / len(score.T)

        return self.acc_val, acc_train, self.eta

    def lmbda_Nfold(self, N_fold, lmbda_l, norm=False):
        """ Function that helps tune the hyperparameters using cross-validation
        Takes a list of lambdas to be tested and the number of Folds
        Outputs training and validation accuracies as well as weights eta  for every Fold and lambda """
        res_tr = pd.DataFrame(index=lmbda_l, columns=range(N_fold))
        res_val = pd.DataFrame(index=lmbda_l, columns=range(N_fold))
        res_eta = pd.DataFrame(index=lmbda_l, columns=range(N_fold))

        tot_card = (self.X).shape[0]
        assert (tot_card % N_fold == 0)
        val_card = tot_card // N_fold

        #Computing the list of mismatch matrices
        self.compute_mismatch_matrice_multi(norm=norm)

        for i in range(N_fold):
            # Separating the training and validation sets
            m_train, m_val = [sparse.vstack([m[: val_card * i, :], m[val_card * (i + 1):, :]]) for m in self.mism_l], \
                             [m[val_card * i: val_card * (i + 1), :] for m in self.mism_l]

            Y_train, Y_val = np.hstack((self.Y[: val_card * i], self.Y[val_card * (i + 1):])).squeeze(), \
                             self.Y[val_card * i: val_card * (i + 1)]

            for lmbda in lmbda_l:
                # Test multiple lambdas
                acc_val, acc_tr, eta = self.svm_accuracy(m_train, m_val, Y_train, Y_val, lmbda)
                res_tr.loc[lmbda, i] = acc_tr
                res_val.loc[lmbda, i] = acc_val
                res_eta.loc[lmbda, i] = eta
            print("Fold {}".format(i))
        return res_val, res_tr, res_eta

    def fit(self, lmbda):
        """ Function that trains an MKL SVM classifier on the whole dataset with a given lambda
        and saves the dual coefficients """

        y_train = self.Y
        self.eta = (1 / len(self.k_l)) * np.ones((len(self.k_l)))
        self.acc_val = 0

        mism_train = self.mism_l
        kernel_train_l = [m @ m.T for m in mism_train]

        w = np.zeros(len(self.k_l))

        for it in range(self.max_iter):

            K = np.sum([eta * k for eta, k in zip(self.eta, kernel_train_l)])

            # Training the classifier on the whole dataset
            n = K.shape[0]
            C = 1 / (2 * lmbda * n)
            x = cp.Variable(n)
            objective = cp.Minimize((1 / 2) * cp.quad_form(x, K) - x.T @ y_train)
            constraints = [0 <= cp.multiply(y_train, x), cp.multiply(y_train, x) <= C]
            prob = cp.Problem(objective, constraints)
            prob.solve()
            res = x.value
            self.res = res

            for i, (eta, k) in enumerate(zip(self.eta, kernel_train_l)):
                w[i] = (eta ** 2) * res @ (k @ res).T
            w = np.sqrt(w)
            etas = w / np.sum(w)
            etas[etas < 1e-7 * etas.max()] = 0
            self.eta = etas

            # Computing training accuracy and saving dual coefs
            f = K @ res
            pred = np.where(f >= 0, 1, -1)
            score = pred * y_train
            self.training_acc = score[score == 1].sum() / len(score.T)


    def predict_multi(self, X_test):
        # Checking that the model has been trained
        assert (self.training_acc != None)
        nb_k = len(self.k_l)
        mism_matr_l = []

        # Calculating feauture matrices of the test set
        for i in range(nb_k):
            k = self.k_l[i]
            m = self.m_l[i]
            mism = mismatch_matrix(X_test, k, m)
            mism_matr_l.append(mism)

        # Calculating Gram matrix of the test set using the learned weights eta
        kernel_test_l = [m_test @ m.T for (m_test, m) in zip(mism_matr_l, self.mism_l)]
        K_test = np.sum([eta * k for eta, k in zip(self.eta, kernel_test_l)])

        # Make a prediction
        f = K_test @ self.res
        pred = np.where(f >= 0, 1, -1)

        return pred


if __name__ == '__main__':
    mism = MKL_SVM(dataset_id=0, k_l=[8, 8, 7, 7, 7, 6, 6, 6, 5, 5, 5], m_l=[1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, ],
                   max_iter=5)
    lmbda_l = [0.1, 0.3]
    res_val, res_tr, res_eta = mism.lmbda_Nfold(N_fold=5, lmbda_l=lmbda_l, norm=False)
    print(res_tr)
    print(res_val)
    print(res_val.mean(axis=1))
