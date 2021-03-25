from mismatch_utils import mismatch_matrix
import os.path
from scipy import sparse
import pandas as pd
import cvxpy as cp
import numpy as np


class MismatchSVM():

    def __init__(self, dataset_id, k_l, m_l):
        """ Takes as argument a dataset id,  a list of k and the corresponding m
        Performs the sum of the given kernels """

        assert (dataset_id in [0, 1, 2])
        assert (len(k_l) == len(m_l))
        self.dataset_id = dataset_id
        self.k_l = k_l
        self.m_l = m_l

        self.X = None
        self.Y = None
        self.create_data()

        self.mismatch_matrice = None
        self.res = None
        self.training_acc = None

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

    def compute_mismatch_matrice_multi(self, norm=False, if_weight=False, weight=None):
        """ For given kernel lists, computes/loads  sparse feauture matrices
        and concatenates them with possible weights and normalization"""

        nb_k = len(self.k_l)
        mism_matr_l = []
        for i in range(nb_k):
            k = self.k_l[i]
            m = self.m_l[i]
            path = 'mismatch_matrices/data{}_k{}_m{}.npz'.format(self.dataset_id, k, m)
            if os.path.isfile(path):
                # If matrix already exists, load it
                print("Loading mismatch matrix")
                mism = sparse.load_npz(path)
                if if_weight:
                    mism = np.sqrt(weight[i]) * mism
                mism_matr_l.append(mism)
            else:
                # If matrice doesn't exist, computes and stores it
                print("Computing mismatch matrix")
                mism = mismatch_matrix(self.X, k, m)
                sparse.save_npz(path, mism)
                if if_weight:
                    mism = np.sqrt(weight[i]) * mism
                mism_matr_l.append(mism)

        # Concatenation of the feature matrices to create the global feature matrix
        self.mismatch_matrice = sparse.hstack(mism_matr_l, format="csr")
        if norm:
            # Normalization
            self.mismatch_matrice = sparse.csr_matrix(
                self.mismatch_matrice / (sparse.linalg.norm(self.mismatch_matrice, axis=1)[:, None]))

    def svm_accuracy(self, mismatch_train, mismatch_val, y, y_val, lmbda):
        """ For given concatenated feature matrices and a given lambda,
        solves the SVM dual problem and computes accuracies """

        # Construct Gram matrices
        K = mismatch_train @ mismatch_train.T
        K_val = mismatch_val @ mismatch_train.T

        # Construct the problem.
        n = mismatch_train.shape[0]
        C = 1 / (2 * lmbda * n)
        x = cp.Variable(n)
        objective = cp.Minimize((1 / 2) * cp.quad_form(x, K) - x.T @ y)
        constraints = [0 <= cp.multiply(y, x), cp.multiply(y, x) <= C]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        res = x.value

        # Compute train accuracy
        f = K @ res
        pred = np.where(f >= 0, 1, -1)
        score = pred * y
        acc_train = score[score == 1].sum() / len(score.T)

        # Compute val accuracy
        f_val = K_val @ res
        pred_val = np.where(f_val >= 0, 1, -1)
        score_val = pred_val * y_val
        acc_val = score_val[score_val == 1].sum() / len(score_val.T)

        return acc_train, acc_val

    def lmbda_Nfold(self, N_fold, lmbda_l):
        """ Function that helps tunning the hyperparameters using cross-validation
        Takes a list of lambdas to be tested and the number of Folds
        Outputs training and validation accuracies for every Fold and lambda """
        res_tr = pd.DataFrame(index=lmbda_l, columns=range(N_fold))
        res_val = pd.DataFrame(index=lmbda_l, columns=range(N_fold))

        tot_card = self.mismatch_matrice.shape[0]
        assert (tot_card % N_fold == 0)
        val_card = tot_card // N_fold

        for i in range(N_fold):
            # Separating the training and validation sets
            m_train, m_val = sparse.vstack(
                [self.mismatch_matrice[: val_card * i, :], self.mismatch_matrice[val_card * (i + 1):, :]]), \
                             self.mismatch_matrice[val_card * i: val_card * (i + 1), :]

            Y_train, Y_val = np.hstack((self.Y[: val_card * i], self.Y[val_card * (i + 1):])).squeeze(), \
                             self.Y[val_card * i: val_card * (i + 1)]

            # Test multiple lambdas
            for lmbda in lmbda_l:
                acc_tr, acc_val = self.svm_accuracy(m_train, m_val, Y_train, Y_val, lmbda)
                res_tr.loc[lmbda, i] = acc_tr
                res_val.loc[lmbda, i] = acc_val
            print("Fold {}".format(i))
        return res_val, res_tr

    def fit(self, lmbda):

        """ Function that trains an SVM classifier on the whole dataset with a given lambda
        and saves the dual coefficients """

        # Construct the global Gram matrix
        K = self.mismatch_matrice @ self.mismatch_matrice.T
        n = self.mismatch_matrice.shape[0]
        y = self.Y

        # Training the classifier on the whole dataset
        C = 1 / (2 * lmbda * n)
        x = cp.Variable(n)
        objective = cp.Minimize((1 / 2) * cp.quad_form(x, K) - x.T @ y)
        constraints = [0 <= cp.multiply(y, x), cp.multiply(y, x) <= C]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        res = x.value

        # Computing training accuracy and saving dual coefs
        f = K @ res
        pred = np.where(f >= 0, 1, -1)
        score = pred * self.Y
        acc_train = score[score == 1].sum() / len(score.T)
        self.res = res
        self.training_acc = acc_train

    def predict_multi(self, X_test):
        """ Function that predicts labels of the test set, after the model has been trained """

        # Checking that the model has been trained
        assert (self.training_acc != None)

        nb_k = len(self.k_l)
        mism_matr_l = []

        # Calculating feauture matrices of the test set and concatenating them
        for i in range(nb_k):
            k = self.k_l[i]
            m = self.m_l[i]
            mism = mismatch_matrix(X_test, k, m)
            mism_matr_l.append(mism)
        mism_test = sparse.hstack(mism_matr_l, format="csr")

        # Calculating Gram matrix of the test set
        K_test = mism_test @ self.mismatch_matrice.T

        # Make a prediction
        f = K_test @ self.res
        pred = np.where(f >= 0, 1, -1)
        return pred


if __name__ == '__main__':
    # Example of N_Fold tuning
    mism = MismatchSVM(dataset_id=2, k_l=[8, 5], m_l=[2, 1])
    mism.compute_mismatch_matrice_multi(norm=False)
    lmbda_l = [1 + 0.5 * k for k in range(-1, 2)]
    N_fold = 5
    res_val, res_tr = mism.lmbda_Nfold(N_fold, lmbda_l)
    print(res_tr)
    print(res_val)
    print(res_val.mean(axis=1))
