from mismatch_utils import mismatch_matrix
import os.path
import scipy as sp
from scipy import sparse
import pandas as pd
import cvxpy as cp
import numpy as np


class MKL_SVM():

    def __init__(self, dataset_id, k_l, m_l, max_iter=10):

        self.dataset_id = dataset_id
        self.train_gram = None
        self.val_gram = None
        self.test_gram = None
        self.X = None
        self.Y = None
        assert (dataset_id in [0, 1, 2])
        assert(len(k_l)==len(m_l))
        self.create_data()
        self.res = None
        self.training_acc = None
        self.k_l = k_l
        self.m_l = m_l
        self.mismatch_matrice = None
        self.eta =  None
        self.max_iter = max_iter
        self.mism_l = None
        self.acc_val=0

    def create_data(self):
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
        nb_k = len(self.k_l)
        mism_l = []
        for i in range(nb_k):
            k = self.k_l[i]
            m = self.m_l[i]
            path = 'mismatch_matrices/data{}_k{}_m{}.npz'.format(self.dataset_id, k, m)
            if os.path.isfile(path):
                print("Loading mismatch matrix")
                mism = sparse.load_npz(path)
                if norm:
                    mism= sparse.csr_matrix(
                        mism / (sparse.linalg.norm(mism, axis=1)[:, None]))

                mism_l.append(mism)
            else:
                print("Computing mismatch matrix")
                mism = mismatch_matrix(self.X, k, m)
                sparse.save_npz(path, mism)
                mism = sparse.load_npz(path)
                if norm:
                    mism= sparse.csr_matrix(
                        mism / (sparse.linalg.norm(mism, axis=1)[:, None]))

                mism_l.append(mism)

        self.mism_l = mism_l


    def svm_accuracy(self, mism_train=None, mism_val=None, y_train=None,y_val=None, lmbda =3):
#        y = self.Y
#        y_train=y[:1600]
#        y_val=y[1600:]
        self.eta = (1 / len(self.k_l)) * np.ones((len(self.k_l)))
        self.acc_val=0
#        mism_train=[m[:1600,:] for m in self.mism_l]
        kernel_train_l=[m@m.T for m in mism_train]

 #       mism_val=[m[1600:,:] for m in self.mism_l]

        kernel_val_l=[m_val@m.T for (m_val,m) in zip(mism_val,mism_train)]

        norms = np.empty(len(self.k_l))

        for it in range(self.max_iter):

            K = np.sum([eta * k for eta, k in zip(self.eta, kernel_train_l)])

            n = K.shape[0]
            C = 1 / (2 * lmbda * n)
            # Construct the problem.
            x = cp.Variable(n)
            objective = cp.Minimize((1 / 2) * cp.quad_form(x, K) - x.T @ y_train)
            constraints = [0 <= cp.multiply(y_train, x), cp.multiply(y_train, x) <= C]
            prob = cp.Problem(objective, constraints)
            # The optimal objective value is returned by `prob.solve()`.
            prob.solve()
            # The optimal value for x is stored in `x.value`.
            # The optimal Lagrange multiplier for a constraint is stored in
            res = x.value
            self.res = res
            #      pd.Series(res).to_csv('res.csv')

            for i, (eta, k) in enumerate(zip(self.eta, kernel_train_l)):
                norms[i] = (eta ** 2) * res @ (k @ res).T
            norms = np.sqrt(norms)
            scaling = np.sum(norms)
            etas = norms / scaling
            etas[etas < 1e-6 * etas.max()] = 0
            self.eta = etas

            #print(self.eta)
            f = K @ res
            pred = np.where(f >= 0, 1, -1)
            score = pred * y_train
            acc_train = score[score == 1].sum() / len(score.T)
 #           print('train : {}'.format(acc_train))


            K_val = np.sum([eta * k for eta, k in zip(self.eta, kernel_val_l)])
            #K_val = np.sum([eta * k for eta, k in zip(etas, kernel_val_l)])
            f = K_val @ res
            pred = np.where(f >= 0, 1, -1)
            score = pred * y_val
            self.acc_val = score[score == 1].sum() / len(score.T)


 #           print('val : {}'.format(self.acc_val))

        return self.acc_val, acc_train, self.eta

    def lmbda_Nfold(self, N_fold, lmbda_l, norm=False):
        res_tr=pd.DataFrame(index=lmbda_l, columns=range(N_fold))
        res_val=pd.DataFrame(index=lmbda_l, columns=range(N_fold))
        res_eta=pd.DataFrame(index=lmbda_l, columns=range(N_fold))

        tot_card=(self.X).shape[0]
        assert(tot_card % N_fold==0)
        val_card = tot_card// N_fold

        self.compute_mismatch_matrice_multi(norm=norm)


        for i in range(N_fold)[-1:]:

            m_train, m_val = [sparse.vstack([ m[ : val_card * i, :],m[val_card * (i+1) :,:]  ])  for m in self.mism_l ], \
                             [m[val_card * i : val_card * (i + 1),:] for m in self.mism_l]

            Y_train, Y_val = np.hstack((self.Y[ : val_card * i],self.Y[val_card * (i+1) :] )).squeeze() , \
                             self.Y[val_card * i : val_card * (i + 1)]

            for lmbda in lmbda_l:
                acc_val, acc_tr, eta=self.svm_accuracy(m_train, m_val, Y_train, Y_val, lmbda)
                res_tr.loc[lmbda, i] = acc_tr
                res_val.loc[lmbda, i] = acc_val
                res_eta.loc[lmbda, i] = eta
            print("Fold {}".format(i))
        return res_val, res_tr, res_eta


    def fit(self, lmbda):

        y_train = self.Y
        self.eta = (1 / len(self.k_l)) * np.ones((len(self.k_l)))
        self.acc_val = 0

        mism_train=self.mism_l
        kernel_train_l = [m @ m.T for m in mism_train]

        norms = np.empty(len(self.k_l))

        for it in range(self.max_iter):

            K = np.sum([eta * k for eta, k in zip(self.eta, kernel_train_l)])

            n = K.shape[0]
            C = 1 / (2 * lmbda * n)
            # Construct the problem.
            x = cp.Variable(n)
            objective = cp.Minimize((1 / 2) * cp.quad_form(x, K) - x.T @ y_train)
            constraints = [0 <= cp.multiply(y_train, x), cp.multiply(y_train, x) <= C]
            prob = cp.Problem(objective, constraints)
            # The optimal objective value is returned by `prob.solve()`.
            prob.solve()
            # The optimal value for x is stored in `x.value`.
            # The optimal Lagrange multiplier for a constraint is stored in
            res = x.value
            self.res = res
            #      pd.Series(res).to_csv('res.csv')

            for i, (eta, k) in enumerate(zip(self.eta, kernel_train_l)):
                norms[i] = (eta ** 2) * res @ (k @ res).T
            norms = np.sqrt(norms)
            etas = norms / np.sum(norms)
            etas[etas < 1e-6 * etas.max()] = 0
            self.eta = etas

            # print(self.eta)
            f = K @ res
            pred = np.where(f >= 0, 1, -1)
            score = pred * y_train
            self.training_acc = score[score == 1].sum() / len(score.T)

    def predict(self, X_test):
        assert (self.training_acc != None)
        mism_test = mismatch_matrix(X_test, self.k, self.m)
        K_test = mism_test @ self.mismatch_matrice.T
        f = K_test @ self.res
        pred = np.where(f >= 0, 1, -1)
        return pred

    def predict_multi(self, X_test):
        assert (self.training_acc != None)
        nb_k = len(self.k_l)
        mism_matr_l = []
        for i in range(nb_k):
            k = self.k_l[i]
            m = self.m_l[i]
            mism = mismatch_matrix(X_test, k, m)
            mism_matr_l.append(mism)
        #mism_test = sparse.hstack(mism_matr_l, format="csr")
        kernel_val_l=[m_val@m.T for (m_val,m) in zip(mism_matr_l,self.mism_l)]
        K_val = np.sum([eta * k for eta, k in zip(self.eta, kernel_val_l)])
        f = K_val @ self.res
        pred = np.where(f >= 0, 1, -1)

        return pred


if __name__ == '__main__':
    mism = MKL_SVM(dataset_id=0, k_l=[8,8, 7,7,7, 6,6,6, 5,5,5], m_l=[ 1, 0, 2, 1, 0,2, 1, 0,2, 1, 0,], max_iter=5)
#   mism.compute_mismatch_matrice_multi()
 #   mism.svm_accuracy(lmbda=3)
    #lmbda_l = [1 + 0.2 * k for k in range(-2, 2)]
    #lmbda_l = [1e-2 , 0.5, 1, 1.1, 1.5, 2, 10]
    lmbda_l = [0.1, 0.3]
    res_val, res_tr, res_eta=mism.lmbda_Nfold(N_fold=5, lmbda_l=lmbda_l, norm=False)

   # res_val.to_csv('val_data3{}_k{}_m{}.csv'.format(mism.dataset_id, 8765, 0))
    res_eta.to_csv('eta_data{}_k{}_m{}.csv'.format(mism.dataset_id, 8765, 0))

    print(res_tr)
    #print(res_eta.round(2))
    print(res_val)
    print(res_val.mean(axis=1))

""""        pred_list=[]

        X_0_test = pd.read_csv('data/Xte0.csv', index_col=0).reset_index(drop=True)

        mism=MKL_SVM(dataset_id=0, k_l=[8,8,8, 7,7,7, 6,6,6, 5,5,5], m_l=[2, 1, 0, 2, 1, 0,2, 1, 0,2, 1, 0], max_iter=5)
        mism.compute_mismatch_matrice_multi(norm=False)
        mism.fit(lmbda=0.92)
        print(mism.training_acc)
        pred_0=mism.predict_multi(X_0_test)
        pred_list.append(pred_0)

        X_1_test = pd.read_csv('data/Xte1.csv', index_col=0).reset_index(drop=True)

        mism = MKL_SVM(dataset_id=1, k_l=[8,8,8, 7,7,7, 6,6,6, 5,5,5], m_l=[2, 1, 0, 2, 1, 0,2, 1, 0,2, 1, 0], max_iter=5)
        mism.compute_mismatch_matrice_multi(norm=False)
        mism.fit(lmbda=1.3)
        print(mism.training_acc)
        pred_1 = mism.predict_multi(X_1_test)
        pred_list.append(pred_1)

        X_2_test = pd.read_csv('data/Xte2.csv', index_col=0).reset_index(drop=True)

        mism = MKL_SVM(dataset_id=2, k_l=[8,8,8, 7,7,7, 6,6,6, 5,5,5], m_l=[2, 1, 0, 2, 1, 0,2, 1, 0,2, 1, 0], max_iter=5)
        mism.compute_mismatch_matrice_multi(norm=False)
        mism.fit(lmbda=0.8)
        print(mism.training_acc)
        pred_2 = mism.predict_multi(X_2_test)
        pred_list.append(pred_2)

        to_submit = pd.DataFrame(np.concatenate(pred_list)).reset_index().rename(columns={'index': 'Id', 0: 'Bound'})
        to_submit.loc[to_submit.Bound == -1, 'Bound'] = 0
        to_submit.to_csv('to_submit_9.csv', sep=',', index=False, header=True) """