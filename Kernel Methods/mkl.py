from mismatch_kernel import mismatch_matrix
import os.path
import scipy as sp
from scipy import sparse
import pandas as pd
import cvxpy as cp
import numpy as np



class MismatchSVM():

    def __init__(self, dataset_id, k, m, k_l, m_l):

        self.dataset_id=dataset_id
        self.train_gram=None
        self.val_gram=None
        self.test_gram=None
        self.X=None
        self.Y=None
        assert(dataset_id in [0,1,2])
        self.create_data()
        self.res=None
        self.training_acc=None
        self.k_l=k_l
        self.m_l=m_l
        self.mismatch_matrice=None
        self.eta=(1/len(k_l))*np.ones((len(k_l), 1))

    def create_data(self):
        if self.dataset_id==0:
            X = pd.read_csv('data/Xtr0.csv', index_col=0)
            Y = pd.read_csv('data/Ytr0.csv', index_col=0)

        elif self.dataset_id==1:
            X = pd.read_csv('data/Xtr1.csv', index_col=0)
            Y = pd.read_csv('data/Ytr1.csv', index_col=0)

        else :
            X = pd.read_csv('data/Xtr2.csv', index_col=0)
            Y = pd.read_csv('data/Ytr2.csv', index_col=0)

        Y.loc[Y.Bound == 0, 'Bound'] = -1
        self.X=X
        self.Y=Y.values.squeeze()

    def compute_mismatch_matrice_multi(self):
        nb_k=len(self.k_l)
        mism_matr_l=[]
        for i in range(nb_k):
            k=self.k_l[i]
            m=self.m_l[i]
            path = 'mismatch_matrices/data{}_k{}_m{}.npz'.format(self.dataset_id, k, m)
            if os.path.isfile(path):
                print("Loading mismatch matrix")
                mism = sparse.load_npz(path)
                mism= np.sqrt(self.eta[i])*mism

                mism_matr_l.append(mism)
            else:
                print("Computing mismatch matrix")
                mism = mismatch_matrix(self.X, k, m)
                sparse.save_npz(path, mism)
                mism= np.sqrt(self.eta[i])*mism
                mism_matr_l.append(mism)

        self.mismatch_matrice = sparse.hstack(mism_matr_l, format="csr")

    def svm_accuracy(self, mismatch_train, mismatch_val, y, y_val, lmbda):

        K = mismatch_train @ mismatch_train.T
        K_val = mismatch_val @ mismatch_train.T
  #      pd.DataFrame(K_val.toarray()).to_csv('K_val.csv')
   #     pd.DataFrame(K.toarray()).to_csv('K.csv')

        n = mismatch_train.shape[0]
        C = 1 / (2 * lmbda * n)
        # Construct the problem.
        x = cp.Variable(n)
        objective = cp.Minimize((1 / 2) * cp.quad_form(x, K) - x.T @ y)
        constraints = [0 <= cp.multiply(y, x), cp.multiply(y, x) <= C]
        prob = cp.Problem(objective, constraints)

        # The optimal objective value is returned by `prob.solve()`.
        prob.solve()
        # The optimal value for x is stored in `x.value`.
        # The optimal Lagrange multiplier for a constraint is stored in

        res = x.value
  #      pd.Series(res).to_csv('res.csv')

        f = K @ res
        pred = np.where(f >= 0, 1, -1)
        score = pred * y
        acc_train = score[score == 1].sum() / len(score.T)

        f_val = K_val @ res

 #       pd.Series(f_val).to_csv('f_val.csv')

        pred_val = np.where(f_val >= 0, 1, -1)
  #      pd.Series(pred_val).to_csv('pred_val.csv')

        score_val = pred_val * y_val
   #     pd.Series(score_val).to_csv('score.csv')
        acc_val = score_val[score_val == 1].sum() / len(score_val.T)

        return acc_train, acc_val

    def lmbda_Nfold(self, N_fold, lmbda_l):
        res_tr=pd.DataFrame(index=lmbda_l, columns=range(N_fold))
        res_val=pd.DataFrame(index=lmbda_l, columns=range(N_fold))

        tot_card=self.mismatch_matrice.shape[0]
        assert(tot_card % N_fold==0)
        val_card = tot_card// N_fold

        for i in range(N_fold):

            m_train, m_val = sparse.vstack([ self.mismatch_matrice[ : val_card * i, :],self.mismatch_matrice[val_card * (i+1) :,:]   ]) , \
                             self.mismatch_matrice[val_card * i : val_card * (i + 1),:]

            Y_train, Y_val = np.hstack((self.Y[ : val_card * i],self.Y[val_card * (i+1) :] )).squeeze() , \
                             self.Y[val_card * i : val_card * (i + 1)]

            for lmbda in lmbda_l:
                acc_tr, acc_val=self.svm_accuracy(m_train, m_val, Y_train, Y_val, lmbda)
                res_tr.loc[lmbda, i] = acc_tr
                res_val.loc[lmbda, i] = acc_val
            print("Fold {}".format(i))
        return res_val, res_tr


    def fit(self, lmbda):

        K = self.mismatch_matrice @ self.mismatch_matrice.T
        n = self.mismatch_matrice.shape[0]
        y = self.Y
        C = 1 / (2 * lmbda * n)
        # Construct the problem.
        x = cp.Variable(n)
        objective = cp.Minimize((1 / 2) * cp.quad_form(x, K) - x.T @ y)
        constraints = [0 <= cp.multiply(y, x), cp.multiply(y, x) <= C]
        prob = cp.Problem(objective, constraints)

        # The optimal objective value is returned by `prob.solve()`.
        prob.solve()
        # The optimal value for x is stored in `x.value`.
        # The optimal Lagrange multiplier for a constraint is stored in

        res = x.value
        f = K @ res
        pred = np.where(f >= 0, 1, -1)
        score = pred * self.Y
        acc_train = score[score == 1].sum() / len(score.T)
        self.res=res
        self.training_acc=acc_train
        print("Fit completed")

    def predict(self, X_test):
        assert(self.training_acc!=None)
        mism_test=mismatch_matrix(X_test,self.k , self.m)
        K_test=mism_test @ self.mismatch_matrice.T
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
        mism_test = sparse.hstack(mism_matr_l, format="csr")
        K_test=mism_test @ self.mismatch_matrice.T
        f = K_test @ self.res
        pred = np.where(f >= 0, 1, -1)
        return pred

if __name__ == '__main__':
    mism=MismatchSVM(dataset_id=0, k=5, m=1,k_l=[8,7], m_l=[2,0])
    mism.compute_mismatch_matrice_multi(norm=False, if_weight=True, weight=[0.7, 0.3])
    #mism.compute_mismatch_matrice(norm=False)

    #lmbda_l = [1 + 0.1 *k for k in range(-2, 2)]
    lmbda_l = [0.8+ 0.1 *k for k in range(-3, 3)]
    N_fold=5
    res_val, res_tr=mism.lmbda_Nfold(N_fold, lmbda_l)
#    res_val.to_csv('lmbda_data{}_k{}_m{}.csv'.format(mism.dataset_id, 87, 0))

    print(res_tr)
    print(res_val)
    print(res_val.mean(axis=1))
