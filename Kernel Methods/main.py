from mismatch_kernel import mismatch_matrix
import os.path
import scipy as sp
from scipy import sparse
import pandas as pd
import cvxpy as cp
import numpy as np



class MismatchSVM():

    def __init__(self, dataset_id, k, m):
        self.k = k
        self.m = m
        self.dataset_id=dataset_id
        self.train_gram=None
        self.val_gram=None
        self.test_gram=None
        self.X=None
        self.Y=None
        self.mismatch_matrice=None
        assert(dataset_id in [0,1,2])
        self.create_data()
        self.res=None
        self.training_acc=None

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

    def compute_mismatch_matrice(self, norm=False):
        path= 'mismatch_matrices/data{}_k{}_m{}.npz'.format(self.dataset_id, self.k, self.m)
        if os.path.isfile(path):
            print("Loading mismatch matrix")
            self.mismatch_matrice=sparse.load_npz(path)
        else:
            print("Computing mismatch matrix")
            self.mismatch_matrice=mismatch_matrix(self.X,self.k , self.m)
            sparse.save_npz(path, self.mismatch_matrice)
        if norm:
            self.mismatch_matrice = sparse.csr_matrix(self.mismatch_matrice / (sparse.linalg.norm(self.mismatch_matrice, axis=1)[:, None]))

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

if __name__ == '__main__':
    mism=MismatchSVM(dataset_id=1, k=8, m=2)
    mism.compute_mismatch_matrice(norm=False)
    lmbda_l = [1+ 0.1*k for k in range(12)]
    N_fold=5
    res_val, res_tr=mism.lmbda_Nfold(N_fold, lmbda_l)
    res_val.to_csv('lmbda_data{}_k{}_m{}.csv'.format(mism.dataset_id, mism.k, mism.m))
    print(res_tr)
    print(res_val)

#    mism_train,mism_val=mism.mismatch_matrice[:1600,:], mism.mismatch_matrice[1600:,:]
#    y_train,y_val=mism.Y[:1600], mism.Y[1600:]
 #   print(mism.svm_accuracy(mism_train,mism_val , y_train, y_val, 1.2))


