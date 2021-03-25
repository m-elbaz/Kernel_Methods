import pandas as pd
import numpy as np
from mismatch import MismatchSVM


def make_prediction(dataset_id, k_l, m_l, lmbda):
    X_test = pd.read_csv('data/Xte{}.csv'.format(dataset_id), index_col=0).reset_index(drop=True)
    mism = MismatchSVM(dataset_id=dataset_id, k_l=k_l, m_l=m_l)
    mism.compute_mismatch_matrice_multi(norm=False)
    mism.fit(lmbda=lmbda)
    pred = mism.predict_multi(X_test)
    return pred


if __name__ == '__main__':
    pred_list = []

    pred_0 = make_prediction(dataset_id=0, k_l=[8, 7], m_l=[2, 0], lmbda=1)
    pred_list.append(pred_0)
    print('Dataset_0 done !')

    pred_1 = make_prediction(dataset_id=1, k_l=[8, 5], m_l=[2, 1], lmbda=1.7)
    pred_list.append(pred_1)
    print('Dataset_1 done !')

    pred_2 = make_prediction(dataset_id=2, k_l=[8, 5], m_l=[2, 1], lmbda=1)
    pred_list.append(pred_2)
    print('Dataset_2 done !')

    to_submit = pd.DataFrame(np.concatenate(pred_list)).reset_index().rename(columns={'index': 'Id', 0: 'Bound'})
    to_submit.loc[to_submit.Bound == -1, 'Bound'] = 0
    to_submit.to_csv('final_submission.csv', sep=',', index=False, header=True)
    print('Labels successfully generated !')
