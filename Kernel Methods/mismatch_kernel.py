import numpy as np
import itertools
import pandas as pd
import scipy as sp
from scipy import special
from scipy.sparse import vstack


def apply_star(x, ind_l):
    l = list(x)
    for ind in ind_l:
        l[ind] = "*"
    return "".join(l)


def mismatch_matrix(data,k , m):

    dict_card = (4 ** (k - m)) * sp.special.comb(k, m)

    all_seq = pd.concat([data.seq.str.slice(i, i + k) for i in range(101 - k + 1)], axis=1)
    all_seq.columns = range(len(all_seq.columns))
    ind_l = list(itertools.combinations(list(range(k)), m))
    star_seq = pd.concat([all_seq.applymap(lambda x: apply_star(x, ind)) for ind in ind_l], axis=1)

    keywords = [''.join(i) for i in itertools.product(['A', 'C', 'G', 'T'], repeat=k)]
    keywords = pd.Series(keywords)
    keywords = pd.concat([keywords.apply(lambda x: apply_star(x, ind)) for ind in ind_l],
                         axis=0).drop_duplicates().reset_index(drop=True)
    assert (len(keywords) == dict_card)

    keywords1 = pd.Series(keywords.index, index=keywords)
    all_seq_id = star_seq.applymap(keywords1.get)
    res_l = []

    for i in all_seq_id.index:
        sample = all_seq_id.loc[i].value_counts()
        sample = sample.reindex(keywords.index).fillna(0)
        res_l.append(sp.sparse.csr_matrix(sample))

    resultat = vstack(res_l)

    return resultat
