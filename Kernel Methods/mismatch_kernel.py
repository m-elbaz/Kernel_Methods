import numpy as np
import itertools
import pandas as pd
import scipy as sp
from scipy import special
from scipy.sparse import vstack


def apply_star(x, ind_l):
    """ Insert * in x at indices ind_l """
    l = list(x)
    for ind in ind_l:
        l[ind] = "*"
    return "".join(l)

def mismatch_matrix(data, k, m):
    """ Computes the feature matrix for given dataset and given kernel params (k,m) """

    # Computes the size of the alphabet
    dict_card = (4 ** (k - m)) * sp.special.comb(k, m)

    #Slices  all sequences to k-mers
    all_seq = pd.concat([data.seq.str.slice(i, i + k) for i in range(101 - k + 1)], axis=1)
    #Renames columns
    all_seq.columns = range(len(all_seq.columns))

    # Expanding the data by inserting m mismatches represanted by '*'
    ind_l = list(itertools.combinations(list(range(k)), m))
    star_seq = pd.concat([all_seq.applymap(lambda x: apply_star(x, ind)) for ind in ind_l], axis=1)

    # Constructs the alphabet of all possible k_mers with mismatches
    keywords = [''.join(i) for i in itertools.product(['A', 'C', 'G', 'T'], repeat=k)]
    keywords = pd.Series(keywords)
    keywords = pd.concat([keywords.apply(lambda x: apply_star(x, ind)) for ind in ind_l],
                         axis=0).drop_duplicates().reset_index(drop=True)
    assert (len(keywords) == dict_card)
    keywords1 = pd.Series(keywords.index, index=keywords)

    # Crucial step : reindex the data buy the alphabet indices
    all_seq_id = star_seq.applymap(keywords1.get)
    res_l = []

    # Counts the number of occurences and agregates them in a single matrix
    for i in all_seq_id.index:
        sample = all_seq_id.loc[i].value_counts()
        sample = sample.reindex(keywords.index).fillna(0)
        res_l.append(sp.sparse.csr_matrix(sample))

    # Outputs the feature matrix of size Nb_samples * Alphabet_size
    resultat = vstack(res_l)

    return resultat
