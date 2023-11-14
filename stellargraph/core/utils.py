from stellargraph.core.utils import normalize_adj, PPNP_Aadj_feats_op, GCN_Aadj_feats_op
import numpy as np
from scipy.sparse.linalg import expm, inv
from scipy.sparse import csr_matrix


def sparsify_top_k(S, k):
    # Not implemented
    return S


def sparsify_threshold_epsilon(S, epsilon):
    S[np.abs(S) < epsilon] = 0
    return csr_matrix(S)


def GCN_Aadj(A):
    S = GCN_Aadj_feats_op(None, A)[1]
    return normalize_adj(sparsify_threshold_epsilon(S, 1e-4))
    
    
def PPNP_Aadj(A, p):
    S = PPNP_Aadj_feats_op(None, A, p)[1]
    return normalize_adj(sparsify_threshold_epsilon(S, 1e-4))
    
def HK_Aadj(A, t):
    A_norm = normalize_adj(A, symmetric=False)
    S = np.exp(-t)*expm(t*A_norm.tocsc()).tocsr()
    return normalize_adj(sparsify_threshold_epsilon(S, 1e-4))
