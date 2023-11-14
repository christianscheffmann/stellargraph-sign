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
    
    
def normalize_adj(adj, symmetric=True, add_self_loops=False):
    if add_self_loops:
        adj = adj + sp.diags(np.ones(adj.shape[0]) - adj.diagonal())
        
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.float_power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj.tocsr()
    

def GCN_Aadj_feats_op():


def PPNP_Aadj_feats_op(A, teleport_probability=0.1):
    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)
    A = A + sp.diags(np.ones(A.shape[0]) - A.diagonal())
    A = normalize_adj(A, symmetric=True)
    A = sp.identity(A.shape[0]) - A.multiply(1 - teleport_probability)
    


