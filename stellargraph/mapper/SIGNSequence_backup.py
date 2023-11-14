from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
import scipy.sparse as sps
import numpy as np
from stellargraph_sign.stellargraph.core.utils import GCN_Aadj, PPNP_Aadj, HK_Aadj
from tqdm import tqdm

def _full_batch_array_and_reshape(array, propagate_none=False):
    if propagate_none and array is None:
        return None
        
    as_np = np.asanyarray(array)
    return np.reshape(as_np, (1,) + as_np.shape)


def _calculate_adj(A, i, alpha = 0.3, t = 0.5):
    if i == 0:
        return GCN_Aadj(A)
    elif i == 1:
        return PPNP_Aadj(A, alpha)
    elif i == 2:
        return HK_Aadj(A, t)
    else:
        raise Exception("Undefined operator")

def _construct_operators(operators, A, features):
    x = features
        
    layer_ops = [x]
    
    for i, j in enumerate(operators):
        x = features
        adj = _calculate_adj(A, i)

        for _ in tqdm(range(j)):
            x = adj @ x
            layer_ops += [x]
        
    return layer_ops

class SIGNSequence(Sequence):

    use_sparse = False
    
    def __init__(self, features, A, operators, targets=None, indices=None):
        
        if (targets is not None) and (len(indices) != len(targets)):
            raise ValueError("When passed together targets and indices should be the same length.")
        
        self.features = np.asanyarray(features)
        self.target_indices = np.asanayarray(indices)
        
        if sps.issparse(A) and hasattr(A, "toarray"):
            self.A_dense = _full_batch_array_and_reshape(A.toarray())
        elif isinstance(A, (np.ndarray, np.matrix)):
            self.A_dense = _full_batch_array_and_reshape(A)
        else:
            raise TypeError("Expected input matrix to be either a Scipy sparse matrix or a Numpy array.")
        
        self.features = _full_batch_array_and_reshape(features)
        self.target_indices = _full_batch_array_and_reshape(indices)
        self.inputs = [np.concatenate(_construct_operators(operators, A, features), axis=-1)[np.newaxis,...]]
        self.targets = _full_batch_array_and_reshape(targets, propagate_none=True)
        
    def __len__(self):
        return 1
        
    def __getitem__(self, index):
        return self.inputs, self.targets

class SparseSIGNSequence(Sequence):

    use_sparse = False
    
    def __init__(self, features, A, operators, targets=None, indices=None):
        
        if (targets is not None) and (len(indices) != len(targets)):
            raise ValueError("When passed together targets and indices should be the same length.")
        
        if sps.isspmatrix(A):
            A = A.tocoo()
        
        else:
            raise ValueError("Adjacency matrix not in expected sparse format.")

        self.features = _full_batch_array_and_reshape(features)
        self.target_indices = _full_batch_array_and_reshape(indices)
        self.inputs = [np.concatenate(_construct_operators(operators, A, features), axis=-1)[np.newaxis,...]]
        self.targets = _full_batch_array_and_reshape(targets, propagate_none=True)
        
    def __len__(self):
        return 1
        
    def __getitem__(self, index):
        return self.inputs, self.targets
        
class SparseBatchSIGNSequence(Sequence):

    use_sparse = False
    
    def __init__(self, features, A, operators, batch_size, targets=None, indices=None):
        
        self.batch_size = batch_size
        
        if (targets is not None) and (len(indices) != len(targets)):
            raise ValueError("When passed together targets and indices should be the same length.")
        
        if sps.isspmatrix(A):
            A = A.tocoo()
        
        else:
            raise ValueError("Adjacency matrix not in expected sparse format.")

        self.features = _full_batch_array_and_reshape(features)
        self.target_indices = _full_batch_array_and_reshape(indices)
        self.inputs = [np.concatenate(_construct_operators(operators, A, features), axis=-1)[np.newaxis,...]]
        self.targets = _full_batch_array_and_reshape(targets, propagate_none=True)
        
    def __len__(self):
        return int(np.ceil(self.data_size / self.batch_size))
        
    def __getitem__(self, batch_num):
        start_idx = self.batch_size * batch_num
        end_idx = start_idx + self.batch_size
        
        if start_idx >= self.data_size:
            raise IndexError()
        
        return self.inputs, self.targets
        
        
        
        
        
        
