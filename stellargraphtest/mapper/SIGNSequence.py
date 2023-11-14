from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
import scipy.sparse as sps
import numpy as np
from tqdm import tqdm

def _full_batch_array_and_reshape(array, propagate_none=False):
    if propagate_none and array is None:
        return None
        
    as_np = np.asanyarray(array)
    return np.reshape(as_np, (1,) + as_np.shape)

class SIGNSequence(Sequence):

    use_sparse = False
    
    def __init__(self, inputs, targets=None, indices=None):
        
        if (targets is not None) and (len(indices) != len(targets)):
            raise ValueError("When passed together targets and indices should be the same length.")
        
        self.target_indices = _full_batch_array_and_reshape(indices)
        self.inputs = [inputs, self.target_indices]
        self.targets = _full_batch_array_and_reshape(targets, propagate_none=True)
        
    def __len__(self):
        return 1
        
    def __getitem__(self, index):
        return self.inputs, self.targets

class BatchSIGNSequence(Sequence):
    
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
        self.inputs = [np.concatenate(_construct_operators(operators, A, features), axis=-1)[np.newaxis,...], self.target_indices]
        self.targets = _full_batch_array_and_reshape(targets, propagate_none=True)
        
    def __len__(self):
        return int(np.ceil(self.data_size / self.batch_size))
        
    def __getitem__(self, batch_num):
        start_idx = self.batch_size * batch_num
        end_idx = start_idx + self.batch_size
        
        if start_idx >= self.data_size:
            raise IndexError()
        
        return self.inputs, self.targets
        
        
        
        
        
        
