import warnings
import numpy as np
from tensorflow.keras import backend as K
from stellargraph.core.graph import StellarGraph
from stellargraph.core.utils import is_real_iterable
from stellargraph.mapper import Generator
from stellargraph_sign.stellargraph.mapper.SIGNSequence import SIGNSequence
from stellargraph_sign.stellargraph.core.utils import GCN_Aadj, PPNP_Aadj, HK_Aadj
import scipy.sparse as sps
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



class SIGNNodeGenerator(Generator):
    multiplicity = 1
    def __init__(self, G, operators, batch_size=None, name=None):
        # operators should be tuple of (p, s, t) from SIGN paper
        if not isinstance(operators, tuple) and len(operators) != 3:
            raise TypeError("Variable 'operators' must be a tuple of elements (p, s, t)")
        
        self.use_sparse = True
        
        self.operators = operators
        
        if not isinstance(G, StellarGraph):
            raise TypeError("Graph must be a StellarGraph of StellarDiGraph object.")
        
        self.graph = G
        self.name = name
        
        G.check_graph_for_ml()
        
        # Check that there is only a single node type
        node_type = G.unique_node_type("G: expected a graph with a single node type, found a graph with node types: %(found)s")
        
        self.node_list = G.nodes()
        A = G.to_adjacency_matrix(weighted=False)
        
        if sps.isspmatrix(A):
            A = A.tocoo()
        
        self.features = G.node_features(node_type=node_type)
        
        self.precomp_ops = np.concatenate(_construct_operators(self.operators, A, self.features), axis=-1)[np.newaxis,...]
        
        
    def num_batch_dims(self):
        return 2
        
    def flow(self, node_ids, batch_size, targets=None, use_ilocs=False):
        if not batch_size:
            return _fullbatch_flow(node_ids, targets, use_ilocs)
        else:
            return _batched_flow(node_ids, batch_size, targets, use_ilocs)

    
    def _get_node_indices(self, node_ids, targets, use_ilocs):
        if targets is not None:
            if not is_real_iterable(targets):
                raise TypeError("Targets must be an iterable or None")
            
            if len(targets) != len(node_ids):
                raise TypeError("Targets must be the same length as node_ids")
            
        node_ids = np.asarray(node_ids)
        if use_ilocs:
            node_indices = node_ids
        else:
            flat_node_ids = node_ids.reshape(-1)
            flat_node_indices = self.graph.node_ids_to_ilocs(flat_node_ids)
            node_indices = flat_node_indices.reshape(node_ids.shape)
        
        return node_indices

    
    def _fullbatch_flow(self, node_ids, targets, use_ilocs):
        node_indices = _get_node_indices(node_ids, targets, use_ilocs)
        return SIGNSequence(self.precomp_ops, targets, node_indices)
    
    def _batched_flow(self, node_ids, batch_size, targets, use_ilocs):
        node_indices = _get_node_indices(node_ids, targets, use_ilocs)
        return SIGNBatchSequence(self.precomp_ops, batch_size, targets, node_indices)
    
    def default_corrupt_input_index_groups(self):
        return [[0]]
