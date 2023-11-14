import warnings
import numpy as np
from tensorflow.keras import backend as K
from stellargraph.core.graph import StellarGraph
from stellargraph.core.utils import is_real_iterable
from stellargraph.mapper import Generator
from stellargraphtest.mapper.SIGNSequence import SIGNSequence, SparseSIGNSequence

class SIGNNodeGenerator(Generator):
    multiplicity = 1
    def __init__(self, G, operators, name=None, sparse=True):
        # operators should be tuple of (p, s, t) from SIGN paper
        if not isinstance(operators, tuple) and len(operators) != 3:
            raise TypeError("Variable 'operators' must be a tuple of elements (p, s, t)")
             
        self.operators = operators
        
        if not isinstance(G, StellarGraph):
            raise TypeError("Graph must be a StellarGraph of StellarDiGraph object.")
            
        self.graph = G
        self.name = name
        
        G.check_graph_for_ml()
        
        # Check that there is only a single node type
        node_type = G.unique_node_type("G: expected a graph with a single node type, found a graph with node types: %(found)s")
        
        self.node_list = G.nodes()
        self.Aadj = G.to_adjacency_matrix(weighted=False)
        
        if sparse and K.backend() != "tensorflow":
            warnings.warn("Sparse adjacency matrices are only supported in tensorflow. Falling back to using a dense adjacency matrix.")
            self.use_sparse = False
        else:
            self.use_sparse = sparse
            
        self.features = G.node_features(node_type=node_type)
        
    def num_batch_dims(self):
        return 2
        
    def flow(self, node_ids, targets=None, use_ilocs=False):
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
        if self.use_sparse:
            return SparseSIGNSequence(self.features, self.Aadj, self.operators, targets, node_indices)
        else:
            return SIGNSequence(self.features, self.Aadj, self.operators, targets, node_indices)
            
    def default_corrupt_input_index_groups(self):
        return [[0]]
