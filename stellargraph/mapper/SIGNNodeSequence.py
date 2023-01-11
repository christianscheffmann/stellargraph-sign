from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
from tqdm import tqdm

from stellargraph.mapper import FullBatchGenerator


class SIGNNodeSequence(FullBatchGenerator):
    def __init__(self, G, operators, name=None, method="gcn", k=1, sparse=True, transform=None, teleport_probability=0.1, weighted=False):
        # operators should be tuple of (p, s, t) from SIGN paper
        if not isinstance(operators, tuple) and len(operators) != 3:
            raise TypeError("Variable 'operators' must be a tuple of elements (p, s, t)")
             
        self.operators = operators
        super().__init__()
        
        
        x = self.generator.features        
        self.layer_ops = [x]
        
        # Implement dropout?
        # Implement other forms/normalizations of A?
        for _ in tqdm(range(self.n_layers)):
            x = self.generator.Aadj @ x
            self.layer_ops += [x]
