from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda, Input, Dense, BatchNormalization, ReLU, Dropout
from tqdm import tqdm
from stellargraph.layer.misc import GatherIndices

from stellargraph_sign.stellargraph.mapper.SIGNNodeGenerator import SIGNNodeGenerator

import tensorflow as tf
from tensorflow.keras import Model

class SIGN():
    def __init__(self, hidden_layers, hidden_layer_channels, out_channels, generator, bias=True, dropout=0.0):
        self.hidden_layers = hidden_layers
        self.hidden_layer_channels = hidden_layer_channels
        self.out_channels = out_channels
        self.generator = generator
        self.bias = bias
        self.dropout = dropout
        
        if not isinstance(generator, SIGNNodeGenerator):
            raise TypeError(f"Generator should be instance of SIGNNodeGenerator")
        
        self.r = sum(generator.operators)
        self.n_features = generator.features.shape[1]
        self.use_sparse = generator.use_sparse
        self.n_nodes = generator.features.shape[0]
    
    def __call__(self, x):
        "Apply SIGN layers to the precomputed operators"
        layer_out, out_indices = x
        for _ in range(self.hidden_layers - 1):
            dense_out = Dense(self.hidden_layer_channels)(layer_out)
            norm_out = BatchNormalization(axis=-1)(dense_out)
            relu_out = ReLU()(norm_out)
            layer_out = Dropout(self.dropout)(relu_out)
        
        dense_final_out = Dense(self.out_channels)(layer_out)
        softmax_out = tf.keras.layers.Softmax()(dense_final_out)
        
        out_layer = GatherIndices(batch_dims=1)([softmax_out, out_indices])
        
        return out_layer

    def in_out_tensors(self):
        x_inp = [Input(batch_shape=(1, self.n_nodes, (self.r+1)*self.n_features)), Input(batch_shape=(1, None), dtype="int32")]
        x_out = self(x_inp)
        
        return x_inp, x_out
        
