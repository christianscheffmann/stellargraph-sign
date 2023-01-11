from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
from tqdm import tqdm

from stellargraph.mapper import CorruptedGenerator, FullBatchNodeGenerator, GraphSAGENodeGenerator

from stellargraph import StellarGraph
from stellargraph.layer import DeepGraphInfomax, GraphSAGE

from stellargraph import datasets
from stellargraph.utils import plot_history

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from IPython.display import display, HTML

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import Model

class SIGN():
    def __init__(self, layer_sizes, hidden_layers, num_classes, generator, bias=True, dropout=0.0, normalize="l2"):
        # layer_sizes should be tuple of (p, s, t) from SIGN paper
        self.layer_sizes = layer_sizes
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.generator = generator
        self.n_layers = len(layer_sizes)
        self.r = self.n_layers
        self.bias = bias
        self.dropout = dropout
        self.num_layers = 3
                
        if normalize == "l2":
            self._normalization = Lambda(lambda x: K.l2_normalize(x, axis=-1))
        
        if not isinstance(generator, FullBatchNodeGenerator):
            raise TypeError(f"Generator should be instance of FullBatchNodeGenerator")
        
        self.method = generator.method
        self.multiplicity = generator.multiplicity
        self.n_features = generator.features.shape[1]
        self.use_sparse = generator.use_sparse
        if isinstance(generator, FullBatchNodeGenerator):
            self.n_nodes = generator.features.shape[0]
        else:
            self.n_nodes = None

        x = self.generator.features        
        self.layer_ops = [x]
        
        # Implement dropout?
        # Implement other forms/normalizations of A?
        for _ in tqdm(range(self.n_layers)):
            x = self.generator.Aadj @ x
            self.layer_ops += [x]
                
    
    def __call__(self, x):
        "Apply SIGN layers to the precomputed operators"
        layer_out = x[0]
        for _ in range(self.num_layers - 1):
            dense_out = tf.keras.layers.Dense(self.hidden_layers)(layer_out)
            norm_out = tf.keras.layers.BatchNormalization(axis=-1)(dense_out)
            relu_out = tf.keras.layers.ReLU()(norm_out)
            layer_out = tf.keras.layers.Dropout(self.dropout)(relu_out)
        
        dense_final_out = tf.keras.layers.Dense(self.num_classes)(layer_out)
        return tf.keras.layers.Softmax()(dense_final_out)
    
        #"Apply SIGN layers to the precomputed operators"
        #model = tf.keras.Sequential()
        #model.add(tf.keras.layers.Input(shape=(x.shape[-1])))
        
        #for _ in range(self.num_layers - 1):
        #    model.add(tf.keras.layers.Dense(self.hidden_layers))
        #    model.add(tf.keras.layers.BatchNormalization(axis=-1))
        #    model.add(tf.keras.layers.ReLU())
        #    model.add(tf.keras.layers.Dropout(self.dropout))
        
        #model.add(tf.keras.layers.Dense(self.num_classes))
        
        #return tf.nn.log_softmax(model(x))

    def in_out_tensors(self):
        x_inp = [tf.keras.layers.Input(batch_shape=(1, self.n_nodes, (self.r+1)*self.n_features))]
        x_out = self(x_inp)
        
        return x_inp, x_out
        
