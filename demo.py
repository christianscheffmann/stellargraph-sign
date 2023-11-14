from stellargraph import datasets
from stellargraphtest.layer.SIGN import SIGN
from stellargraphtest.mapper.SIGNNodeGenerator import SIGNNodeGenerator
from stellargraph.layer import DeepGraphInfomax
from stellargraph.mapper import CorruptedGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model

dataset = datasets.Cora()
G, node_subjects = dataset.load()

generator = SIGNNodeGenerator(G, (2, 1, 1))

sign_model = SIGN(4, 16, 4, generator)

z = generator.flow(G.nodes())

corrupted_generator = CorruptedGenerator(generator)
gen = corrupted_generator.flow(G.nodes())

infomax = DeepGraphInfomax(sign_model, corrupted_generator)
x_in, x_out = infomax.in_out_tensors()

model = Model(inputs=x_in, outputs=x_out)
model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=Adam(lr=1e-3))

epochs = 100

es = EarlyStopping(monitor="loss", min_delta=0, patience=20)
history = model.fit(gen, epochs=epochs, verbose=0, callbacks=[es])